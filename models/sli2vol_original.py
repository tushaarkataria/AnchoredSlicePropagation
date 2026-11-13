"""
Original Sli2Vol correspondence-based registration wrapped as a backend.

This wraps the Correspondence_Flow_Net from Pak Hei Yeung's Sli2Vol paper
to conform to the BaseRegistrationNet interface. The network classes below are
adapted from the authors' code (Yeung et al., MICCAI 2021,
https://arxiv.org/abs/2105.12722); see LICENSE, "Third-party code". Since Sli2Vol uses
correlation-based soft attention (not a deformation field), the 'deformation'
key in the output is None.

Architecture:
    1. ResNet18 (stride-1, no downsampling) extracts 128-ch features per slice
    2. Post-conv reduces to 64-ch
    3. CorrelationLayer computes local affinity in a (2R+1)x(2R+1) window
    4. Affinity weights used to reconstruct target from source pixels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_registration import BaseRegistrationNet


# ============================================================
# Building blocks (from original Sli2Vol)
# ============================================================

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, activation=F.relu):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )
        self.activation = activation

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet18_original_size(nn.Module):
    """ResNet18 feature extractor that preserves spatial resolution (all stride=1)."""

    def __init__(self, in_ch=1):
        super().__init__()
        self.inchannel = 16
        start_channel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, start_channel, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(ResidualBlock, start_channel, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, start_channel * 2, 2, stride=1)
        self.layer3 = self._make_layer(ResidualBlock, start_channel * 4, 2, stride=1)
        self.layer4 = self._make_layer(ResidualBlock, start_channel * 8, 2, stride=1)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inchannel, channels, s))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5


def one_hot(labels, C):
    """Convert integer labels to one-hot encoding."""
    one_hot_t = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3),
                            device=labels.device)
    target = one_hot_t.scatter_(1, labels, 1)
    return target


class CorrelationLayer(nn.Module):
    """
    Local correlation between two feature maps within a displacement window.

    For each pixel in x_1, computes dot-product similarity with all pixels
    in x_2 within a (2*pad+1) x (2*pad+1) neighborhood. Also extracts
    corresponding patches from the source image for weighted reconstruction.
    """

    def __init__(self, padding=20):
        super().__init__()
        self.pad = padding
        self.max_displacement = padding

    def forward(self, x_1, x_2, img):
        """
        Args:
            x_1: (B, C, H, W) target features
            x_2: (B, C, H, W) source features
            img: (B, C_img, H, W) source image (or one-hot mask)

        Returns:
            corr_scores: (B, (2R+1)^2, H, W) correlation scores
            img_patches:  (B, (2R+1)^2 * C_img, H, W) source image patches
        """
        x_1 = x_1.transpose(1, 2).transpose(2, 3)  # (B, H, W, C)
        x_2 = F.pad(x_2, [self.pad] * 4).transpose(1, 2).transpose(2, 3)
        img = F.pad(img, [self.pad] * 4)

        out_vb = torch.zeros(1, device=x_1.device)
        out_img = torch.zeros(1, device=x_1.device)

        for _y in range(self.max_displacement * 2 + 1):
            for _x in range(self.max_displacement * 2 + 1):
                c_out = torch.sum(
                    x_1 * x_2[:, _y:_y + x_1.size(1), _x:_x + x_1.size(2), :],
                    3, keepdim=True
                ).transpose(2, 3).transpose(1, 2)

                patch = img[:, :, _y:_y + x_1.size(1), _x:_x + x_1.size(2)]

                if len(out_vb.size()) == 1:
                    out_vb = c_out
                    out_img = patch
                else:
                    out_vb = torch.cat((out_vb, c_out), 1)
                    out_img = torch.cat((out_img, patch), 1)

        return out_vb, out_img


class CorrReconstruction(nn.Module):
    """
    Reconstructs the target slice from source using correlation-based soft attention.

    During training: warps raw intensity images.
    During inference: warps one-hot encoded label masks.
    """

    def __init__(self, R=6, is_training=True, temperature=1.0):
        super().__init__()
        self.R = R
        self.is_training = is_training
        self.temperature = temperature
        self.correlation_sampler = CorrelationLayer(padding=self.R)

    def prep(self, image):
        if self.is_training:
            return image.float()
        else:
            x = image.float()
            return one_hot(x.long(), 7).float()

    def forward(self, feats_r, feats_t, img_r, return_corr=False):
        """
        Warp img_r toward img_t using feature similarity.

        Args:
            feats_r: (B, C, H, W) source features
            feats_t: (B, C, H, W) target features
            img_r:   (B, C_img, H, W) source image/mask
            return_corr: if True, also return the softmax correspondence
                weights (B, (2R+1)^2, H, W). Used by training diagnostics.

        Returns:
            img_t: (B, C_img, H, W) reconstructed target
            corr:  (B, (2R+1)^2, H, W) softmax weights [only if return_corr]
        """
        img_r = self.prep(img_r)
        _, c, _, _ = img_r.size()

        corrs, corrs_img = self.correlation_sampler(feats_t, feats_r, img_r)

        corr = F.softmax(corrs / self.temperature, dim=1)

        img_t = []
        for start in range(c):
            img_t.append((corr * corrs_img[:, start::c, :, :]).sum(1, keepdim=True))
        img_t = torch.cat(img_t, 1)

        if return_corr:
            return img_t, corr
        return img_t


# ============================================================
# Edge profile feature extractor
# ============================================================

def edge_profile(img_list, with_image=False, radius=3, interval=1):
    """
    Compute local difference features for each pixel.

    For each pixel, computes the difference to all neighbors in a
    (2*radius+1) x (2*radius+1) window (excluding center), then applies softmax.
    Produces ((2*radius+1)^2 - 1) channels per image.

    Args:
        img_list: list of (B, 1, H, W) tensors
        with_image: if True, concatenate original image as extra channel
        radius: neighborhood radius
        interval: sampling interval within neighborhood

    Returns:
        list of (B, C_out, H, W) tensors where C_out = (2R+1)^2 - 1 [+ 1 if with_image]
    """
    padding = radius * interval
    max_displacement = padding

    img_list_update = []
    for img in img_list:
        x_1 = img.transpose(1, 2).transpose(2, 3)  # (B, H, W, C)
        x_2 = F.pad(img, [padding] * 4, mode='replicate').transpose(1, 2).transpose(2, 3)

        out_vb = torch.zeros(1, device=img.device)
        for _y in range(0, max_displacement * 2 + 1, interval):
            for _x in range(0, max_displacement * 2 + 1, interval):
                if _y == max_displacement and _x == max_displacement:
                    continue
                c_out = torch.sum(
                    x_1 - x_2[:, _x:_x + x_1.size(1), _y:_y + x_1.size(2), :],
                    3, keepdim=True
                ).transpose(2, 3).transpose(1, 2)

                if len(out_vb.size()) == 1:
                    out_vb = c_out
                else:
                    out_vb = torch.cat((out_vb, c_out), 1)

        out_vb = nn.Softmax(dim=1)(out_vb)
        if with_image:
            out_vb = torch.cat([out_vb, x_1.transpose(2, 3).transpose(1, 2)], dim=1)
        img_list_update.append(out_vb)

    return img_list_update


# ============================================================
# Main Sli2Vol model wrapped as a registration backend
# ============================================================

class Sli2VolRegistration(BaseRegistrationNet):
    """
    Original Sli2Vol Correspondence Flow Network wrapped as a registration backend.

    Note: Sli2Vol does NOT produce a deformation field. It uses correlation-based
    soft attention to directly reconstruct the target from the source. Therefore:
      - forward() returns 'deformation': None
      - warp() re-runs the model to warp an image/mask
      - For mask propagation at inference, use propagate_mask() which handles
        the one-hot conversion and argmax internally.
    """

    def __init__(self, in_channels=None, R=6, is_training=True,
                 use_edge_profile=True, edge_radius=3, edge_interval=1,
                 temperature=1.0):
        super().__init__()
        self.use_edge_profile = use_edge_profile
        self.edge_radius = edge_radius
        self.edge_interval = edge_interval
        self.is_training = is_training
        self.temperature = temperature

        # Derive the feature-extractor input channel count from the input
        # representation: edge profile gives (2R+1)^2 - 1 stacked neighbour
        # differences (48 for R=3), raw grayscale gives 1. The `in_channels`
        # arg is an explicit override for both branches; it validates against
        # the edge-profile derivation so a silent mismatch becomes an error.
        if use_edge_profile:
            derived = ((2 * edge_radius) + 1) ** 2 - 1
        else:
            derived = 1

        if in_channels is None:
            model_in_channels = derived
        else:
            if in_channels != derived:
                raise ValueError(
                    f"in_channels={in_channels} does not match the "
                    f"{'edge_profile' if use_edge_profile else 'raw'} "
                    f"representation (expected {derived})."
                )
            model_in_channels = in_channels

        self.feature_extraction = ResNet18_original_size(model_in_channels)
        self.post_convolution = nn.Conv2d(128, 64, 3, 1, 1)
        self.corr_reconstruction = CorrReconstruction(
            R=R, is_training=is_training, temperature=temperature,
        )

    def _extract_features(self, x):
        """Extract and post-process features."""
        _, _, _, _, feats = self.feature_extraction(x)
        return self.post_convolution(feats)

    def forward(self, source, target, return_corr=False):
        """
        Register source to target via correlation-based reconstruction.

        Args:
            source: (B, 1, H, W) source slice (raw intensity)
            target: (B, 1, H, W) target slice (raw intensity)
            return_corr: if True, the softmax correspondence weights are
                included in the output dict under 'corr_weights'. Used by
                training diagnostics to detect blur collapse / overconfident
                matches / noisy correspondences.

        Returns:
            dict:
                'warped_source': (B, 1, H, W) source reconstructed as target
                'deformation': None (Sli2Vol doesn't produce displacement fields)
                'corr_weights': (B, (2R+1)^2, H, W) softmax weights
                    [only when return_corr=True; omitted otherwise]
        """
        # Apply edge profile if enabled
        if self.use_edge_profile:
            source_input, target_input = edge_profile(
                [source, target], False, self.edge_radius, self.edge_interval
            )
        else:
            source_input = source
            target_input = target

        _, _, h, w = source_input.size()

        # Extract features
        feats_source = self._extract_features(source_input)
        feats_target = self._extract_features(target_input)

        # Reconstruct target from source
        if return_corr:
            warped, corr = self.corr_reconstruction(
                feats_source, feats_target, source, return_corr=True,
            )
        else:
            warped = self.corr_reconstruction(feats_source, feats_target, source)
            corr = None

        warped = F.interpolate(warped, (h, w), mode='bilinear', align_corners=False)

        out = {
            'warped_source': warped,
            'deformation': None,
        }
        if corr is not None:
            out['corr_weights'] = corr
        return out

    def warp(self, image, deformation_field):
        """
        Not applicable for Sli2Vol (no deformation field).
        Use propagate_mask() for mask warping at inference.
        """
        raise NotImplementedError(
            "Sli2Vol uses correlation-based reconstruction, not deformation fields. "
            "Use forward() to warp images, or propagate_mask() for inference."
        )

    def forward_with_mask(self, source, target, mask):
        """
        Warp a mask from source to target at inference time.

        Args:
            source: (B, 1, H, W) source slice (raw intensity)
            target: (B, 1, H, W) target slice (raw intensity)
            mask:   (B, 1, H, W) integer label mask on source

        Returns:
            warped_mask: (B, 1, H, W) label mask warped to target
            logits:      (B, C, H, W) soft one-hot predictions (before argmax)
        """
        if self.use_edge_profile:
            source_input, target_input = edge_profile(
                [source, target], False, self.edge_radius, self.edge_interval
            )
        else:
            source_input = source
            target_input = target

        _, _, h, w = source_input.size()

        feats_source = self._extract_features(source_input)
        feats_target = self._extract_features(target_input)

        # For inference, corr_reconstruction converts mask to one-hot internally
        # when is_training=False
        self.corr_reconstruction.is_training = False
        logits = self.corr_reconstruction(feats_source, feats_target, mask)
        self.corr_reconstruction.is_training = self.is_training

        logits = F.interpolate(logits, (h, w), mode='bilinear', align_corners=False)
        warped_mask = torch.argmax(logits, 1, keepdim=True).float()

        return warped_mask, logits
