import numpy as np
from contours.pseudo_objects import fix_hu_offset, HU_MIN, HU_MAX


def _u8(hu):
    return np.clip((hu - HU_MIN) / (HU_MAX - HU_MIN), 0, 1) * 255


def test_fix_hu_offset_shifts_unsigned_chaos_style_volumes():
    hu = np.array([-1000.0, -100.0, 60.0, 300.0, 1200.0], np.float32)       # air, fat, liver, kidney+contrast, bone
    chaos = hu + 1024.0                                                       # stored uint16 style, min 0
    out = fix_hu_offset(chaos.copy())
    assert np.allclose(out, hu)
    assert np.allclose(_u8(out), _u8(hu))
    # without the fix liver/kidney would saturate at 255
    assert _u8(chaos)[2] == 255 and _u8(hu)[2] < 255


def test_fix_hu_offset_leaves_hu_volumes_alone_and_maps_padding_to_air():
    hu = np.array([-2048.0, -1024.0, 40.0, 3000.0], np.float32)
    assert np.allclose(fix_hu_offset(hu.copy()), hu)
    chaos = np.array([0.0, 1084.0, 50944.0], np.float32)
    out = fix_hu_offset(chaos.copy())
    assert np.allclose(out, [-1024.0, 60.0, -1024.0])


def test_slicewise_upsampling_matches_full_volume_path():
    from contours import anchors as AN
    from contours.pseudo_objects import upsample_mask, upsample_slice, object_slice_areas
    rng = np.random.RandomState(0)
    Z, Hs, Ws, f = 7, 12, 10, 2
    obj_s = rng.rand(Z, Hs, Ws) > 0.6
    shape_hw = (Hs * f, Ws * f)
    full = upsample_mask(obj_s, f, shape_hw)
    for axis in range(3):
        ref = AN.permute_to_axis(full, axis)
        areas = object_slice_areas(obj_s, f, axis, shape_hw)
        assert areas.shape[0] == ref.shape[0]
        assert np.array_equal(areas, ref.reshape(ref.shape[0], -1).sum(1))
        for k in range(ref.shape[0]):
            assert np.array_equal(upsample_slice(obj_s, f, axis, k, shape_hw), ref[k]), (axis, k)
    # cropped / padded shapes still agree with the reference path
    for shape_hw in [(Hs * f - 3, Ws * f + 5), (Hs * f + 4, Ws * f - 2)]:
        full = upsample_mask(obj_s, f, shape_hw)
        for axis in range(3):
            ref = AN.permute_to_axis(full, axis)
            for k in range(0, ref.shape[0], 3):
                assert np.array_equal(upsample_slice(obj_s, f, axis, k, shape_hw), ref[k]), (shape_hw, axis, k)
