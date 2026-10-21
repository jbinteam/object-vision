"""Camera selection tests do not enumerate or open physical devices."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import camera


class BackendSelectionTests(unittest.TestCase):
    def orbbec_module(self, found=()):
        module = types.ModuleType("orbbec_camera")
        module.sdk_available = Mock(return_value=True)
        module.devices = Mock(return_value=list(found))
        module.Gemini305 = Mock()
        return module

    def test_explicit_orbbec_does_not_require_realsense(self):
        module = self.orbbec_module([dict(name="Gemini305", serial="g", backend="orbbec")])
        with patch.dict(sys.modules, orbbec_camera=module), patch("camera.rs", None):
            result = camera.open_camera("orbbec")
        self.assertIs(result, module.Gemini305.return_value)
        module.Gemini305.assert_called_once_with("g")

    def test_missing_realsense_sdk_is_actionable(self):
        with patch("camera.rs", None):
            with self.assertRaisesRegex(RuntimeError, "requires pyrealsense2"):
                camera.devices("realsense")

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown camera backend"):
            camera.open_camera("invalid")

    def test_auto_requires_selection_across_supported_cameras(self):
        found = [dict(name="Gemini305", serial="g", backend="orbbec"),
                 dict(name="Intel RealSense L515", serial="l", backend="realsense")]
        with patch("camera.devices", return_value=found):
            with self.assertRaisesRegex(RuntimeError, "Multiple supported"):
                camera.open_camera()

    def test_serial_selects_camera_without_fallback(self):
        module = self.orbbec_module()
        found = [dict(name="Gemini305", serial="g", backend="orbbec"),
                 dict(name="Intel RealSense L515", serial="l", backend="realsense")]
        module.Gemini305.side_effect = RuntimeError("alignment unsupported")
        with patch("camera.devices", return_value=found), patch.dict(sys.modules, orbbec_camera=module), \
                patch("camera.L515") as legacy:
            with self.assertRaisesRegex(RuntimeError, "alignment unsupported"):
                camera.open_camera(serial="g")
            legacy.assert_not_called()

    def test_unsupported_realsense_is_not_opened(self):
        found = [dict(name="Intel RealSense D435", serial="d", backend="realsense")]
        with patch("camera.devices", return_value=found):
            with self.assertRaisesRegex(RuntimeError, "No matching supported"):
                camera.open_camera()

    def test_auto_ignores_absent_optional_sdks(self):
        module = self.orbbec_module()
        module.sdk_available.return_value = False
        with patch.dict(sys.modules, orbbec_camera=module), patch("camera.rs", None):
            self.assertEqual(camera.devices(), [])
            module.devices.assert_not_called()


class DepthMillimetreTests(unittest.TestCase):
    def test_integer_millimetres_keep_zero_and_uint16_maximum(self):
        raw = np.array([[0, 1, 2, 3, 65535]], dtype=np.uint16)
        np.testing.assert_array_equal(camera.prepare_depth(raw, .001, .001, 65.535, 1.), raw)

    def test_fractional_sensor_scale_rounds_and_saturates_without_wrapping(self):
        raw = np.array([[0, 1, 2, 3, 65535]], dtype=np.uint16)
        np.testing.assert_array_equal(camera.prepare_depth(raw, .002, .001, 132., 1.),
                                      [[0, 2, 4, 6, 65535]])
        np.testing.assert_array_equal(camera.prepare_depth(np.array([[4000]], np.uint16),
                                                         .0001, .1, 2., 1.), [[400]])


if __name__ == "__main__":
    unittest.main()
