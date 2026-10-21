import unittest
from unittest.mock import Mock

from viewer import CloudPreview


class ViewerTests(unittest.TestCase):
    def preview(self):
        # Exercise the real event-loop logic without creating a GPU window.
        preview = CloudPreview.__new__(CloudPreview)
        preview.vis = Mock()
        preview.running = True
        return preview

    def test_quit_during_poll_is_not_overwritten(self):
        preview = self.preview()

        def dispatch_quit():
            preview.quit(preview.vis)
            return True  # Window still open when the callback finishes.

        preview.vis.poll_events.side_effect = dispatch_quit
        self.assertFalse(preview.poll())
        self.assertFalse(preview.running)
        preview.vis.update_renderer.assert_not_called()
        self.assertFalse(preview.poll())
        preview.vis.poll_events.assert_called_once()

    def test_window_close_stops_rendering(self):
        preview = self.preview()
        preview.vis.poll_events.return_value = False
        self.assertFalse(preview.poll())
        preview.vis.update_renderer.assert_not_called()

    def test_open_window_keeps_rendering(self):
        preview = self.preview()
        preview.vis.poll_events.return_value = True
        self.assertTrue(preview.poll())
        preview.vis.update_renderer.assert_called_once()
