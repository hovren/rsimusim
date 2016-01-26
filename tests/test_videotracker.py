from __future__ import print_function, division
import unittest

import numpy as np
import numpy.testing as nt

from video_matches import Track, FrameStore

class TrackTests(unittest.TestCase):
    def test_empty_track(self):
        t = Track()
        self.assertEqual(len(t), 0)

    def test_add_measurement(self):
        t = Track()
        t.add_measurement(1, np.array([2., 50.]))
        self.assertEqual(len(t), 1)

    def test_require_continuous(self):
        t = Track()
        x = np.array([1., 2.])
        first_frame = 1
        t.add_measurement(first_frame, x)
        t.add_measurement(first_frame + 1, x)
        t.add_measurement(first_frame + 2, x)
        with self.assertRaises(ValueError):
            t.add_measurement(first_frame + 10, x)

    def test_get_measurement(self):
        t = Track()
        data = [(12 + i, np.array([2.*i, 2.*i + 5])) for i in range(5)]
        for framenum, pos in data:
            t.add_measurement(framenum, pos)

        for framenum, expected_pos in data:
            pos = t[framenum]
            nt.assert_equal(pos, expected_pos)

    def test_last(self):
        t = Track()
        x = np.array([13., 13.])
        t.add_measurement(12, np.array([12., 12.]))
        t.add_measurement(13, x)
        self.assertEqual(t.last_framenum, 13)
        nt.assert_equal(t.last_position, x)

    def test_first(self):
        t = Track()
        x = np.array([13., 13.])
        t.add_measurement(12, np.array([12., 12.]))
        t.add_measurement(13, x)
        self.assertEqual(t.first_framenum, 12)

    def test_in(self):
        t = Track()
        x = np.array([13., 13.])
        t.add_measurement(12, np.array([12., 12.]))
        t.add_measurement(13, x)
        self.assertTrue(12 in t)
        self.assertTrue(13 in t)
        self.assertFalse(14 in t)

    def test_id(self):
        t1 = Track()
        t2 = Track()
        self.assertNotEqual(t1.id, t2.id)

class TestFrameStore(unittest.TestCase):
    def test_empty(self):
        fs = FrameStore()
        self.assertEqual(len(fs), 0)

    def test_continuous(self):
        fs = FrameStore()
        for i, framenum in enumerate(range(12, 16)):
            image = np.empty((480, 640))
            fs.add(framenum, image)
            self.assertEqual(len(fs), i + 1)
        with self.assertRaises(ValueError):
            fs.add(200, np.empty((480, 640)))

    def test_fetch(self):
        data = [
            (framenum, np.ones((32, 24)) * framenum)
            for framenum in range(11, 17)
        ]

        fs = FrameStore()
        for framenum, frame in data:
            fs.add(framenum, frame)

        for framenum, expected_frame in data:
            frame = fs[framenum]
            nt.assert_equal(frame, expected_frame)

    def test_max_size(self):
        max_size = 3
        fs = FrameStore(max_size)
        # Store frames 0, 1, 2, 3, 4, but only 4, 3, 2 should remain
        for i in range(5):
            fs.add(i, i*np.ones((32, 24)))

        self.assertEqual(len(fs), max_size)
        valid_framenums = [4, 3, 2]
        for framenum in valid_framenums:
            frame = fs[framenum]
            nt.assert_equal(frame, np.ones(frame.shape) * framenum)

        invalid_framenums = [0, 1]
        for framenum in invalid_framenums:
            with self.assertRaises(IndexError):
                frame = fs[framenum]

    def test_items(self):
        max_size = 3
        fs = FrameStore(max_size)
        data = [(i, i*np.ones((32, 24))) for i in range(max_size + 3)]
        for framenum, frame in data:
            fs.add(framenum, frame)

        expected_items = data[-max_size:]
        for i, (framenum, frame) in enumerate(fs.items()):
            expected_framenum, expected_frame = expected_items[i]
            self.assertEqual(framenum, expected_framenum)
            nt.assert_equal(frame, expected_frame)




