"""Synthetic math, temporal, immutability and portable-font regression checks."""
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ENGINE = Path(__file__).resolve().parents[1] / 'skills/media-robustness/scripts/_engine'
sys.path.insert(0, str(ENGINE))
import core
import font_utils
from media_utils import geometry
from process_grid import grid_field
import static_transform
import tiger_transform as tiger
import variants


def synthetic_frame(index=0, height=320, width=180):
    y, x = np.mgrid[:height, :width]
    return np.stack(((3*x+y+7*index) % 206+25,
                     (x+2*y+11*index) % 206+25,
                     (2*x+3*y+13*index) % 206+25), axis=2).astype(np.uint8)


class GridMathTests(unittest.TestCase):
    def test_frozen_multiscale_field_digest(self):
        # Original quantized V04 fields, encoded as little-endian int16.
        expected = {
            (320, 180): 'd9c7cd73c2a77b8374b9bb8a154b4d3ea0d5b82387e9dbffab796a7c0582cfbc',
            (1920, 1080): '5a99142bd72004c627508ba43d3f007c639dfe1ee40f1de5fe1c8c201ecd5ed0',
        }
        for shape, digest in expected.items():
            actual = variants._mixed_grid(*shape)
            self.assertEqual(hashlib.sha256(actual.astype('<i2', copy=False).tobytes()).hexdigest(), digest)

    def test_matched_grid_bounds_energy_mean_and_cache(self):
        for shape in ((320, 180), (1920, 1080)):
            field, info = variants._matched_grid(*shape)
            reference = grid_field(*shape, 10, 1, 24).astype(np.float64)
            actual = field.astype(np.float64)
            rms = float(np.sqrt(np.mean(actual**2)))
            self.assertLessEqual(int(np.abs(field).max()), 24)
            self.assertLess(abs(rms / np.sqrt(np.mean(reference**2)) - 1), .02)
            self.assertLessEqual(abs(float(actual.mean() - reference.mean())), .20)
            self.assertAlmostEqual(info['actual_rms'], rms, places=11)
            self.assertFalse(field.flags.writeable)
            self.assertIs(field, variants._matched_grid(*shape)[0])
            with self.assertRaises(ValueError):
                field[0, 0] = 0

    def test_dct_bound_is_actual_pixel_bound_and_does_not_mutate(self):
        frame = synthetic_frame()
        original = frame.copy()
        out, metrics = core.change_frame(frame, eps=12, target_bits=24, steps=40)
        np.testing.assert_array_equal(frame, original)
        self.assertLessEqual(int(cv2.absdiff(frame, out).max()), 12)
        self.assertEqual(metrics['phash_hamming_64'], int(np.count_nonzero(core.phash(frame) != core.phash(out))))


class FrameContractTests(unittest.TestCase):
    def test_every_variant_preserves_input_shape_and_dtype(self):
        frame = synthetic_frame()
        original = frame.copy()
        frame.setflags(write=False)
        for profile in variants.PROFILES:
            out, metrics = variants.VideoTransform(profile, 30)(frame)
            np.testing.assert_array_equal(frame, original)
            self.assertEqual(out.shape, frame.shape)
            self.assertEqual(out.dtype, np.uint8)
            self.assertFalse(np.shares_memory(out, frame))
            self.assertEqual(metrics['max_rgb_delta'], int(cv2.absdiff(frame, out).max()))
            if profile == 'encode_only':
                np.testing.assert_array_equal(out, frame)

    def test_protection_masks_are_copied_and_use_output_coordinates(self):
        frame = synthetic_frame()
        mask = variants.protection_from_spec({'rectangles': [[.2, .1, .7, .8]], 'feather_px_1080': 0}, 320, 180)
        transform = variants.VideoTransform('protected', 30, mask)
        mask[:] = 0
        self.assertGreater(transform.manual_mask.sum(), 0)
        out, _ = transform(frame)
        clean = geometry(frame, 1.02)
        selected = transform.manual_mask == 1
        np.testing.assert_array_equal(out[selected], clean[selected])

    def test_bad_profiles_frames_masks_and_rates_are_rejected(self):
        for rate in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                variants.VideoTransform('control', rate)
        with self.assertRaises(ValueError):
            variants.VideoTransform('missing', 30)
        with self.assertRaises(ValueError):
            variants.VideoTransform('protected', 30, np.full((320, 180), np.nan))
        transform = variants.VideoTransform('control', 30)
        for frame in (np.zeros((320, 180, 3), np.float32), np.zeros((95, 180, 3), np.uint8)):
            with self.assertRaises(ValueError):
                transform(frame)
        self.assertEqual(transform.frames, 0)

    def test_hard_image_is_bounded_against_clean_zoom(self):
        frame = synthetic_frame(height=1920, width=1080)
        original = frame.copy()
        out, info = static_transform.transform(frame, static_transform.EXTREME_SETTINGS)
        np.testing.assert_array_equal(frame, original)
        self.assertFalse(np.shares_memory(frame, out))
        self.assertEqual(out.shape, frame.shape)
        measured = int(cv2.absdiff(geometry(frame, 1.02), out).max())
        self.assertEqual(info['max_rgb_delta_after_zoom_before_encoding'], measured)
        self.assertLessEqual(measured, 96)
        with self.assertRaisesRegex(ValueError, '1080x1920'):
            static_transform.transform(synthetic_frame(), static_transform.EXTREME_SETTINGS)


class TemporalTests(unittest.TestCase):
    def test_stable_footer_hysteresis_and_independent_state(self):
        frames = [np.full((320, 180, 3), level, np.uint8) for level in [126, 130]*18]
        control = variants.VideoTransform('control', 30)
        first = variants.VideoTransform('stable_text', 30)
        second = variants.VideoTransform('stable_text', 30)
        unrelated = variants.VideoTransform('stable_text', 30)
        for i, frame in enumerate(frames):
            control(frame)
            unrelated(np.full_like(frame, 20 if i % 2 else 235))
            np.testing.assert_array_equal(first(frame)[0], second(frame)[0])
        self.assertGreater(control.polarity_switches, 18)
        self.assertLessEqual(first.polarity_switches, 1)
        self.assertEqual(first.summary(), second.summary())

    def test_exact_replacement_schedule_with_fractional_fps(self):
        self.assertEqual(tiger.replacement_indices(122, 30), (0, 60, 120, 121))
        expected = (0, 60, 120, 180, 240, 300, 360, 420, 480, 539, 599, 659, 719, 779, 839, 899, 959, 999)
        self.assertEqual(tiger.replacement_indices(1000, Fraction(30000, 1001)), expected)
        for total, rate in ((1, 30), (10, 0), (10, -1)):
            with self.assertRaises(ValueError):
                tiger.replacement_indices(total, rate)

    def test_title_cards_replace_one_frame_without_temporal_state_leakage(self):
        enabled = tiger.TigerTransform(5, 12, 'deadbeef01234567')
        spatial = tiger.TigerTransform(5, 12, 'deadbeef01234567', context_enabled=False)
        self.assertEqual(enabled.schedule, (0, 10, 11))
        for index in range(12):
            frame = synthetic_frame(index)
            original = frame.copy()
            base, _ = spatial(frame)
            out, _ = enabled(frame)
            expected = tiger.context_frame(base) if index in enabled.schedule else base
            np.testing.assert_array_equal(out, expected)
            np.testing.assert_array_equal(frame, original)
        summary = enabled.summary()
        summary['applied_replacements'].append(-1)
        self.assertEqual(enabled.summary()['applied_replacements'], [0, 10, 11])
        with self.assertRaisesRegex(ValueError, 'verified source count'):
            enabled(synthetic_frame())

    def test_warp_is_static_finite_bounded_and_read_only(self):
        h, w = 320, 180
        phase = tiger.TigerTransform(30, 3, 'deadbeef').phase
        mx, my = tiger.warp_maps(h, w, phase)
        y, x = np.mgrid[:h, :w].astype(np.float64)
        for actual, base, bound in ((mx, (w-1)/2+(x-(w-1)/2)/1.015, 11*w/1080),
                                    (my, (h-1)/2+(y-(h-1)/2)/1.015, 7*h/1920)):
            self.assertTrue(np.isfinite(actual).all())
            self.assertFalse(actual.flags.writeable)
            self.assertLessEqual(float(np.abs(actual-base).max()), bound+.0003)
            np.testing.assert_allclose((actual-base)[[0, -1], :], 0, atol=.0003)
        self.assertIs(mx, tiger.warp_maps(h, w, phase)[0])
        dx_dy, dx_dx = np.gradient(mx)
        dy_dy, dy_dx = np.gradient(my)
        self.assertGreater(float((dx_dx*dy_dy-dx_dy*dy_dx).min()), 0)


class FontPortabilityTests(unittest.TestCase):
    def test_explicit_font_takes_precedence(self):
        real_font = font_utils.font_path()
        with patch.dict(os.environ, {'MEDIA_ROBUSTNESS_FONT': str(real_font)}), patch.object(font_utils, 'FONT_CANDIDATES', ()):
            self.assertEqual(font_utils.font_path(), real_font)
            self.assertIsNotNone(font_utils.load_font(12))

    def test_absolute_override_and_actionable_missing_font_errors(self):
        for value in ('relative.ttf', '/nonexistent/media-robustness-font.ttf', ''):
            with patch.dict(os.environ, {'MEDIA_ROBUSTNESS_FONT': value}):
                with self.assertRaisesRegex(ValueError, 'absolute path'):
                    font_utils.font_path()
        with patch.dict(os.environ, {}, clear=True), patch.object(font_utils, 'FONT_CANDIDATES', ()):
            with self.assertRaisesRegex(RuntimeError, 'MEDIA_ROBUSTNESS_FONT'):
                font_utils.font_path()

    def test_fallback_and_invalid_font_content(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate = Path(folder) / 'font.ttf'
            candidate.write_text('not a font', encoding='utf-8')
            with patch.dict(os.environ, {}, clear=True), patch.object(font_utils, 'FONT_CANDIDATES', (candidate,)):
                self.assertEqual(font_utils.font_path(), candidate)
                with self.assertRaisesRegex(ValueError, 'TrueType/OpenType'):
                    font_utils.load_font(12)


if __name__ == '__main__':
    unittest.main()
