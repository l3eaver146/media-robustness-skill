"""Frozen frame transforms for local media robustness comparisons."""
from __future__ import annotations

from functools import lru_cache
import math

import cv2
import numpy as np

import refined
from faint_labels import baseline, bottom_mask
from media_utils import geometry
from process_grid import grid_field
from core import phash

PROFILES = {
    'control': 'Exact prior adaptive_smooth pixels; comparison reference.',
    'encode_only': 'No pixel transform; identical codec settings as the other profiles.',
    'stable_text': 'Only text polarity selection changes: temporal smoothing and hysteresis.',
    'glyph_blur': 'Only blur support changes: restricted to the vicinity of letters.',
    'protected': 'Only spatial protection changes: attenuate effects at edges/manual regions.',
    'multiscale': 'Only the grid changes: pitches 7/13/23 at the prior maximum amplitude.',
    'multiscale_matched': 'V05: multiscale grid with rounded RMS and mean matched approximately to production, peak <=24.',
}


def protection_from_spec(spec: dict | None, height: int, width: int):
    """Normalized rectangles in the OUTPUT frame. Feathering extends outwards."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - {'rectangles', 'feather_px_1080'}:
        raise ValueError('Protection JSON accepts rectangles and feather_px_1080 only.')
    rects = spec.get('rectangles', [])
    if not isinstance(rects, list):
        raise ValueError('rectangles must be a list of [left, top, right, bottom].')
    mask = np.zeros((height, width), np.float32)
    for rect in rects:
        if not isinstance(rect, list) or len(rect) != 4:
            raise ValueError('Each rectangle needs four normalized coordinates.')
        try:
            x0, y0, x1, y1 = map(float, rect)
        except (ValueError, TypeError) as exc:
            raise ValueError('Rectangle coordinates must be numbers.') from exc
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('Rectangle coordinates must satisfy 0 <= start < end <= 1.')
        mask[math.floor(y0 * height):math.ceil(y1 * height),
             math.floor(x0 * width):math.ceil(x1 * width)] = 1
    feather = float(spec.get('feather_px_1080', 12))
    if not math.isfinite(feather) or not 0 <= feather <= 200:
        raise ValueError('feather_px_1080 must be within 0..200.')
    if feather:
        halo = cv2.GaussianBlur(mask, (0, 0), max(.1, feather * width / 1080))
        mask = np.maximum(mask, halo)
    return mask


@lru_cache(maxsize=4)
def _glyph_support(h, w):
    radius = max(1, round(8 * w / 1080))
    text = bottom_mask(h, w)
    near = cv2.dilate(text, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    support = np.maximum(text, cv2.GaussianBlur(near, (0, 0), max(.5, 4 * w / 1080)))
    support.setflags(write=False)
    return support


@lru_cache(maxsize=4)
def _mixed_grid(h, w):
    field = sum(weight * grid_field(h, w, pitch, 1, 24).astype(np.float32)
                for pitch, weight in ((7, .25), (13, .45), (23, .30)))
    field = np.rint(np.clip(field, -24, 24)).astype(np.int16)
    field.setflags(write=False)
    return field


@lru_cache(maxsize=4)
def _matched_grid(h, w):
    """Match rounded field statistics, not detector performance or visibility.

    Work on a 49-bin histogram rather than repeatedly scanning the whole frame.
    Joint gain/offset search accounts for asymmetric saturation and rounding.
    Every result retains the original absolute grid bound of 24.
    """
    mixed = _mixed_grid(h, w)
    reference = grid_field(h, w, 10, 1, 24).astype(np.float64)
    target_mean = float(reference.mean())
    target_rms = float(np.sqrt(np.mean(reference**2)))
    values = np.arange(-24, 25, dtype=np.float64)
    weights = np.bincount((mixed.ravel() + 24), minlength=49).astype(np.float64)
    weights /= weights.sum()
    best = None
    # Deterministic search of quantized fields; no random dither or temporal noise.
    for gain in np.linspace(1., 3., 601):
        lo, hi = -72., 72.
        for _ in range(20):
            offset = (lo + hi) / 2
            lut = np.clip(np.rint(values * gain - offset), -24, 24)
            if float(lut @ weights) > target_mean:
                lo = offset
            else:
                hi = offset
        for offset in (lo, hi):
            lut = np.clip(np.rint(values * gain - offset), -24, 24)
            mean = float(lut @ weights)
            rms = float(np.sqrt((lut**2) @ weights))
            # Bound DC drift first; a weighted penalty can trade away this limit.
            if abs(mean - target_mean) > .20:
                continue
            score = abs(rms - target_rms)
            if best is None or score < best[0]:
                best = (score, lut.astype(np.int16), float(gain), float(offset), mean, rms)
    if best is None:
        raise ValueError('No grid field satisfies the mean constraint for this frame size.')
    _, lut, gain, offset, mean, rms = best
    field = lut[mixed + 24]
    field.setflags(write=False)
    info = dict(reference_rms=target_rms, actual_rms=rms,
                reference_mean=target_mean, actual_mean=mean,
                max_abs_delta=int(np.abs(field).max()), gain=gain, offset=offset,
                mean_absolute_error=abs(mean - target_mean), mean_error_limit=.20,
                rms_relative_error=abs(rms - target_rms) / max(target_rms, 1e-9),
                scope='Bare grid before image clipping, blur, text or video encoding')
    return field, info


def _automatic_protection(clean):
    """Contrast-edge proxy, NOT a face/text/button detector."""
    h, w = clean.shape[:2]
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (min(w, 270), max(1, round(h * min(w, 270) / w))))
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3) / 4
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3) / 4
    edges = np.clip((cv2.magnitude(gx, gy) - 12) / 32, 0, 1)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    edges = cv2.GaussianBlur(edges, (0, 0), 1.2)
    return .85 * cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)


class VideoTransform:
    """One instance per sequential video; no state is shared between clips."""

    def __init__(self, profile: str, fps: float, protection_mask=None):
        if profile not in PROFILES:
            raise ValueError(f'Unknown profile: {profile}')
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('FPS must be positive and finite.')
        if protection_mask is not None:
            if profile != 'protected':
                raise ValueError('A protection mask is supported only by protected.')
            protection_mask = np.array(protection_mask, dtype=np.float32, copy=True)
            if (protection_mask.ndim != 2 or not np.isfinite(protection_mask).all()
                    or protection_mask.min() < 0 or protection_mask.max() > 1):
                raise ValueError('Protection mask must be a finite HxW array in [0,1].')
        self.profile, self.fps = profile, fps
        self.manual_mask = protection_mask
        self.shape = None
        self.frames = self.polarity_switches = self.scene_resets = 0
        self.within_scene_switches = 0
        self._black = self._ema = self._target = self._previous_thumb = None
        self._protection_sum = 0.0

    def _target_color(self, hard):
        h, w = hard.shape[:2]
        (x0, x1, y0, y1), _ = refined.region(h, w)
        gray = cv2.cvtColor(hard[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        median = float(np.median(gray))
        cut = False
        if self.profile == 'stable_text':
            thumb = cv2.resize(cv2.cvtColor(hard, cv2.COLOR_BGR2GRAY), (32, 56))
            if self._previous_thumb is not None:
                difference = float(cv2.absdiff(thumb, self._previous_thumb).mean())
                first = np.bincount((thumb // 16).ravel(), minlength=16) / thumb.size
                second = np.bincount((self._previous_thumb // 16).ravel(), minlength=16) / thumb.size
                cut = difference > 40 and float(np.abs(first - second).sum()) > .8
            self._previous_thumb = thumb
            if self._ema is None or cut:
                self._ema = median
                black = median >= 128
            else:
                self._ema += (median - self._ema) * (1 - math.exp(-1 / (self.fps * .25)))
                black = self._black
                if black and self._ema < 116:
                    black = False
                elif not black and self._ema > 140:
                    black = True
            if cut:
                self.scene_resets += 1
        else:
            black = median >= 128
        if self._black is not None and black != self._black:
            self.polarity_switches += 1
            if not cut:
                self.within_scene_switches += 1
        self._black = black
        desired = 0.0 if black else 255.0
        if self.profile != 'stable_text' or self._target is None or cut:
            self._target = desired
        else:
            self._target += (desired - self._target) * (1 - math.exp(-1 / (self.fps * .18)))
        return self._target

    def __call__(self, frame, config=None):
        if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3
                or min(frame.shape[:2]) < 96):
            raise ValueError('Expected uint8 BGR frames, with each dimension >=96.')
        if self.shape is not None and self.shape != frame.shape:
            raise ValueError('Dimensions changed within a video.')
        self.shape = frame.shape
        h, w = frame.shape[:2]
        if self.manual_mask is not None and self.manual_mask.shape != (h, w):
            raise ValueError('Protection mask dimensions differ from the video.')
        if self.profile == 'encode_only':
            out = frame.copy()
        else:
            if self.profile in {'multiscale', 'multiscale_matched'}:
                clean = geometry(frame, 1.02)
                field = _matched_grid(h, w)[0] if self.profile == 'multiscale_matched' else _mixed_grid(h, w)
                hard = np.clip(clean.astype(np.int16) + field[:, :, None], 0, 255).astype(np.uint8)
            else:
                hard = baseline(frame)
            target = self._target_color(hard)
            (x0, x1, y0, y1), weight = refined.region(h, w)
            patch = hard[y0:y1, x0:x1].astype(np.float32)
            smooth = cv2.GaussianBlur(patch, (0, 0), 6 * w / 1080)
            if self.profile == 'glyph_blur':
                weight = weight * _glyph_support(h, w)[y0:y1, x0:x1, None]
            patch += (smooth - patch) * weight * .9
            text = bottom_mask(h, w)[y0:y1, x0:x1, None]
            out = hard.copy()
            out[y0:y1, x0:x1] = np.clip(np.rint(patch + .20 * text * (target - patch)), 0, 255).astype(np.uint8)
            if self.profile == 'protected':
                clean = geometry(frame, 1.02)
                protect = _automatic_protection(clean)
                if self.manual_mask is not None:
                    protect = np.maximum(protect, self.manual_mask)
                self._protection_sum += float(protect.mean())
                mixed = out.astype(np.float32) + protect[:, :, None] * (clean.astype(np.float32) - out)
                out = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
                out[protect >= 1] = clean[protect >= 1]
        self.frames += 1
        mse = cv2.norm(frame, out, cv2.NORM_L2SQR) / frame.size
        info = dict(phash_hamming_64=int(np.count_nonzero(phash(frame) != phash(out))),
                    psnr_rgb_db=10 * math.log10(255**2 / max(mse, 1e-12)),
                    max_rgb_delta=int(cv2.absdiff(frame, out).max()))
        return out, info

    def summary(self):
        result = dict(profile=self.profile, frames=self.frames,
                    polarity_switches=self.polarity_switches,
                    within_scene_polarity_switches=self.within_scene_switches,
                    scene_resets=self.scene_resets,
                    mean_protection_weight=self._protection_sum / max(1, self.frames),
                    manual_protection=self.manual_mask is not None,
                    text_opacity=0 if self.profile == 'encode_only' else .20,
                    preencode_metrics_reference='original decoded frame, including all transformations')
        if self.profile == 'multiscale_matched' and self.shape is not None:
            result['grid_statistics'] = dict(_matched_grid(*self.shape[:2])[1])
        return result
