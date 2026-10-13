"""Deterministic bounded DCT pHash perturbations; NumPy/OpenCV, no I/O.

This measures and changes one perceptual hash. It makes no claim about OCR,
speech recognition, learned embeddings, or semantic video understanding.
"""
import math
import cv2
import numpy as np

_N = 32
_D = np.cos(np.pi / _N * (np.arange(_N)[None, :] + .5)
            * np.arange(_N)[:, None]) * np.sqrt(2 / _N)
_D[0, :] /= np.sqrt(2)
_BASIS = np.stack([
    np.outer(_D[i // 8], _D[i % 8]).ravel() for i in range(64)
], axis=1)


def coeffs(frame):
    """Root hash convention: BGR uint8 → gray → INTER_AREA 32² → DCT-II 8²."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(float)
    return (_D @ small @ _D.T)[:8, :8].ravel()


def phash(frame):
    v = coeffs(frame)
    return v > np.median(v[1:])


def _check_reference(v, reference_bits):
    if reference_bits is None:
        return v > np.median(v[1:])
    reference = np.asarray(reference_bits, dtype=bool)
    if reference.shape != (64,):
        raise ValueError('reference_bits must contain exactly 64 values')
    return reference.copy()


def optimized_field(v, eps, target_bits, steps=180, reference_bits=None):
    """Return a bounded smooth 32² field and its optimized target margin.

    Pixel values and eps use the 0–255 convention. Selection includes equally
    many true/false reference bits. Already changed easy bits are retained when
    a separate reference hash is supplied. DC is never targeted.
    """
    v = np.asarray(v, dtype=float)
    reference = _check_reference(v, reference_bits)
    median = np.median(v[1:])
    # Negative costs correspond to bits already different from the reference.
    costs = np.where(reference, 1., -1.) * (v - median)
    upper = sorted([i for i in range(1, 64) if reference[i]], key=lambda i: (costs[i], i))
    lower = sorted([i for i in range(1, 64) if not reference[i]], key=lambda i: (costs[i], i))
    indices = upper[:target_bits // 2] + lower[:target_bits // 2]
    selected = _BASIS[:, indices]
    desired = np.where(reference[indices], -1., 1.)
    field = np.zeros(1024)
    best_margin = -np.inf
    best_field = field.copy()
    temperature = max(2., eps * .35)
    for step in range(steps):
        margins = desired * (v[indices] + field @ selected - median)
        margin = float(margins.min())
        if margin > best_margin:
            best_margin, best_field = margin, field.copy()
        weights = np.exp(-(margins - margins.min()) / temperature)
        weights /= weights.sum()
        gradient = selected @ (weights * desired)
        rate = eps * (.055 if step < steps * 2 / 3 else .015)
        field = np.clip(field + rate * gradient / max(abs(gradient).max(), 1e-10), -eps, eps)
    return best_field.reshape(32, 32), best_margin


def _apply(frame, field, amplitude, eps):
    h, w = frame.shape[:2]
    full = cv2.resize(field * amplitude, (w, h), interpolation=cv2.INTER_LINEAR)
    delta = np.clip(np.rint(full), -math.floor(eps), math.floor(eps)).astype(np.int16)
    return np.clip(frame.astype(np.int16) + delta[:, :, None], 0, 255).astype(np.uint8)


def _measure(frame, altered, reference, target_bits):
    actual = coeffs(altered)
    median = np.median(actual[1:])
    changes = reference != (actual > median)
    distance = int(changes.sum())
    margins = abs(actual[changes] - median)
    mse = cv2.norm(frame, altered, cv2.NORM_L2SQR) / frame.size
    difference = altered.astype(np.int16) - frame.astype(np.int16)
    return dict(phash_hamming_64=distance, target_achieved=distance >= target_bits,
                mean_flipped_margin=float(margins.mean()) if len(margins) else 0.,
                psnr_rgb_db=10 * math.log10(255 ** 2 / max(mse, 1e-12)),
                rms_rgb_delta=float(math.sqrt(mse)),
                max_rgb_delta=int(np.max(abs(difference))))


def change_frame(frame, eps, target_bits, steps=180, reference_bits=None):
    """Return (altered BGR uint8 frame, metrics), without changing the input.

    eps bounds actual pre-encode per-channel RGB/BGR differences. target_bits is
    a desired distance, not a promise: inspect returned target_achieved. This
    function does not alter audio, frame order, timestamps, or dimensions.
    """
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('frame must be an H×W×3 uint8 BGR array')
    if not np.isfinite(eps) or not 0 <= eps <= 255:
        raise ValueError('eps must be within [0,255]')
    if not isinstance(target_bits, (int, np.integer)) or not 2 <= target_bits <= 62 or target_bits % 2:
        raise ValueError('target_bits must be an even integer between 2 and 62')
    if steps < 1:
        raise ValueError('steps must be positive')
    v = coeffs(frame)
    reference = _check_reference(v, reference_bits)
    if eps < 1 or np.count_nonzero(reference != phash(frame)) >= target_bits:
        altered = frame.copy()
        info = _measure(frame, altered, reference, target_bits)
        info.update(eps=float(eps), requested_target_bits=int(target_bits), optimized_target_bits=0,
                    amplitude=0., steps=0, optimized_margin=None)
        return altered, info
    best = None
    # A small deterministic search compensates for median changes and clipping.
    for count in sorted(set([target_bits, min(62, target_bits + 4)])):
        field, optimized_margin = optimized_field(v, eps, count, steps, reference)
        predicted = field.ravel() @ _BASIS
        chosen = 1.
        for amplitude in [.3, .4, .5, .6, .7, .8, .9, 1.]:
            trial = v + predicted * amplitude
            median = np.median(trial[1:])
            changed = reference != (trial > median)
            margins = abs(trial[changed] - median)
            if changed.sum() >= target_bits and margins.mean() >= 3.:
                chosen = amplitude
                break
        altered = _apply(frame, field, chosen, eps)
        info = _measure(frame, altered, reference, target_bits)
        # Native clipping/rounding can defeat the low-resolution estimate.
        if not info['target_achieved'] and chosen < 1.:
            for amplitude in np.arange(round(chosen + .1, 1), 1.001, .1):
                chosen = min(float(amplitude), 1.)
                altered = _apply(frame, field, chosen, eps)
                info = _measure(frame, altered, reference, target_bits)
                if info['target_achieved']:
                    break
        # Among solutions meeting the desired distance, choose minimum RMS.
        score = (min(info['phash_hamming_64'], target_bits),
                 -info['rms_rgb_delta'] if info['target_achieved'] else info['mean_flipped_margin'])
        info.update(eps=float(eps), requested_target_bits=int(target_bits),
                    optimized_target_bits=count, amplitude=float(chosen), steps=int(steps),
                    optimized_margin=float(optimized_margin))
        if best is None or score > best[0]:
            best = score, altered, info
    assert best[2]['max_rgb_delta'] <= eps
    assert best[1].shape == frame.shape
    return best[1], best[2]
