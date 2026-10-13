"""Frozen strong still-image transforms; file encoding belongs to the caller."""
import cv2
import numpy as np
from core import change_frame, phash
from faint_labels import bottom_mask
from media_utils import geometry
from process_grid import grid_field
from refined import region

PROFILES = {
    'strong': dict(grid_eps=24, dct_eps=24, target_bits=36, total_bound=64),
    'stronger': dict(grid_eps=32, dct_eps=32, target_bits=42, total_bound=80),
    'extreme': dict(grid_eps=40, dct_eps=40, target_bits=48, total_bound=96),
}
EXTREME_SETTINGS = PROFILES['extreme']


def transform(original, settings):
    """Transform one 1080x1920 uint8 BGR image without changing the input."""
    if (not isinstance(original, np.ndarray) or original.dtype != np.uint8
            or original.shape != (1920, 1080, 3)):
        raise ValueError('The frozen hard-image profile requires a 1080x1920 uint8 BGR image.')
    clean = geometry(original, 1.02)
    h, w = clean.shape[:2]
    # Preserve high-contrast edges by attenuating only the fine grid there.
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 4
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 4
    edge = np.clip((cv2.magnitude(gx, gy) - 12) / 40, 0, 1)
    edge = cv2.GaussianBlur(cv2.dilate(edge, np.ones((5, 5), np.uint8)), (0, 0), 2)
    grid = sum(weight * grid_field(h, w, pitch, 1, settings['grid_eps']).astype(np.float32)
               for pitch, weight in ((7, .25), (10, .25), (13, .3), (23, .2)))
    grid *= 1 - .65 * edge
    altered = np.clip(np.rint(clean.astype(np.float32) + grid[:, :, None]), 0, 255).astype(np.uint8)
    # Existing labels at the bottom, with smoothing confined to letter vicinity.
    (x0, x1, y0, y1), feather = region(h, w)
    mask = bottom_mask(h, w)[y0:y1, x0:x1]
    near = cv2.dilate(mask, np.ones((17, 17), np.uint8))
    near = cv2.GaussianBlur(near, (0, 0), 4)[:, :, None] * feather
    patch = altered[y0:y1, x0:x1].astype(np.float32)
    smooth = cv2.GaussianBlur(patch, (0, 0), 6)
    patch += (smooth - patch) * near * .9
    target = 0 if float(np.median(gray[y0:y1, x0:x1])) >= 128 else 255
    altered[y0:y1, x0:x1] = np.clip(np.rint(patch + .20 * mask[:, :, None] * (target - patch)), 0, 255).astype(np.uint8)
    # This targets ONE explicit pHash convention, not a semantic vision model.
    altered, dct = change_frame(altered, settings['dct_eps'], settings['target_bits'],
                                steps=220, reference_bits=phash(original))
    bound = settings['total_bound']
    delta = np.clip(altered.astype(np.int16) - clean.astype(np.int16), -bound, bound)
    altered = np.clip(clean.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return altered, dict(dct_before_total_cap=dct, max_rgb_delta_after_zoom_before_encoding=int(cv2.absdiff(clean, altered).max()),
                         measured_phash_after_total_cap=int(np.count_nonzero(phash(original) != phash(altered))))

