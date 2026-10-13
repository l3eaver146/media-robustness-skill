"""Resolve a local TrueType font without bundling or downloading fonts.

Set MEDIA_ROBUSTNESS_FONT before starting the process to freeze font choice.
The historical Arial default is preferred when installed. Different font files
produce different glyph masks; exact pixel parity requires the same font file.
"""
import os
from pathlib import Path

from PIL import ImageFont


FONT_CANDIDATES = (
    Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
    Path('/Library/Fonts/Arial.ttf'),
    Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    Path('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'),
    Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'),
    Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'),
    Path('/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf'),
    Path('C:/Windows/Fonts/arial.ttf'),
)


def font_path():
    """Return an installed font path, honoring a strict absolute override."""
    override = os.environ.get('MEDIA_ROBUSTNESS_FONT')
    if override is not None:
        path = Path(override)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('MEDIA_ROBUSTNESS_FONT must be the absolute path to an existing font file.')
        return path
    for path in FONT_CANDIDATES:
        if path.is_file():
            return path
    raise RuntimeError('No supported font found. Install Arial, DejaVu Sans or Liberation Sans, '
                       'or set MEDIA_ROBUSTNESS_FONT to an absolute local font-file path.')


def load_font(size):
    """Load the selected font and explain an invalid-font override clearly."""
    path = font_path()
    try:
        return ImageFont.truetype(str(path), size)
    except OSError as exc:
        raise ValueError('Cannot load the selected font. Set MEDIA_ROBUSTNESS_FONT '
                         'to an absolute path to a usable TrueType/OpenType font file.') from exc
