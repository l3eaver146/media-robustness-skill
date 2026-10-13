"""Frozen sample footer strings and baseline grid, without experiment I/O."""
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw
from font_utils import load_font
from media_utils import geometry
from process_grid import grid_field

# Historical sample context labels are retained for snapshot reproducibility.
CORE_LABELS = ['ZEUS ADVENTURES', 'MOBILE FANTASY GAME']
BOTTOM_ROWS = [(CORE_LABELS[0],1800,54),(CORE_LABELS[1],1852,34)]


@lru_cache(maxsize=4)
def bottom_mask(h,w):
    image=Image.new('L',(w,h));draw=ImageDraw.Draw(image)
    for label,y,size in BOTTOM_ROWS:
        font=load_font(max(10,round(size*w/1080)))
        draw.text((w//2,round(y*h/1920)),label,anchor='mt',font=font,fill=255,stroke_width=1)
    mask=np.asarray(image,np.float32)/255
    mask.setflags(write=False)
    return mask


def baseline(frame):
    base=geometry(frame,1.02);h,w=base.shape[:2]
    return np.clip(base.astype(np.int16)+grid_field(h,w,10,1,24)[:,:,None],0,255).astype(np.uint8)

