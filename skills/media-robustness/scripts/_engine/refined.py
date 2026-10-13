"""Frozen feathered footer region shared by image and video transforms."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=4)
def region(h,w):
    x0,x1=round(230*w/1080),round(850*w/1080)
    y0,y1=round(1780*h/1920),round(1880*h/1920)
    yy,xx=np.mgrid[:y1-y0,:x1-x0].astype(np.float32)
    edge=np.minimum(np.minimum(xx,x1-x0-1-xx)/20,np.minimum(yy,y1-y0-1-yy)/10)
    weight=np.clip(edge,0,1)[:,:,None];weight.setflags(write=False)
    return (x0,x1,y0,y1),weight

