"""Shared geometry, measurements and file digests. No hardcoded paths."""
import hashlib
import math
import cv2
import numpy as np
from core import change_frame, phash

cv2.setNumThreads(1)

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def geometry(frame,zoom):
    if zoom==1:return frame
    h,w=frame.shape[:2];nw=round(w*zoom);nh=round(h*zoom)
    enlarged=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
    x=(nw-w)//2;y=(nh-h)//2
    return enlarged[y:y+h,x:x+w].copy()

def alter(frame,config):
    base=geometry(frame,config['zoom'])
    changed,info=change_frame(base,eps=config['eps'],target_bits=config['target_bits'],steps=180,reference_bits=phash(frame))
    return changed,info

def metrics(a,b):
    mse=cv2.norm(a,b,cv2.NORM_L2SQR)/a.size
    ag=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(np.float32)
    ua=cv2.GaussianBlur(ag,(11,11),1.5);ub=cv2.GaussianBlur(bg,(11,11),1.5)
    va=cv2.GaussianBlur(ag*ag,(11,11),1.5)-ua*ua
    vb=cv2.GaussianBlur(bg*bg,(11,11),1.5)-ub*ub
    cov=cv2.GaussianBlur(ag*bg,(11,11),1.5)-ua*ub
    s=((2*ua*ub+6.5025)*(2*cov+58.5225))/((ua*ua+ub*ub+6.5025)*(va+vb+58.5225))
    return {'phash_hamming_64':int(np.count_nonzero(phash(a)!=phash(b))),
            'psnr_rgb_db':10*math.log10(255**2/max(mse,1e-12)),'ssim_gray':float(s[5:-5,5:-5].mean())}

