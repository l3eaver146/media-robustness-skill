"""Frozen hard-video experiment; numeric behavior is preserved from V06."""
from functools import lru_cache
from fractions import Fraction
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw
from font_utils import load_font

from variants import _matched_grid, _automatic_protection, _glyph_support, VideoTransform
from variants import bottom_mask, refined, phash

SETTINGS = dict(profile='tiger_custom_v06', crf=21, zoom=1.015,
                warp_x_px_at_1080=11., warp_y_px_at_1920=7.,
                grid_gain=1.30, grid_limit=31, edge_attenuation=.50,
                text_opacity=.20, blur_scope='glyph_vicinity',
                context_cadence_seconds=2, context_frame_count_per_event=1,
                replace_first_and_last=True,
                context_label='AVENTURA DO TIGRE / JOGO DE FANTASIA / PARA CELULAR')


def replacement_indices(total, fps):
    """Single-frame replacements, no timeline insertions or speed changes."""
    rate = Fraction(str(fps))
    if total < 2 or rate <= 0:
        raise ValueError('Expected at least two frames and positive FPS.')
    result = {0, total-1}
    seconds = SETTINGS['context_cadence_seconds']
    while seconds * rate < total-1:
        k = round(seconds * rate)
        # Keep boundaries separate; no assumptions about target sampling times.
        if 0 < k < total-1:
            result.add(k)
        seconds += SETTINGS['context_cadence_seconds']
    return tuple(sorted(result))


@lru_cache(maxsize=8)
def warp_maps(h, w, phase):
    y, x = np.mgrid[:h,:w].astype(np.float32)
    nx, ny = x/max(1,w-1), y/max(1,h-1)
    cx, cy = (w-1)/2, (h-1)/2
    # Smooth static displacements taper to zero at borders. One interpolation.
    taper = np.sin(np.pi*nx)**2 * np.sin(np.pi*ny)**2
    dx = SETTINGS['warp_x_px_at_1080']*w/1080*taper*np.sin(4*np.pi*ny+phase)
    dy = SETTINGS['warp_y_px_at_1920']*h/1920*taper*np.sin(4*np.pi*nx-phase)
    mx = (cx+(x-cx)/SETTINGS['zoom']+dx).astype(np.float32)
    my = (cy+(y-cy)/SETTINGS['zoom']+dy).astype(np.float32)
    mx.setflags(write=False); my.setflags(write=False)
    return mx,my


@lru_cache(maxsize=4)
def context_text(h,w):
    mask=Image.new('L',(w,h));draw=ImageDraw.Draw(mask)
    for words,y,size in [('AVENTURA',.35,106),('DO TIGRE',.43,106),
                         ('JOGO DE FANTASIA',.55,62),('PARA CELULAR',.62,58)]:
        font=load_font(max(8,round(size*w/1080)))
        draw.text((w//2,round(y*h)),words,font=font,anchor='mm',fill=255,stroke_width=1)
    a=np.asarray(mask,dtype=np.float32)/255
    a.setflags(write=False)
    return a


def context_frame(frame):
    """Color-matched abstract title card; still potentially visible to humans."""
    h,w=frame.shape[:2]
    low=cv2.resize(frame,(4,7),interpolation=cv2.INTER_AREA)
    low=cv2.GaussianBlur(low,(3,3),.8)
    card=cv2.resize(low,(w,h),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    gray=cv2.cvtColor(card,cv2.COLOR_BGR2GRAY)
    # No all-white flash. Match local polarity; no instruction to an analyzer.
    target=np.where(gray>=128,18.,237.).astype(np.float32)
    glyph=context_text(h,w)[:,:,None]
    return np.clip(np.rint(card+.90*glyph*(target[:,:,None]-card)),0,255).astype(np.uint8)


class TigerTransform:
    def __init__(self, fps, total_frames, seed, context_enabled=True):
        self.fps=Fraction(str(fps));self.total=int(total_frames)
        self.schedule=replacement_indices(self.total,self.fps) if context_enabled else ()
        self.phase=(int(seed[:8],16)%10000)/10000*2*math.pi
        self.footer=VideoTransform('stable_text',float(self.fps))
        self.frames=0;self.changed_frames=0;self.applied=[];self.shape=None
        self.mean_grid_rms=0.;self.max_grid_delta=0
        self.replacement_mean_luma_delta=[]

    def spatial(self,frame):
        h,w=frame.shape[:2]
        mx,my=warp_maps(h,w,self.phase)
        clean=cv2.remap(frame,mx,my,cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
        grid=_matched_grid(h,w)[0].astype(np.float32)
        # Edge protection is a contrast proxy, not a text/face detector.
        protect=_automatic_protection(clean)
        field=np.rint(np.clip(grid*SETTINGS['grid_gain']*(1-SETTINGS['edge_attenuation']*protect),
                             -SETTINGS['grid_limit'],SETTINGS['grid_limit'])).astype(np.int16)
        self.mean_grid_rms+=float(np.sqrt(np.mean(field.astype(np.float32)**2)))
        self.max_grid_delta=max(self.max_grid_delta,int(np.abs(field).max()))
        hard=np.clip(clean.astype(np.int16)+field[:,:,None],0,255).astype(np.uint8)
        target=self.footer._target_color(hard)
        (x0,x1,y0,y1),weight=refined.region(h,w)
        patch=hard[y0:y1,x0:x1].astype(np.float32)
        smooth=cv2.GaussianBlur(patch,(0,0),6*w/1080)
        weight=weight*_glyph_support(h,w)[y0:y1,x0:x1,None]
        patch+=(smooth-patch)*weight*.9
        glyph=bottom_mask(h,w)[y0:y1,x0:x1,None]
        out=hard.copy()
        out[y0:y1,x0:x1]=np.clip(np.rint(patch+SETTINGS['text_opacity']*glyph*(target-patch)),0,255).astype(np.uint8)
        return out

    def __call__(self,frame,config=None):
        if (frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3
                or min(frame.shape[:2])<96):
            raise ValueError('Expected uint8 BGR frames >=96 px.')
        if self.shape is not None and frame.shape!=self.shape:
            raise ValueError('Frame dimensions changed.')
        if self.frames>=self.total:
            raise ValueError('More frames than verified source count.')
        self.shape=frame.shape
        index=self.frames
        out=self.spatial(frame)
        if index in self.schedule:
            card=context_frame(out)
            self.replacement_mean_luma_delta.append(float(cv2.cvtColor(card,cv2.COLOR_BGR2GRAY).mean()-cv2.cvtColor(out,cv2.COLOR_BGR2GRAY).mean()))
            out=card;self.applied.append(index)
        diff=cv2.absdiff(frame,out)
        maximum=int(diff.max())
        self.changed_frames+=int(maximum>0);self.frames+=1
        mse=cv2.norm(frame,out,cv2.NORM_L2SQR)/frame.size
        return out,dict(phash_hamming_64=int(np.count_nonzero(phash(frame)!=phash(out))),
                        psnr_rgb_db=10*math.log10(255**2/max(mse,1e-12)),max_rgb_delta=maximum)

    def summary(self):
        return dict(profile=SETTINGS['profile'],processed_frames=self.frames,
                    actual_changed_frames_preencode=self.changed_frames,
                    expected_frames=self.total,scheduled_replacements=list(self.schedule),
                    applied_replacements=list(self.applied),
                    replacement_times_seconds=[float(Fraction(i,1)/self.fps) for i in self.applied],
                    replaced_fraction=len(self.applied)/self.total,
                    frame_duration_seconds=float(1/self.fps),warp_phase=self.phase,
                    max_grid_delta=self.max_grid_delta,
                    mean_bare_grid_rms_after_edge_attenuation=self.mean_grid_rms/max(1,self.frames),
                    replacement_mean_luma_delta=list(self.replacement_mean_luma_delta),
                    footer_polarity_switches=self.footer.polarity_switches,
                    target_system_success_verified=False,human_invisibility_verified=False,
                    metrics_reference='Original decoded source, including geometry and replacements; grid bound excludes geometry, titles and encoding.')
