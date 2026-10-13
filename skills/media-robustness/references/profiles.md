# Profiles in v1.0.0

| CLI name | Video | Still image |
|---|---|---|
| `production` | Historical `adaptive_smooth`, exposed as `control` | Same transform on one decoded image |
| `hard` | V06: static nonlinear geometry, enhanced grid, stabilized footer and brief context-frame replacements | Historical static `extreme`: grid + bounded DCT pHash transform |
| `multiscale` | V05 `multiscale_matched` | Same V05 transform on one image |
| `protected` | Production with contrast-edge attenuation and optional fixed rectangles | Same spatial protection on one image |
| `stable` | Production with temporal footer-color smoothing and hysteresis | First-frame behavior; no temporal benefit on a still |
| `encode-only` | Codec control, no pixel transform before encoding | Format/encoding control, no numeric pixel transform after orientation normalization |

The public names are stable aliases, not strength rankings. The hard image and hard video algorithms are distinct. Static-image hard is not a per-frame video mode: applying its independently optimized DCT fields to video could introduce visible temporal artifacts.

## Production

Zoom 1.02 with Lanczos interpolation and center crop; static diamond grid pitch 10 px, line width 1 px, maximum grid addition ±24 in 8-bit levels. Footer rows are `ZEUS ADVENTURES` and `MOBILE FANTASY GAME`, with font sizes 54/34 and y positions 1800/1852 scaled from 1080×1920. Opacity 20%; adaptive black/white at median 128. Footer region x230…850, y1780…1880, feathered; Gaussian sigma 6 at width 1080, blend .9.

Video uses H.264 CRF22, preset fast, yuv420p. Audio is stream-copied. The production pixel transform should match its frozen predecessor with the same library versions and font. Byte-identical video across FFmpeg versions is not promised.

## Multiscale V05

Grid pitches 7/13/23 and weights .25/.45/.30. A deterministic gain/offset lookup is fitted to the rounded field: peak remains ±24, candidates must keep the mean within .20 levels of production, then RMS error is minimized. Matching is approximate and depends on frame size.

At 1080×1920: production bare-grid RMS 12.0208; old V04 mixture 6.6118; V05 matched 12.2001, mean −.2545 versus production −.14. These describe the grid before image clipping, text and encoding. They do not measure model error or human visibility.

## Protected and stable

`protected` uses a contrast-edge proxy (maximum automatic weight .85), not a semantic face/text detector. A supplied rectangle restores the clean 1.02 zoom before encoding inside that rectangle. Example JSON:

```json
{"rectangles": [[0.2, 0.3, 0.8, 0.5]], "feather_px_1080": 12}
```

Coordinates are normalized `[left,top,right,bottom]` in the OUTPUT frame after zoom. Feather extends outward. Masks are static and must cover a moving object's full path; the example is illustrative, not automatically suitable for a source.

`stable` uses footer median EMA τ=.25 s, polarity hysteresis 116/140, color-transition τ=.18 s, and resets at detected scene changes (thumbnail mean difference>40 and histogram L1>.8). It changes no audio.

## Hard video V06

Zoom 1.015; smooth static displacement up to 11 px horizontally and 7 px vertically at 1080×1920, tapering at borders. Phase derives from the original SHA256. V05 grid ×1.30, capped ±31, attenuated by .50 × the edge-protection proxy. Historical footer opacity 20%, stabilized color, blur limited to glyph surroundings. H.264 CRF21.

By default first/last frames and one frame every 2 seconds are replaced with a color-matched abstract card: `AVENTURA DO TIGRE / JOGO DE FANTASIA / PARA CELULAR`. These historical sample labels are retained to reproduce the snapshot, not inferred from input content. No instructions to an analyzer are embedded. `--no-context` retains the spatial transform and disables all replacements. At 60 fps each replacement lasts 1/60 s; it can still be visible and changes the paused first/last image. Frame count, duration and audio remain unchanged. No target's sampling schedule is assumed or measured.

## Hard image, historical extreme

Only opaque RGB 1080×1920. Zoom 1.02; grids 7/10/13/23, weights .25/.25/.30/.20, grid bound 40 with 65% edge attenuation; footer 20%, blur around letters. DCT field bound 40, requested pHash distance 48/64, 220 optimization steps. Total difference is capped ±96 relative to the clean zoom BEFORE encoding. JPEG 95, 4:4:4; PNG outputs are lossless.

The pHash convention is BGR→gray→AREA 32×32→DCT-II upper 8×8; threshold is the median of the 63 non-DC coefficients. Changing this explicit hash is not evidence of defeating another hash or a semantic model. Requested distances can fail; inspect final decoded metrics.

## Provenance

- Production: accepted adaptive_smooth batch workflow, frozen 2026-10-05.
- Multiscale: V05 correction and validation, frozen 2026-10-08.
- Hard video: V06 custom experiment, frozen 2026-10-10.
- Hard image: static extreme profile, frozen 2026-10-06.
- Packaging: portable font discovery, one batch CLI, privacy-clean source tree and version 1.0.0 on 2026-10-13.

This provenance intentionally excludes media names, source paths and private run reports.
