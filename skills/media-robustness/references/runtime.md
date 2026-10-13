# Runtime and outputs

Python 3.11+; NumPy, OpenCV and Pillow in the skill's `requirements.txt`; FFmpeg and ffprobe on PATH for video. Tested dependency versions are pinned for the first snapshot. Use a virtual environment if they are absent. No model downloads, remote inference or media uploads are performed by the processor.

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/python -m pip install -r /path/to/media-robustness/requirements.txt
```

The font resolver uses `--font` / `MEDIA_ROBUSTNESS_FONT`, then macOS Arial or installed DejaVu/Liberation fonts. Fonts are not bundled. A fallback font changes glyph pixels; compare runs using the same font. A supplied font path must exist. Reports record runtime/configuration for reproducibility.

## Supported media

- Video: MP4/MOV/MKV; portrait, even dimensions ≥96, square pixels, zero start time, verified constant frame rate, progressive supported 8-bit SDR formats. Source with missing color tags is recorded as assumed SDR, not certified SDR. HDR, VFR, rotated/mirrored/interlaced input and unsupported audio codecs fail explicitly. The first video and optional first audio streams are retained.
- Images: JPEG/PNG, opaque RGB; portrait dimensions ≥96. EXIF orientation is normalized, ICC is retained, other EXIF/GPS is not written. No implicit alpha flattening or HDR/tone-map conversion. Hard images require 1080×1920 after orientation normalization.
- Hard video's context cards and footer positions were designed for 9:16 portrait media. Other portrait ratios require visual review; landscape is rejected in this frozen version.

## Batch behavior

`--profiles` defaults to `production`; `--list` lists choices. `--kind auto` includes supported images/videos. Folder traversal is recursive; `--top-level` narrows scope. Each worker has separate state. `--jobs` is 1…3. Source files are hashed before and after. Destination must not exist or be inside the source tree. Filename conversion and generated file/directory collisions are checked before starting.

Output media are under `images/<profile>/<relative input path>` and `videos/<profile>/<relative path>.mp4`. JSON and logs are under `_reports/`. Status, source manifest, code hashes and media checksums make a completed run auditable. Run reports can contain private paths and should not be committed to the public repository.

There is no resume/overwrite mode. If a job fails, inspect its log; fix the specific input/dependency/problem and choose a new output location. Do not repeatedly rerun unsupported media or silently normalize it in ways the user did not request. Completed output is not evidence of target model failure.
