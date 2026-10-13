"""Portable CFR/SDR video pipeline; requires ffmpeg, ffprobe and OpenCV.

The first video and, optionally, the first audio stream are retained. Video is
re-encoded; audio is copied. This module does not promise byte-identical video
across codec versions, or failure of a semantic/OCR/speech recognition model.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np

from media_utils import alter, metrics, sha


class UnsupportedVideoError(ValueError):
    """Input cannot be processed without unsupported timeline/color changes."""


def _json_command(command: list[str]) -> dict:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"{command[0]} failed: {proc.stderr.strip()[-6000:]}")
    return json.loads(proc.stdout)


def _probe(path: Path) -> dict:
    return _json_command(['ffprobe', '-v', 'error', '-show_streams',
                          '-show_format', '-of', 'json', str(path)])


def _fraction(value, description: str) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise UnsupportedVideoError(f'Missing/invalid {description}: {value}') from exc
    if result <= 0:
        raise UnsupportedVideoError(f'{description} must be positive: {value}')
    return result


def _check_display_metadata(stream: dict) -> None:
    tags = {str(k).lower(): v for k, v in stream.get('tags', {}).items()}
    for key in ('rotate', 'rotation'):
        if key in tags and abs(float(tags[key]) % 360) > 1e-9:
            raise UnsupportedVideoError('Rotated video is unsupported; normalize orientation first.')
    for side in stream.get('side_data_list', []):
        kind = str(side.get('side_data_type', '')).lower()
        if 'display matrix' in kind:
            rotation = float(side.get('rotation', 0))
            values = []
            for line in str(side.get('displaymatrix', '')).splitlines():
                if ':' in line:
                    values.extend(int(x) for x in line.split(':', 1)[1].split())
            identity = [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824]
            if abs(rotation % 360) > 1e-9 or values != identity:
                raise UnsupportedVideoError('Rotation/mirroring/display transforms are unsupported.')


def _check_sdr(metadata: dict) -> None:
    transfer = str(metadata.get('color_transfer', '')).lower()
    if transfer in {'smpte2084', 'arib-std-b67'}:
        raise UnsupportedVideoError(f'HDR transfer function {transfer} is unsupported.')
    for side in metadata.get('side_data_list', []):
        kind = str(side.get('side_data_type', '')).lower()
        if any(token in kind for token in ('mastering display', 'content light', 'dovi', 'dolby vision', 'hdr10', 'hdr dynamic')):
            raise UnsupportedVideoError(f'HDR metadata is unsupported: {kind}')


def _preflight(source: Path) -> tuple[dict, dict, dict | None, dict]:
    metadata = _probe(source)
    videos = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'video']
    if not videos:
        raise UnsupportedVideoError('No video stream found.')
    stream = videos[0]
    if stream.get('disposition', {}).get('attached_pic'):
        raise UnsupportedVideoError('The first video stream is attached artwork, not a timeline.')
    width, height = int(stream['width']), int(stream['height'])
    if width < 12 or height < 12 or width % 2 or height % 2:
        raise UnsupportedVideoError('H.264 yuv420p requires even dimensions; each dimension must be at least 12.')
    _check_display_metadata(stream)
    _check_sdr(stream)
    # Missing transfer tags are common in SDR files. Restrict such inputs to
    # ordinary 8-bit formats, and record the assumption rather than certify it.
    pixel_format = str(stream.get('pix_fmt', ''))
    allowed_pixels = {'yuv420p', 'yuv422p', 'yuv444p', 'yuvj420p', 'yuvj422p',
                      'yuvj444p', 'nv12', 'nv21', 'rgb24', 'bgr24', 'rgba',
                      'bgra', 'argb', 'abgr', 'gray', 'gbrp', 'yuva420p'}
    if pixel_format not in allowed_pixels:
        raise UnsupportedVideoError(f'Only supported 8-bit SDR pixel formats are accepted, got {pixel_format}.')
    if stream.get('field_order', 'unknown') not in ('progressive', 'unknown'):
        raise UnsupportedVideoError('Interlaced video is unsupported.')
    sar = str(stream.get('sample_aspect_ratio', '1:1'))
    if sar not in ('1:1', '0:1', 'N/A'):
        raise UnsupportedVideoError('Non-square pixel aspect ratio is unsupported.')
    fps = _fraction(stream.get('avg_frame_rate'), 'average frame rate')
    time_base = _fraction(stream.get('time_base'), 'video time base')
    period = Fraction(1, 1) / fps
    if time_base >= period / 4:
        raise UnsupportedVideoError('Video timestamp precision is too coarse to verify CFR.')
    if stream.get('start_time') not in (None, 'N/A') and abs(float(stream['start_time'])) > 1e-9:
        raise UnsupportedVideoError('Nonzero video start time is unsupported.')
    frames_data = _json_command([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=pts,best_effort_timestamp,duration,pkt_duration,color_transfer,interlaced_frame:frame_side_data',
        '-of', 'json', str(source)])
    frames = frames_data.get('frames', [])
    if not frames:
        raise UnsupportedVideoError('No decodable timestamped video frames found.')
    timestamps = []
    for index, frame in enumerate(frames):
        _check_sdr(frame)
        if frame.get('interlaced_frame', 0):
            raise UnsupportedVideoError(f'Interlaced frame {index} is unsupported.')
        value = frame.get('pts', frame.get('best_effort_timestamp'))
        if value is None:
            raise UnsupportedVideoError(f'Frame {index} has no usable presentation timestamp.')
        timestamp = int(value) * time_base
        if index == 0 and timestamp != 0:
            raise UnsupportedVideoError('The first video frame must have PTS zero.')
        if index and timestamp <= timestamps[-1]:
            raise UnsupportedVideoError('Repeated or non-monotonic frame timestamps are unsupported.')
        # One source tick permits normal Matroska timestamp quantization while
        # checking every frame against the same constant-rate timeline.
        if abs(timestamp - index * period) > time_base:
            raise UnsupportedVideoError(f'VFR/inconsistent timestamps detected at frame {index}.')
        timestamps.append(timestamp)
    count = len(frames)
    declared = stream.get('nb_frames')
    if declared not in (None, 'N/A') and int(declared) != count:
        raise UnsupportedVideoError('Declared frame count differs from decoded timestamp count.')
    duration = count * period
    declared_duration_delta = None
    if stream.get('duration') not in (None, 'N/A'):
        declared_duration_delta = float(stream['duration']) - float(duration)
        # MP4 edit lists/container timescales can trim the last display interval
        # by a fraction of a millisecond despite an exactly CFR PTS sequence.
        # Every frame PTS has already passed the stricter timeline check above.
        if abs(declared_duration_delta) > max(float(time_base), .001):
            raise UnsupportedVideoError('Video duration does not match its verified CFR frame timeline.')
    audio = next((s for s in metadata['streams'] if s.get('codec_type') == 'audio'), None)
    if audio and audio.get('codec_name') not in {'aac', 'mp3', 'ac3', 'eac3', 'alac', 'opus'}:
        raise UnsupportedVideoError(f"Audio codec {audio.get('codec_name')} is not supported for lossless stream-copy into this MP4 pipeline.")
    timeline = dict(width=width, height=height, frames=count, fps=str(fps),
                    duration_seconds=float(duration), time_base=str(time_base),
                    source_declared_duration_delta_seconds=declared_duration_delta,
                    all_frame_timestamps_checked=True, first_video_pts_zero=True,
                    cfr_verified=True, no_rotation=True, pixel_format=pixel_format,
                    color_transfer=stream.get('color_transfer', 'unspecified'),
                    sdr_status='declared_sdr' if stream.get('color_transfer') not in (None, 'unknown', 'unspecified') else 'assumed_sdr_8bit_no_hdr_metadata')
    return metadata, stream, audio, timeline


def _audio_payload_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                                 '-i', str(path), '-map', '0:a:0', '-c', 'copy', '-f', 'data', 'pipe:1'],
                                stdout=subprocess.PIPE, stderr=errors)
        try:
            for block in iter(lambda: proc.stdout.read(1024 * 1024), b''):
                digest.update(block)
            code = proc.wait()
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill(); proc.wait()
        if code:
            errors.seek(0)
            raise RuntimeError(errors.read().decode(errors='replace')[-6000:])
    return digest.hexdigest()


def _verify_audio(source: Path, encoded: Path, source_audio: dict | None, output_metadata: dict) -> dict:
    output_audio = next((s for s in output_metadata['streams'] if s.get('codec_type') == 'audio'), None)
    if source_audio is None:
        if output_audio is not None:
            raise RuntimeError('Unexpected output audio stream.')
        return {'present': False, 'payload_identical': None}
    if output_audio is None:
        raise RuntimeError('The source audio stream was lost.')
    for key in ('codec_name', 'sample_rate', 'channels'):
        if source_audio.get(key) != output_audio.get(key):
            raise RuntimeError(f'Copied audio changed {key}.')
    tolerance = max(1 / int(source_audio.get('sample_rate', 48000)), .0011)
    timing_checks = {}
    for key in ('start_time', 'duration'):
        a, b = source_audio.get(key), output_audio.get(key)
        if a not in (None, 'N/A') and b not in (None, 'N/A'):
            delta = float(b) - float(a)
            if abs(delta) > tolerance:
                raise RuntimeError(f'Copied audio {key} changed by {delta:.6f} seconds.')
            timing_checks[key + '_delta_seconds'] = delta
    first, second = _audio_payload_hash(source), _audio_payload_hash(encoded)
    if first != second:
        raise RuntimeError('Compressed audio payload is not identical after stream-copy.')
    return dict(present=True, payload_identical=True, source_payload_sha256=first,
                output_payload_sha256=second, codec=source_audio['codec_name'],
                sample_rate=source_audio.get('sample_rate'), channels=source_audio.get('channels'),
                timing_checks=timing_checks)


def process_video(source: Path, dest: Path, config: dict, transform=None) -> dict:
    """Create and verify a new MP4; return JSON-serializable measured results.

    transform(frame, config) returns (BGR frame, metrics); the legacy numeric
    transform is used when omitted. config accepts crf (0..51). Existing
    destinations are refused. A temporary MP4 is renamed only after all checks
    pass, so an interrupted/failed run does not publish a completed-looking file.
    """
    source, dest = Path(source).expanduser().resolve(), Path(dest).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in {'.mp4', '.mov', '.mkv'} or dest.suffix.lower() != '.mp4':
        raise UnsupportedVideoError('Sources must be .mp4/.mov/.mkv; output must be .mp4.')
    if source == dest or dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    for executable in ('ffmpeg', 'ffprobe'):
        if shutil.which(executable) is None:
            raise RuntimeError(f'{executable} is required on PATH.')
    crf = config.get('crf', 18)
    if not isinstance(crf, (int, float)) or not math.isfinite(crf) or not 0 <= crf <= 51:
        raise ValueError('crf must be a finite number between 0 and 51.')
    frame_transform = alter if transform is None else transform
    started = time.monotonic()
    _, _, source_audio, timeline = _preflight(source)
    source_hash = sha(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f'.{dest.stem}.', suffix='.partial.mp4', dir=dest.parent)
    os.close(handle)
    temporary = Path(name)
    cap = None
    proc = None
    pre_count = target_success = 0
    pre_distances, pre_psnr, pre_maxima = [], [], []
    try:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError('OpenCV could not open the source video.')
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostdin', '-y',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-video_size', f"{timeline['width']}x{timeline['height']}",
                   '-framerate', timeline['fps'], '-i', 'pipe:0', '-i', str(source),
                   '-map', '0:v:0', '-map', '1:a:0?', '-map_metadata', '-1', '-map_chapters', '-1',
                   '-c:v', 'libx264', '-preset', 'fast', '-crf', str(crf), '-threads', '2',
                   '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', '-c:a', 'copy',
                   '-movflags', '+faststart', str(temporary)]
        with tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.shape != (timeline['height'], timeline['width'], 3):
                        raise RuntimeError('Decoder changed native frame dimensions/orientation.')
                    changed, info = frame_transform(frame, config)
                    if changed.shape != frame.shape or changed.dtype != np.uint8:
                        raise RuntimeError('alter must return a same-size uint8 BGR frame.')
                    proc.stdin.write(changed.tobytes())
                    pre_count += 1
                    if pre_count == 1 or pre_count % 96 == 0:
                        print(f'VIDEO {source.name}: {pre_count}/{timeline["frames"]} frames', flush=True)
                    target_success += int(bool(info.get('target_achieved', False)))
                    pre_distances.append(float(info['phash_hamming_64']))
                    pre_psnr.append(float(info['psnr_rgb_db']))
                    pre_maxima.append(float(info['max_rgb_delta']))
                proc.stdin.close()
                code = proc.wait()
            except BaseException as exc:
                if proc.stdin and not proc.stdin.closed:
                    try: proc.stdin.close()
                    except BrokenPipeError: pass
                if proc.poll() is None:
                    proc.kill(); proc.wait()
                if isinstance(exc, BrokenPipeError):
                    errors.seek(0)
                    raise RuntimeError('FFmpeg encoder failed: ' + errors.read().decode(errors='replace')[-6000:]) from exc
                raise
            if code:
                errors.seek(0)
                raise RuntimeError('FFmpeg encoder failed: ' + errors.read().decode(errors='replace')[-6000:])
        cap.release(); cap = None
        if pre_count != timeline['frames']:
            raise RuntimeError(f"OpenCV decoded {pre_count} frames; ffprobe found {timeline['frames']}.")
        final = _probe(temporary)
        video = next(s for s in final['streams'] if s.get('codec_type') == 'video')
        if (int(video['width']), int(video['height']), int(video['nb_frames'])) != (timeline['width'], timeline['height'], timeline['frames']):
            raise RuntimeError('Encoded dimensions/frame count changed.')
        if Fraction(video['avg_frame_rate']) != Fraction(timeline['fps']):
            raise RuntimeError('Encoded frame rate changed.')
        duration_delta = float(video['duration']) - timeline['duration_seconds']
        if abs(duration_delta) > 1e-5 or abs(float(video.get('start_time', 0))) > 1e-9:
            raise RuntimeError('Encoded video duration or start time changed.')
        check = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-xerror',
                                '-threads', '2', '-i', str(temporary), '-f', 'null', '-'],
                               capture_output=True, text=True)
        if check.returncode:
            raise RuntimeError('Final decode failed: ' + check.stderr[-6000:])
        audio = _verify_audio(source, temporary, source_audio, final)
        left, right = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(temporary))
        samples = []
        try:
            for fraction in (.1, .3, .5, .7, .9):
                index = min(pre_count - 1, round((pre_count - 1) * fraction))
                left.set(cv2.CAP_PROP_POS_FRAMES, index); right.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok_a, a = left.read(); ok_b, b = right.read()
                if not (ok_a and ok_b):
                    raise RuntimeError(f'Could not decode comparison frame {index}.')
                samples.append(dict(frame=index, fraction=fraction, **metrics(a, b)))
        finally:
            left.release(); right.release()
        means = {key: float(np.mean([r[key] for r in samples]))
                 for key in ('phash_hamming_64', 'psnr_rgb_db', 'ssim_gray')}
        if dest.exists():
            raise FileExistsError(f'Destination appeared during processing: {dest}')
        output_hash = sha(temporary)
        result = dict(source=str(source), output=str(dest), settings=dict(config),
                      source_sha256=source_hash, output_sha256=output_hash,
                      output_bytes=temporary.stat().st_size,
                      processing_seconds=time.monotonic() - started,
                      preencode=dict(frame_count=pre_count, target_achieved_frames=target_success,
                                     mean_phash_hamming_64=float(np.mean(pre_distances)),
                                     mean_psnr_rgb_db=float(np.mean(pre_psnr)),
                                     max_rgb_delta=max(pre_maxima),
                                     rgb_bound_after_geometry=config.get('eps'),
                                     bound_note=('Bound and pre-encode PSNR are relative to the geometrically transformed frame; encoding adds losses.'
                                                 if transform is None else
                                                 'Pre-encode metrics are reported by the selected transform; its summary defines the reference. Encoding adds losses.')),
                      final_metrics=dict(samples=samples, means=means,
                                         **{'mean_' + k: v for k, v in means.items()}),
                      video=dict(**timeline, frame_count_verified=True, full_decode_ok=True,
                                 video_duration_delta_seconds=duration_delta, codec='h264', pixel_format_output='yuv420p'),
                      audio=audio, semantic_or_ocr_or_asr_success_verified=False,
                      ffmpeg_version=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
                      codec_version_byte_reproducibility_claimed=False)
        temporary.replace(dest)
        return result
    finally:
        if cap is not None:
            cap.release()
        if proc is not None and proc.poll() is None:
            proc.kill(); proc.wait()
        if temporary.exists():
            temporary.unlink()
