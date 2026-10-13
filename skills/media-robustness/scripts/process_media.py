#!/usr/bin/env python3
"""Create measured media variants in a fresh output directory.

All profiles are deterministic experiments. Their measurements do not establish
human invisibility or a failure of OCR, speech, or semantic recognition systems.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unicodedata

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_engine"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}
ENGINE_PROFILES = {
    "production": "control",
    "multiscale": "multiscale_matched",
    "protected": "protected",
    "stable": "stable_text",
    "encode-only": "encode_only",
}
PROFILES = {
    "production": "Established zoom, grid and bottom-label transform.",
    "multiscale": "Multiscale grid with approximately matched mean and RMS.",
    "protected": "Edge attenuation and optional manual region protection.",
    "stable": "Production with temporal text-polarity smoothing for video.",
    "encode-only": "Identity pixels before encoding; useful as a control.",
    "hard": "Stronger static transform or seeded video warp and context frames.",
}
METRICS_NOTE = (
    "Metrics compare with the original decoded, orientation-normalized pixels, "
    "including geometry. Grid amplitude limits are not full-transform bounds. "
    "Encoding may add losses. Null PSNR denotes identical pixels or a nonfinite metric."
)


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_safe(value):
    """Keep reports standard JSON and remove inherited detector-success claims."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()
                if key not in {"target_achieved", "target_achieved_frames",
                               "requested_phash_distance_achieved"}}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_text(value) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json_text(value))
    temporary.replace(path)


def lexists(path: Path) -> bool:
    return os.path.lexists(path)


def classify(path: Path) -> str | None:
    suffix = path.suffix.lower()
    return "image" if suffix in IMAGE_EXTENSIONS else "video" if suffix in VIDEO_EXTENSIONS else None


def output_relative(relative: Path, kind: str) -> Path:
    suffix = ".mp4" if kind == "video" else ".png" if relative.suffix.lower() == ".png" else ".jpg"
    return relative.with_suffix(suffix)


def job_paths(output: Path, relative: Path, kind: str, profile: str) -> tuple[Path, Path, Path]:
    converted = output_relative(relative, kind)
    dest = output / ("videos" if kind == "video" else "images") / profile / converted
    report = (output / "_reports" / kind / profile / converted).with_suffix(".json")
    return dest, report, report.with_suffix(".log")


def validate_paths(output: Path, rows: list[dict], profiles: list[str]) -> None:
    """Reject duplicate files and files that would become generated directories.

    Use case-insensitive, Unicode-normalized comparison so a plan remains safe
    when moved between common Linux, macOS and Windows filesystems.
    """
    planned = []
    for row in rows:
        for profile in profiles:
            dest, report, log = job_paths(output, Path(row["relative"]), row["kind"], profile)
            planned.extend((dest, report, log, report.with_suffix(".json.tmp")))
    for name in ("config.json", "code_sha256.json", "source_manifest.json", "status.json", "checksums.json"):
        path = output / "_reports" / name
        planned.extend((path, path.with_suffix(".json.tmp")))
    planned.append(output / "_reports" / "SHA256SUMS.txt")

    def key(path):
        return unicodedata.normalize("NFC", str(path)).casefold()

    seen = {}
    for path in planned:
        normalized = key(path)
        if normalized in seen:
            raise ValueError(f"Generated output paths collide: {seen[normalized]} and {path}")
        seen[normalized] = path
    for path in planned:
        for parent in path.parents:
            if key(parent) in seen:
                raise ValueError(f"Generated output file/directory collision: {seen[key(parent)]} and {path}")


def read_image(path: Path):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError(f"Only JPEG/PNG image contents are supported: {path}")
        expected_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        if image.format != expected_format:
            raise ValueError(f"Image extension does not match its contents: {path}")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Animated/multiframe images are unsupported: {path}")
        image.load()
        if image.mode != "RGB" or "transparency" in image.info:
            raise ValueError(f"Only opaque 8-bit RGB images are supported: {path}")
        exif = image.getexif()
        orientation = exif.get(274, 1)
        if not isinstance(orientation, int) or isinstance(orientation, bool) or orientation not in range(1, 9):
            raise ValueError(f"Invalid EXIF orientation {orientation!r}: {path}")
        icc = image.info.get("icc_profile")
        normalized = ImageOps.exif_transpose(image)
        pixels = np.array(normalized, dtype=np.uint8)[:, :, ::-1].copy()
        metadata = dict(width=normalized.width, height=normalized.height,
                        input_format=image.format, input_exif_orientation=orientation,
                        orientation_normalized=orientation != 1,
                        icc_present=icc is not None,
                        icc_sha256=hashlib.sha256(icc).hexdigest() if icc is not None else None)
    return pixels, icc, metadata


def validate_dimensions(width: int, height: int, kind: str, profiles: list[str]) -> None:
    if min(width, height) < 96 or height <= width:
        raise ValueError("Portrait inputs with both dimensions >=96 pixels are required.")
    if kind == "image" and "hard" in profiles and (width, height) != (1080, 1920):
        raise ValueError("The hard image profile requires orientation-normalized 1080x1920 input.")


def verify_source(row: dict) -> None:
    if sha(Path(row["source"])) != row["source_sha256"]:
        raise RuntimeError(f"Source changed during processing: {row['source']}")


def measured(a, b) -> dict:
    import numpy as np
    from media_utils import metrics

    result = metrics(a, b)
    identical = bool(np.array_equal(a, b))
    result["pixels_identical"] = identical
    if identical:
        result["psnr_rgb_db"] = None
    result["max_rgb_delta"] = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
    return result


def process_image(job: dict) -> dict:
    import numpy as np
    from PIL import Image
    from variants import VideoTransform, protection_from_spec

    row, profile = job["source"], job["profile"]
    source, dest = Path(row["source"]), Path(job["output"])
    verify_source(row)
    original, icc, metadata = read_image(source)
    validate_dimensions(metadata["width"], metadata["height"], "image", [profile])
    if profile == "hard":
        from static_transform import PROFILES as STATIC_PROFILES, transform
        changed, info = transform(original, STATIC_PROFILES["extreme"])
        experiment = dict(profile="extreme", settings=STATIC_PROFILES["extreme"])
    else:
        mask = protection_from_spec(job["protection"], *original.shape[:2]) if profile == "protected" else None
        engine = VideoTransform(ENGINE_PROFILES[profile], 1.0, mask)
        changed, info = engine(original)
        experiment = engine.summary()
    if changed.shape != original.shape or changed.dtype != np.uint8:
        raise RuntimeError("Image transform must return same-size uint8 BGR pixels.")
    preencode = dict(measured(original, changed), transform_details=info, bound_note=METRICS_NOTE)
    if preencode["pixels_identical"]:
        info["psnr_rgb_db"] = None
    verify_source(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(changed[:, :, ::-1])
    encoding = {"icc_profile": icc} if icc is not None else {}
    is_png = dest.suffix == ".png"
    if not is_png:
        encoding.update(quality=95, subsampling=0, optimize=True)
    # Exclusive creation prevents overwrite, including a concurrent destination.
    try:
        with dest.open("xb") as stream:
            image.save(stream, format="PNG" if is_png else "JPEG", **encoding)
        decoded, output_icc, output_metadata = read_image(dest)
        with Image.open(dest) as check:
            if check.getexif():
                raise RuntimeError("Output unexpectedly contains EXIF metadata.")
        if decoded.shape != original.shape or output_icc != icc:
            raise RuntimeError("Image dimensions or ICC profile changed during encoding.")
        if is_png and not np.array_equal(decoded, changed):
            raise RuntimeError("PNG encoding did not preserve transformed pixels losslessly.")
        verify_source(row)
        return dict(source=str(source), output=str(dest), source_sha256=row["source_sha256"],
                    output_sha256=sha(dest), output_bytes=dest.stat().st_size,
                    source_hash_unchanged=True, image=dict(metadata, full_decode_ok=True,
                    dimensions_verified=True, output_format=output_metadata["input_format"],
                    icc_preserved=True, exif_removed=True,
                    encoding={"lossless": True} if is_png else {"quality": 95, "subsampling": 0}),
                    preencode=preencode, final_metrics=measured(original, decoded), experiment=experiment)
    except BaseException:
        # Failed items are recorded in status.json; a partial image is not a result.
        # Never remove a pre-existing file if exclusive creation itself failed.
        if "stream" in locals() and stream.closed and dest.is_file():
            dest.unlink()
        raise


def process_video(job: dict) -> dict:
    import video_pipeline
    from variants import VideoTransform, protection_from_spec

    row, profile = job["source"], job["profile"]
    source, dest = Path(row["source"]), Path(job["output"])
    verify_source(row)
    _, _, _, timeline = video_pipeline._preflight(source)
    validate_dimensions(timeline["width"], timeline["height"], "video", [profile])
    fps = float(Fraction(timeline["fps"]))
    if profile == "hard":
        from tiger_transform import SETTINGS, TigerTransform
        engine = TigerTransform(timeline["fps"], timeline["frames"], row["source_sha256"],
                                context_enabled=not job["no_context"])
        settings = dict(SETTINGS, context_enabled=not job["no_context"])
    else:
        mask = protection_from_spec(job["protection"], timeline["height"], timeline["width"]) if profile == "protected" else None
        engine = VideoTransform(ENGINE_PROFILES[profile], fps, mask)
        settings = dict(profile=ENGINE_PROFILES[profile], crf=22, protection=job["protection"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The legacy pipeline renames its temporary output. Keep that rename inside
    # a private directory; publish with an exclusive hard link after validation.
    # The staging directory lives on the destination filesystem.
    with tempfile.TemporaryDirectory(prefix=".media-robustness-", dir=dest.parent) as temporary:
        staged = Path(temporary) / "verified.mp4"
        result = video_pipeline.process_video(source, staged, settings, transform=engine)
        verify_source(row)
        summary = engine.summary()
        if profile == "hard":
            if (summary["processed_frames"] != timeline["frames"] or
                    summary["applied_replacements"] != summary["scheduled_replacements"]):
                raise RuntimeError("Hard video transform frame accounting failed.")
        elif summary["frames"] != timeline["frames"]:
            raise RuntimeError("Video transform frame accounting failed.")
        if not (result["source_sha256"] == row["source_sha256"] and
                result["output_sha256"] == sha(staged) and result["video"]["full_decode_ok"] and
                result["video"]["frame_count_verified"] and
                (not result["audio"]["present"] or result["audio"]["payload_identical"])):
            raise RuntimeError("Video verification failed.")
        os.link(staged, dest)
    result["output"] = str(dest)
    result["preencode"].pop("rgb_bound_after_geometry", None)
    result["preencode"]["bound_note"] = METRICS_NOTE
    if profile == "encode-only":
        result["preencode"].update(mean_psnr_rgb_db=None, pixels_identical=True)
    result.update(source_hash_unchanged=True, experiment=summary)
    return result


def worker(job: dict) -> None:
    dest, report = Path(job["output"]), Path(job["report"])
    if lexists(dest) or lexists(report):
        raise FileExistsError("Worker destination or report already exists.")
    result = process_image(job) if job["source"]["kind"] == "image" else process_video(job)
    result.update(version=VERSION, profile=job["profile"], kind=job["source"]["kind"],
                  config_sha256=job["config_sha256"], no_context_requested=job["no_context"],
                  context_enabled=(not job["no_context"]) if job["profile"] == "hard" and job["source"]["kind"] == "video" else None,
                  human_invisibility_verified=False, semantic_or_ocr_or_asr_success_verified=False)
    save(report, result)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, help="Original file or directory; directories are recursive by default")
    result.add_argument("--output", type=Path, help="NEW output directory outside the input")
    result.add_argument("--profiles", nargs="+", choices=PROFILES, default=["production"])
    result.add_argument("--kind", choices=("auto", "image", "video"), default="auto")
    result.add_argument("--jobs", type=int, choices=(1, 2, 3), default=2)
    result.add_argument("--top-level", action="store_true", help="Only inspect immediate input directory files")
    result.add_argument("--font", type=Path, help="Override the installed font; sets MEDIA_ROBUSTNESS_FONT")
    result.add_argument("--protect", type=Path, help="Rectangle JSON; requires the protected profile")
    result.add_argument("--no-context", action="store_true", help="Disable context replacement frames in hard videos only")
    result.add_argument("--list", action="store_true", help="List profiles without processing")
    result.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return result


def prepare(args) -> tuple[Path, list[dict], list[str], dict | None, dict]:
    if args.input is None or args.output is None:
        raise ValueError("--input and --output are required unless --list is used.")
    profiles = list(dict.fromkeys(args.profiles))
    if args.protect and "protected" not in profiles:
        raise ValueError("--protect requires selecting the protected profile.")
    source = args.input.expanduser().resolve()
    # Check the spelled path too: resolve() otherwise hides dangling symlinks.
    spelled_output = args.output.expanduser().absolute()
    output = spelled_output.resolve()
    if lexists(spelled_output) or lexists(output):
        raise ValueError("Output already exists. Use a NEW directory; nothing is overwritten.")
    if not source.exists() or not (source.is_dir() or source.is_file()):
        raise ValueError(f"Input is not an existing file or directory: {source}")
    if source == output or source.is_dir() and source in output.parents:
        raise ValueError("Output must be outside the input directory.")
    base = source if source.is_dir() else source.parent
    candidates = (source.glob("*") if args.top_level else source.rglob("*")) if source.is_dir() else [source]
    files = sorted(path for path in candidates if path.is_file() and classify(path) and
                   (args.kind == "auto" or classify(path) == args.kind))
    if not files:
        raise ValueError("No supported media found for the selected kind (JPEG, PNG, MP4, MOV, MKV).")
    if any(path.is_symlink() for path in files):
        raise ValueError("Symlink media inputs are unsupported; use original files.")
    rows = [dict(source=str(path), relative=str(path.relative_to(base)), kind=classify(path)) for path in files]
    validate_paths(output, rows, profiles)
    spec = None
    if args.protect:
        from variants import protection_from_spec
        spec = json.loads(args.protect.expanduser().read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value: {value}")))
        # Validate once before creating any output; each worker scales the mask.
        if spec is None:
            raise ValueError("Protection JSON must be an object, not null.")
        protection_from_spec(spec, 1920, 1080)
    font_value = str(args.font.expanduser().resolve()) if args.font else os.environ.get("MEDIA_ROBUSTNESS_FONT")
    font_info = None
    if not font_value and any(profile != "encode-only" for profile in profiles):
        from font_utils import font_path
        font_value = str(font_path())
    if font_value:
        from PIL import ImageFont
        font = Path(font_value).expanduser().resolve()
        if not font.is_file():
            raise ValueError(f"Font override does not exist: {font}")
        ImageFont.truetype(str(font), 24)
        os.environ["MEDIA_ROBUSTNESS_FONT"] = str(font)
        font_info = dict(path=str(font), sha256=sha(font))
    for row in rows:
        path = Path(row["source"])
        row["source_sha256"] = sha(path)
        row["source_bytes"] = path.stat().st_size
        if row["kind"] == "image":
            _, _, metadata = read_image(path)
        else:
            import video_pipeline
            _, _, _, metadata = video_pipeline._preflight(path)
            if "hard" in profiles and not args.no_context and metadata["frames"] < 2:
                raise ValueError("Hard video context frames require at least two source frames.")
        validate_dimensions(metadata["width"], metadata["height"], row["kind"], profiles)
        row["preflight"] = metadata
        verify_source(row)
    import cv2
    runtime = dict(python=platform.python_version(), system=platform.system(),
                   machine=platform.machine(), opencv=cv2.__version__,
                   numpy=importlib.metadata.version("numpy"),
                   pillow=importlib.metadata.version("Pillow"))
    if any(row["kind"] == "video" for row in rows):
        for executable in ("ffmpeg", "ffprobe"):
            runtime[executable] = subprocess.run(
                [executable, "-version"], capture_output=True, text=True,
                check=True, timeout=15).stdout.splitlines()[0]
    config = dict(version=VERSION, runtime=runtime, input=str(source), kind=args.kind, profiles=profiles, jobs=args.jobs,
                  top_level=args.top_level, protection=spec, no_context_requested=args.no_context,
                  no_context_scope="hard video only", font_override=font_info,
                  engine_profiles=dict(ENGINE_PROFILES, hard={"image": "extreme", "video": "tiger_custom_v06"}),
                  image_encoding={"jpeg": {"quality": 95, "subsampling": 0}, "png": "lossless"},
                  metrics_note=METRICS_NOTE)
    return output, rows, profiles, spec, config


def checksum_line(digest: str, relative: str) -> str:
    escaped = relative.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
    return ("\\" if escaped != relative else "") + digest + "  " + escaped + "\n"


def run(args) -> int:
    output, inventory, profiles, protection, config = prepare(args)
    code_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sorted(ROOT.rglob("*"))
                   if path.is_file() and path.suffix.lower() in {".py", ".ttf", ".otf"}}
    config_hash = hashlib.sha256(json_text(config).encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    reports = output / "_reports"
    reports.mkdir()
    save(reports / "config.json", dict(config, config_sha256=config_hash))
    save(reports / "code_sha256.json", dict(version=VERSION, files=code_hashes))
    save(reports / "source_manifest.json", dict(version=VERSION, sources=inventory))
    state = dict(version=VERSION, status="running", total=len(inventory) * len(profiles), done=0, failed=0,
                 config_sha256=config_hash, started_utc=datetime.now(timezone.utc).isoformat(), results=[])
    save(reports / "status.json", state)

    def execute(row, profile):
        dest, report, log = job_paths(output, Path(row["relative"]), row["kind"], profile)
        log.parent.mkdir(parents=True, exist_ok=True)
        job = dict(source=row, output=str(dest), report=str(report), profile=profile,
                   protection=protection if profile == "protected" else None,
                   no_context=args.no_context, config_sha256=config_hash)
        # Each clip has a fresh interpreter, including all temporal and cache state.
        with log.open("x", encoding="utf-8") as stream:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker"],
                                  input=json_text(job), text=True, stdout=stream, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"Worker exited {proc.returncode}; see {log}")
        result = json.loads(report.read_text(encoding="utf-8"))
        verify_source(row)
        if not (result["source_sha256"] == row["source_sha256"] and
                result["output_sha256"] == sha(dest) and result["source_hash_unchanged"] and
                result["config_sha256"] == config_hash):
            raise RuntimeError(f"Final file/hash verification failed: {dest}")
        return dict(status="verified", kind=row["kind"], profile=profile, relative=row["relative"],
                    output=str(dest), output_sha256=result["output_sha256"], report=str(report))

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(execute, row, profile): (row, profile) for row in inventory for profile in profiles}
        for future in as_completed(futures):
            row, profile = futures[future]
            try:
                item = future.result()
                state["done"] += 1
            except Exception as exc:
                item = dict(status="failed", kind=row["kind"], profile=profile, relative=row["relative"], error=str(exc))
                state["failed"] += 1
            state["results"].append(item)
            save(reports / "status.json", state)
            print(f"VERIFIED {state['done']}/{state['total']}; FAILED {state['failed']}: {profile} / {row['relative']}", flush=True)
    changed = []
    for row in inventory:
        try:
            verify_source(row)
        except (OSError, RuntimeError) as exc:
            changed.append(dict(source=row["source"], error=str(exc)))
    state.update(status="complete" if not state["failed"] and not changed else "failed",
                 source_hashes_unchanged=not changed, source_changes=changed,
                 finished_utc=datetime.now(timezone.utc).isoformat())
    save(reports / "status.json", state)
    # Include media and reports; exclude the checksum files themselves.
    checksums = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()}
    save(reports / "checksums.json", dict(version=VERSION, files=checksums))
    with (reports / "SHA256SUMS.txt").open("x", encoding="utf-8") as stream:
        stream.write("".join(checksum_line(digest, relative) for relative, digest in checksums.items()))
    print(f"BATCH {state['status'].upper()}", flush=True)
    return int(state["status"] != "complete")


def main(argv=None) -> int:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args._worker:
        worker(json.load(sys.stdin))
        return 0
    if args.list:
        for name, description in PROFILES.items():
            print(f"{name}: {description}")
        return 0
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        arg_parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
