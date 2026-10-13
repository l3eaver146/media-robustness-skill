"""CLI contract checks using generated media only; no private assets are read."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms

SCRIPT = Path(__file__).resolve().parents[1] / "skills/media-robustness/scripts/process_media.py"
spec = importlib.util.spec_from_file_location("process_media_cli", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def image_file(path, size=(108, 192), mode="RGB", exif=None, icc=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    y, x = np.mgrid[:height, :width]
    data = np.stack(((x * 3 + y) % 256, (x + y * 2) % 256, (x * 2 + y * 3) % 256), axis=2).astype(np.uint8)
    image = Image.fromarray(data).convert(mode)
    options = {}
    if exif is not None:
        options["exif"] = exif
    if icc is not None:
        options["icc_profile"] = icc
    image.save(path, **options)
    return path


def strict_json(path):
    def invalid(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="media-robustness-cli-test-")
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.output = self.root / "result"

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *arguments, expected=0):
        environment = dict(os.environ)
        environment.pop("MEDIA_ROBUSTNESS_FONT", None)
        result = subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                                text=True, capture_output=True, env=environment)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def fail_cli(self, *arguments, contains=None):
        result = self.run_cli("--input", self.input, "--output", self.output,
                              *arguments, expected=2)
        self.assertFalse(self.output.exists(), result.stdout + result.stderr)
        if contains:
            self.assertIn(contains.lower(), result.stderr.lower())

    def test_list_does_not_require_paths_or_engine_imports(self):
        result = self.run_cli("--list")
        self.assertEqual({line.split(":")[0] for line in result.stdout.splitlines()}, set(cli.PROFILES))

    def test_empty_and_unsupported_input_create_nothing(self):
        self.fail_cli(contains="No supported media")
        (self.input / "document.txt").write_text("synthetic non-media")
        self.fail_cli(contains="No supported media")
        self.run_cli("--input", self.input / "document.txt", "--output", self.output, expected=2)
        self.assertFalse(self.output.exists())

    def test_wrong_profile_protection_and_invalid_arguments(self):
        image_file(self.input / "sample.png")
        protection = self.root / "protect.json"
        protection.write_text('{"rectangles": []}')
        self.fail_cli("--protect", protection, contains="protected profile")
        self.fail_cli("--jobs", "4", contains="invalid choice")
        self.fail_cli("--profiles", "unknown", contains="invalid choice")
        self.fail_cli("--kind", "video", contains="No supported media")
        self.fail_cli("--font", self.root / "missing.ttf", contains="Font override")

    def test_invalid_protection_is_rejected_before_output_creation(self):
        image_file(self.input / "sample.png")
        protection = self.root / "protect.json"
        for data in ("null", '{"rectangles": [[0,0,2,1]]}', '{"feather_px_1080": NaN}', '{"unknown": 1}'):
            with self.subTest(data=data):
                protection.write_text(data)
                self.fail_cli("--profiles", "protected", "--protect", protection)

    def test_existing_output_is_not_modified(self):
        image_file(self.input / "sample.png")
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("must remain")
        self.run_cli("--input", self.input, "--output", self.output, expected=2)
        self.assertEqual(sentinel.read_text(), "must remain")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_dangling_output_symlink_is_not_followed(self):
        image_file(self.input / "sample.png")
        missing = self.root / "missing"
        self.output.symlink_to(missing, target_is_directory=True)
        self.run_cli("--input", self.input, "--output", self.output, expected=2)
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(missing.exists())

    def test_output_inside_input_is_rejected(self):
        image_file(self.input / "sample.png")
        destination = self.input / "result"
        self.run_cli("--input", self.input, "--output", destination, expected=2)
        self.assertFalse(destination.exists())

    def test_colliding_converted_video_filenames(self):
        for suffix in (".mov", ".mp4"):
            (self.input / ("clip" + suffix)).write_bytes(b"not decoded: collision preflight runs first")
        self.fail_cli(contains="collide")

    def test_image_report_collision_despite_distinct_media_extensions(self):
        image_file(self.input / "photo.jpg")
        image_file(self.input / "photo.png")
        self.fail_cli(contains="collide")

    def test_generated_media_and_report_parent_collisions(self):
        for parent_suffix in (".mp4", ".json", ".log", ".json.tmp"):
            with self.subTest(parent_suffix=parent_suffix):
                source = self.root / ("case" + parent_suffix)
                source.mkdir()
                (source / "clip.mov").write_bytes(b"synthetic placeholder")
                (source / ("clip" + parent_suffix)).mkdir()
                (source / ("clip" + parent_suffix) / "other.mp4").write_bytes(b"synthetic placeholder")
                self.run_cli("--input", source, "--output", self.output, expected=2)
                self.assertFalse(self.output.exists())

    def test_portable_case_and_unicode_collision_preflight(self):
        cases = [("A.png", "a.png"), ("caf\u00e9.png", "cafe\u0301.png")]
        for names in cases:
            with self.subTest(names=names):
                rows = [dict(relative=name, kind="image") for name in names]
                with self.assertRaisesRegex(ValueError, "collide"):
                    cli.validate_paths(self.output, rows, ["production"])

    def test_image_validation_is_before_directory_creation(self):
        for name, size, mode in (("landscape.png", (192, 108), "RGB"),
                                 ("small.png", (80, 160), "RGB"),
                                 ("alpha.png", (108, 192), "RGBA"),
                                 ("gray.png", (108, 192), "L")):
            path = image_file(self.root / name, size, mode)
            self.run_cli("--input", path, "--output", self.output, expected=2)
            self.assertFalse(self.output.exists())
        path = image_file(self.root / "hard-small.png")
        self.run_cli("--input", path, "--output", self.output, "--profiles", "hard", expected=2)
        self.assertFalse(self.output.exists())

    def test_png_transparency_metadata_and_bad_orientation_are_rejected(self):
        image = Image.new("RGB", (108, 192), (1, 2, 3))
        transparent = self.root / "transparent.png"
        image.save(transparent, transparency=(1, 2, 3))
        self.run_cli("--input", transparent, "--output", self.output, expected=2)
        exif = Image.Exif()
        exif[274] = 9
        invalid = image_file(self.root / "orientation.jpg", exif=exif)
        self.run_cli("--input", invalid, "--output", self.output, expected=2)
        self.assertFalse(self.output.exists())

    def test_identity_png_hierarchy_manifests_and_checksums(self):
        source = image_file(self.input / "nested" / "synthetic.png")
        original_hash = cli.sha(source)
        self.run_cli("--input", self.input, "--output", self.output, "--profiles", "encode-only")
        dest = self.output / "images/encode-only/nested/synthetic.png"
        report = strict_json(self.output / "_reports/image/encode-only/nested/synthetic.json")
        with Image.open(source) as before, Image.open(dest) as after:
            np.testing.assert_array_equal(np.array(before), np.array(after))
        self.assertEqual(cli.sha(source), original_hash)
        self.assertIsNone(report["preencode"]["psnr_rgb_db"])
        self.assertIsNone(report["final_metrics"]["psnr_rgb_db"])
        self.assertTrue(report["source_hash_unchanged"])
        self.assertTrue(report["image"]["full_decode_ok"])
        self.assertTrue(report["final_metrics"]["pixels_identical"])
        self.assertEqual(report["version"], "1.0.0")
        for path in self.output.rglob("*.json"):
            strict_json(path)
        status = strict_json(self.output / "_reports/status.json")
        self.assertEqual(status["status"], "complete")
        self.assertTrue(status["source_hashes_unchanged"])
        checksums = strict_json(self.output / "_reports/checksums.json")["files"]
        for relative, digest in checksums.items():
            self.assertEqual(cli.sha(self.output / relative), digest)
        config = strict_json(self.output / "_reports/config.json")
        config_hash = config.pop("config_sha256")
        self.assertEqual(hashlib.sha256(cli.json_text(config).encode()).hexdigest(), config_hash)
        self.assertTrue(all(path.parts[0] in {"images", "_reports"} for path in
                            (path.relative_to(self.output) for path in self.output.iterdir())))

    def test_jpeg_exif_transpose_icc_retention_and_metadata_removal(self):
        exif = Image.Exif()
        exif[274] = 6
        exif[315] = "Synthetic test artist"
        exif[34853] = {1: "N", 2: (1, 2, 3)}
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        source = image_file(self.input / "rotated.jpg", size=(192, 108), exif=exif, icc=icc)
        self.run_cli("--input", source, "--output", self.output, "--profiles", "encode-only")
        destination = self.output / "images/encode-only/rotated.jpg"
        with Image.open(destination) as image:
            self.assertEqual(image.size, (108, 192))
            self.assertEqual(image.info["icc_profile"], icc)
            self.assertEqual(len(image.getexif()), 0)
            from PIL import JpegImagePlugin
            self.assertEqual(JpegImagePlugin.get_sampling(image), 0)
        report = strict_json(self.output / "_reports/image/encode-only/rotated.json")
        self.assertTrue(report["image"]["orientation_normalized"])
        self.assertTrue(report["image"]["exif_removed"])
        self.assertTrue(report["image"]["icc_preserved"])
        self.assertEqual(report["image"]["encoding"], {"quality": 95, "subsampling": 0})

    def test_top_level_ignores_subdirectories(self):
        image_file(self.input / "top.png")
        image_file(self.input / "nested" / "below.png")
        self.run_cli("--input", self.input, "--output", self.output,
                     "--profiles", "encode-only", "--top-level")
        self.assertTrue((self.output / "images/encode-only/top.png").exists())
        self.assertFalse((self.output / "images/encode-only/nested").exists())
        self.assertEqual(strict_json(self.output / "_reports/status.json")["total"], 1)

    def test_regular_image_profiles_and_protection(self):
        image_file(self.input / "sample.png")
        protection = self.root / "protect.json"
        protection.write_text('{"rectangles": [[0.2, 0.2, 0.8, 0.4]], "feather_px_1080": 0}')
        self.run_cli("--input", self.input, "--output", self.output,
                     "--profiles", "production", "multiscale", "stable", "protected",
                     "--protect", protection, "--jobs", "2", "--no-context")
        for profile in ("production", "multiscale", "stable", "protected"):
            report = strict_json(self.output / f"_reports/image/{profile}/sample.json")
            self.assertEqual(report["profile"], profile)
            self.assertEqual(report["experiment"]["frames"], 1)
            self.assertTrue(report["no_context_requested"])
            self.assertIsNone(report["context_enabled"])
        self.assertEqual(strict_json(self.output / "_reports/status.json")["done"], 4)

    def test_source_change_detection_and_standard_json(self):
        path = image_file(self.input / "source.png")
        row = dict(source=str(path), source_sha256=cli.sha(path))
        path.write_bytes(b"changed synthetic content")
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            cli.verify_source(row)
        result = json.loads(cli.json_text({"nested": [float("inf"), float("nan")],
                                           "target_achieved": True, "target_achieved_frames": 4}))
        self.assertEqual(result, {"nested": [None, None]})

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
    def test_video_profiles_audio_and_hard_context_accounting(self):
        source = self.input / "clip.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=108x192:rate=12:duration=1",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)], check=True)
        before = cli.sha(source)
        self.run_cli("--input", source, "--output", self.output, "--profiles", "encode-only", "hard", "--jobs", "2")
        for profile in ("encode-only", "hard"):
            report = strict_json(self.output / f"_reports/video/{profile}/clip.json")
            self.assertTrue(report["video"]["full_decode_ok"])
            self.assertTrue(report["video"]["frame_count_verified"])
            self.assertEqual(report["video"]["frames"], 12)
            self.assertEqual(report["video"]["fps"], "12")
            self.assertTrue(report["audio"]["payload_identical"])
            self.assertEqual(Path(report["output"]), (self.output / f"videos/{profile}/clip.mp4").resolve())
            self.assertNotIn("target_achieved_frames", report["preencode"])
            self.assertNotIn("rgb_bound_after_geometry", report["preencode"])
        hard = strict_json(self.output / "_reports/video/hard/clip.json")
        self.assertEqual(hard["experiment"]["scheduled_replacements"], [0, 11])
        self.assertEqual(hard["experiment"]["applied_replacements"], [0, 11])
        self.assertTrue(hard["context_enabled"])
        self.assertEqual(cli.sha(source), before)
        no_context = self.root / "no-context"
        self.run_cli("--input", source, "--output", no_context, "--profiles", "hard", "--no-context")
        report = strict_json(no_context / "_reports/video/hard/clip.json")
        self.assertEqual(report["experiment"]["scheduled_replacements"], [])
        self.assertEqual(report["experiment"]["applied_replacements"], [])
        self.assertFalse(report["context_enabled"])
        self.assertFalse(any(path.name.startswith(".media-robustness-") for path in self.output.rglob("*")))


if __name__ == "__main__":
    unittest.main()
