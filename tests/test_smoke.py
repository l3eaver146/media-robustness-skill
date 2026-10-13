"""Exercise every shipped profile against synthetic image and audio/video inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw


CLI = Path(__file__).resolve().parents[1] / "skills/media-robustness/scripts/process_media.py"
PROFILES = ("production", "multiscale", "protected", "stable", "encode-only", "hard")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and ffprobe required")
class AllProfilesSmokeTest(unittest.TestCase):
    def test_mixed_batch_decodes_preserves_sources_and_audio(self):
        with tempfile.TemporaryDirectory(prefix="media-skill-smoke-") as temporary:
            root = Path(temporary)
            inputs = root / "inputs" / "nested"
            inputs.mkdir(parents=True)
            y, x = np.mgrid[:1920, :1080]
            pixels = np.stack(((x // 5 + y // 9) % 256,
                               (x // 8 + 80) % 256,
                               (y // 12 + 40) % 256), axis=-1).astype(np.uint8)
            still = Image.fromarray(pixels)
            draw = ImageDraw.Draw(still)
            draw.rectangle((200, 400, 870, 800), fill=(28, 35, 60))
            draw.text((280, 540), "SYNTHETIC FIXTURE", fill="white", font_size=48)
            still_path = inputs / "still.png"
            still.save(still_path)
            clip_path = inputs / "clip.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=30:duration=1.1",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.1",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(clip_path),
            ], check=True, capture_output=True, timeout=30)
            originals = {path: digest(path) for path in (still_path, clip_path)}
            output = root / "result"
            completed = subprocess.run([
                sys.executable, str(CLI), "--input", str(root / "inputs"),
                "--output", str(output), "--profiles", *PROFILES, "--jobs", "2",
            ], capture_output=True, text=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            status = json.loads((output / "_reports/status.json").read_text())
            self.assertEqual((status["status"], status["done"], status["failed"]), ("complete", 12, 0))
            for source, before in originals.items():
                self.assertEqual(digest(source), before)
            for profile in PROFILES:
                with Image.open(output / "images" / profile / "nested/still.png") as image:
                    image.load()
                    self.assertEqual(image.size, (1080, 1920))
                    if profile == "encode-only":
                        np.testing.assert_array_equal(np.array(image), np.array(still))
                report = json.loads((output / "_reports/video" / profile / "nested/clip.json").read_text())
                self.assertTrue(report["video"]["full_decode_ok"])
                self.assertTrue(report["video"]["frame_count_verified"])
                self.assertTrue(report["audio"]["payload_identical"])
                self.assertTrue(report["source_hash_unchanged"])
                self.assertFalse(report["semantic_or_ocr_or_asr_success_verified"])
                if profile == "hard":
                    self.assertEqual(report["experiment"]["applied_replacements"],
                                     report["experiment"]["scheduled_replacements"])
            checksums = json.loads((output / "_reports/checksums.json").read_text())["files"]
            for relative, expected in checksums.items():
                self.assertEqual(digest(output / relative), expected)


if __name__ == "__main__":
    unittest.main()
