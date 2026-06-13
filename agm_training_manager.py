"""
Background training pipeline manager — fetch, label, prepare, train.

Progress written to agm_deer_ml/runs/pipeline_status.json for the web UI.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ML_ROOT = ROOT / "agm_deer_ml"
STATUS_FILE = ML_ROOT / "runs" / "pipeline_status.json"
PYTHON = sys.executable


@dataclass
class PipelineStatus:
    running: bool = False
    phase: str = "idle"
    progress: int = 0
    message: str = ""
    last_log: str = ""
    started_at: str = ""
    finished_at: str = ""
    visual_images: int = 0
    thermal_images: int = 0
    train_images: int = 0
    val_images: int = 0
    error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "phase": self.phase,
            "progress": self.progress,
            "message": self.message,
            "last_log": self.last_log,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "visual_images": self.visual_images,
            "thermal_images": self.thermal_images,
            "train_images": self.train_images,
            "val_images": self.val_images,
            "error": self.error,
        }

    def save(self):
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            STATUS_FILE.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k) and k != "lock":
                    setattr(self, k, v)
        self.save()


_status = PipelineStatus()
_thread: threading.Thread | None = None


def get_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _status.to_dict()


def is_running() -> bool:
    return get_status().get("running", False)


def _run_cmd(cmd: list[str], phase: str, progress: int, message: str) -> None:
    _status.update(phase=phase, progress=progress, message=message, last_log=message)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            lines.append(line)
            if len(lines) > 8:
                lines.pop(0)
            _status.update(last_log=line)
    code = proc.wait()
    if code != 0:
        tail = "\n".join(lines[-15:])
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}\n{tail}")


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _pipeline_worker(epochs: int, batch: int, max_visual: int, max_thermal: int):
    try:
        _status.update(
            running=True,
            phase="fetch",
            progress=5,
            message="Downloading roadside deer images (visual + thermal)…",
            started_at=datetime.now().isoformat(timespec="seconds"),
            error="",
            finished_at="",
        )

        _run_cmd(
            [
                PYTHON,
                str(ML_ROOT / "scripts" / "fetch_roadside_deer.py"),
                "--max-visual", str(max_visual),
                "--max-thermal", str(max_thermal),
            ],
            "fetch",
            15,
            "Fetching images from iNaturalist & Wikimedia…",
        )

        visual_n = _count_images(ML_ROOT / "dataset_internet" / "visual")
        thermal_n = _count_images(ML_ROOT / "dataset_internet" / "thermal")
        _status.update(visual_images=visual_n, thermal_images=thermal_n)

        _run_cmd(
            [PYTHON, str(ML_ROOT / "scripts" / "auto_label_deer.py")],
            "label",
            35,
            "Auto-labeling deer with YOLO-World…",
        )

        _run_cmd(
            [
                PYTHON,
                str(ML_ROOT / "scripts" / "prepare_dataset.py"),
                "--clean",
                "--merge-internet",
            ],
            "prepare",
            50,
            "Building train/val dataset (DAID-T + internet)…",
        )

        train_n = _count_images(ML_ROOT / "dataset" / "images" / "train")
        val_n = _count_images(ML_ROOT / "dataset" / "images" / "val")
        _status.update(train_images=train_n, val_images=val_n, progress=55)

        _run_cmd(
            [
                PYTHON,
                str(ML_ROOT / "scripts" / "train_deer_yolo.py"),
                "--epochs", str(epochs),
                "--batch", str(batch),
            ],
            "train",
            60,
            f"Training YOLOv8n ({epochs} epochs, {train_n} images)…",
        )

        _status.update(
            running=False,
            phase="done",
            progress=100,
            message="Training complete. Weights: agm_deer_ml/models/deer_thermal_best.pt",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            last_log="Pipeline finished successfully.",
        )
    except Exception as e:
        _status.update(
            running=False,
            phase="error",
            progress=_status.progress,
            message="Pipeline failed",
            error=str(e),
            finished_at=datetime.now().isoformat(timespec="seconds"),
            last_log=str(e),
        )


def start_pipeline(
    epochs: int = 30,
    batch: int = 4,
    max_visual: int = 80,
    max_thermal: int = 40,
) -> dict:
    global _thread
    if is_running():
        return {"ok": False, "error": "Pipeline already running", "status": get_status()}

    _thread = threading.Thread(
        target=_pipeline_worker,
        args=(epochs, batch, max_visual, max_thermal),
        name="training-pipeline",
        daemon=True,
    )
    _thread.start()
    time.sleep(0.3)
    return {"ok": True, "status": get_status()}
