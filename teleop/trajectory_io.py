"""Trajectory recording and loading for keyboard teleoperation.

The current STM32 interface is command-only, so recorded joint angles are
COMMAND angles sent (or, in dry-run, intended to be sent) to the controller.
They are not measured encoder feedback.

CSV columns:
    frame, t_s, x_mm, y_mm, z_mm,
    j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, j6_deg

A JSON file with the same stem stores session metadata.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np


CSV_FIELDS = [
    "frame",
    "t_s",
    "x_mm",
    "y_mm",
    "z_mm",
    "j1_deg",
    "j2_deg",
    "j3_deg",
    "j4_deg",
    "j5_deg",
    "j6_deg",
]

DEFAULT_RECORDING_DIR = Path(__file__).resolve().parent / "recordings"


def default_recording_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RECORDING_DIR / f"trajectory_{stamp}.csv"


@dataclass(frozen=True)
class TrajectoryData:
    t_s: np.ndarray
    xyz_mm: np.ndarray
    joints_deg: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.t_s.shape[0])

    @property
    def duration_s(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return float(self.t_s[-1] - self.t_s[0])


class TrajectoryRecorder:
    """Incrementally persist commanded frames during teleoperation."""

    def __init__(
        self,
        csv_path: str | Path | None = None,
        *,
        metadata: dict | None = None,
        flush_every: int = 25,
    ):
        self.csv_path = (
            Path(csv_path).expanduser().resolve()
            if csv_path is not None
            else default_recording_path().resolve()
        )
        if self.csv_path.suffix.lower() != ".csv":
            self.csv_path = self.csv_path.with_suffix(".csv")

        self.json_path = self.csv_path.with_suffix(".json")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = self.csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=CSV_FIELDS,
        )
        self._writer.writeheader()
        self._file.flush()

        self._start_perf = time.perf_counter()
        self._frame_count = 0
        self._flush_every = max(1, int(flush_every))
        self._closed = False

        self.metadata = dict(metadata or {})
        self.metadata.update(
            {
                "format_version": 1,
                "angle_semantics": "commanded_target_deg",
                "feedback_available": False,
                "csv_file": self.csv_path.name,
                "started_local": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
        )
        self._write_metadata(final=False)

    def _write_metadata(self, *, final: bool) -> None:
        payload = dict(self.metadata)
        payload["frame_count"] = self._frame_count
        payload["finalized"] = bool(final)

        if final:
            payload["finished_local"] = datetime.now().isoformat(
                timespec="seconds"
            )

        temp_path = self.json_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_path.replace(self.json_path)

    def record(
        self,
        xyz_mm,
        joints_deg,
        *,
        timestamp_s: float | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("trajectory recorder is already closed")

        xyz = np.asarray(xyz_mm, dtype=np.float64)
        joints = np.asarray(joints_deg, dtype=np.float64)

        if xyz.shape != (3,):
            raise ValueError("xyz_mm must contain 3 values")
        if joints.shape != (6,):
            raise ValueError("joints_deg must contain 6 values")
        if not np.all(np.isfinite(xyz)):
            raise ValueError("xyz_mm contains non-finite values")
        if not np.all(np.isfinite(joints)):
            raise ValueError("joints_deg contains non-finite values")

        if timestamp_s is None:
            timestamp_s = time.perf_counter() - self._start_perf

        t_s = float(timestamp_s)
        if not np.isfinite(t_s) or t_s < 0.0:
            raise ValueError("timestamp must be finite and non-negative")

        row = {
            "frame": self._frame_count,
            "t_s": f"{t_s:.9f}",
            "x_mm": f"{xyz[0]:.9f}",
            "y_mm": f"{xyz[1]:.9f}",
            "z_mm": f"{xyz[2]:.9f}",
        }
        for index in range(6):
            row[f"j{index + 1}_deg"] = f"{joints[index]:.9f}"

        self._writer.writerow(row)
        self._frame_count += 1

        if self._frame_count % self._flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        if self._closed:
            return

        self._file.flush()
        self._file.close()
        self._closed = True
        self._write_metadata(final=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def load_trajectory(csv_path: str | Path) -> TrajectoryData:
    path = Path(csv_path).expanduser().resolve()

    t_values = []
    xyz_values = []
    joint_values = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(
                "trajectory CSV header does not match format version 1"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                t_s = float(row["t_s"])
                xyz = [
                    float(row["x_mm"]),
                    float(row["y_mm"]),
                    float(row["z_mm"]),
                ]
                joints = [
                    float(row[f"j{i}_deg"])
                    for i in range(1, 7)
                ]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid numeric data at CSV row {row_number}"
                ) from error

            values = np.asarray(
                [t_s, *xyz, *joints],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"non-finite data at CSV row {row_number}"
                )

            t_values.append(t_s)
            xyz_values.append(xyz)
            joint_values.append(joints)

    if not t_values:
        raise ValueError("trajectory contains no frames")

    t_array = np.asarray(t_values, dtype=np.float64)
    xyz_array = np.asarray(xyz_values, dtype=np.float64)
    joints_array = np.asarray(joint_values, dtype=np.float64)

    if np.any(np.diff(t_array) < 0.0):
        raise ValueError("trajectory timestamps are not monotonic")

    # Normalize so replay always starts from t=0 even if the first recorded
    # sample was written a few milliseconds after recorder creation.
    t_array = t_array - t_array[0]

    return TrajectoryData(
        t_s=t_array,
        xyz_mm=xyz_array,
        joints_deg=joints_array,
    )


def load_metadata(csv_path: str | Path) -> dict:
    path = Path(csv_path).expanduser().resolve()
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
