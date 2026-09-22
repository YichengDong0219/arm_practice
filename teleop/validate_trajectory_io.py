"""Self-test for trajectory record/load/replay data integrity."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop.trajectory_io import (
    TrajectoryRecorder,
    load_metadata,
    load_trajectory,
)


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.csv"

        with TrajectoryRecorder(
            path,
            metadata={"test": True},
            flush_every=2,
        ) as recorder:
            for frame in range(5):
                recorder.record(
                    [frame, frame + 1, frame + 2],
                    [
                        frame + 0.1,
                        frame + 0.2,
                        frame + 0.3,
                        frame + 0.4,
                        frame + 0.5,
                        frame + 0.6,
                    ],
                    timestamp_s=frame * 0.02,
                )

        data = load_trajectory(path)
        metadata = load_metadata(path)

        assert data.frame_count == 5
        assert np.allclose(
            data.t_s,
            [0.00, 0.02, 0.04, 0.06, 0.08],
        )
        assert np.allclose(
            data.xyz_mm[-1],
            [4, 5, 6],
        )
        assert np.allclose(
            data.joints_deg[-1],
            [4.1, 4.2, 4.3, 4.4, 4.5, 4.6],
        )
        assert metadata["frame_count"] == 5
        assert metadata["finalized"] is True
        assert (
            metadata["angle_semantics"]
            == "commanded_target_deg"
        )

    print("PASS: trajectory record/load integrity")


if __name__ == "__main__":
    main()
