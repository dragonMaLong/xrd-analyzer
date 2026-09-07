from pathlib import Path

import numpy as np
import pytest

from xrd_analyzer.io.project_file import (
    ALGORITHM_VERSION,
    data_sha256,
    file_sha256,
    load_project,
    save_project,
)


def _sample_record(source: Path):
    x = np.linspace(60.0, 62.0, 9)
    y = np.array([1, 2, 5, 12, 20, 12, 5, 2, 1], dtype=float)
    return {
        "sample_id": "sample-1",
        "path": str(source),
        "name": "测试样品",
        "metadata": {"sample_name": "内部样品名", "date": None},
        "status": "complete",
        "compare_visible": True,
        "data_fingerprint": data_sha256(x, y),
        "file_fingerprint": file_sha256(source),
        "parameter_state": {"alpha": 1.0, "kernel": "pearson7"},
        "peak_states": [{"checked": True, "value": 61.0}],
        "analysis_state": {"angle_min": 60.0, "angle_max": 62.0},
        "baseline_state": {},
        "marker_label_state": {},
        "plot_view_state": {},
        "result_signature": "signature",
        "result_is_current": True,
        "x_data": x,
        "y_data": y,
        "results": {
            "D_range": np.array([1.0, 2.0, 3.0]),
            "nested": [{"weights": np.array([0.2, 0.3, 0.5])}],
            "signature": (("alpha", 1.0),),
        },
    }


def test_project_round_trip_preserves_arrays_and_state(tmp_path):
    source = tmp_path / "source.raw"
    source.write_bytes(b"RAW test payload")
    target = tmp_path / "analysis.xrdproj"

    save_progress = []
    load_progress = []
    manifest = save_project(
        target,
        [_sample_record(source)],
        app_version="2.0.4",
        progress_callback=lambda value, stage: save_progress.append((value, stage)),
    )
    loaded = load_project(
        target,
        progress_callback=lambda value, stage: load_progress.append((value, stage)),
    )

    assert manifest["algorithm_version"] == ALGORITHM_VERSION
    assert loaded["app_version"] == "2.0.4"
    assert len(loaded["samples"]) == 1
    sample = loaded["samples"][0]
    np.testing.assert_allclose(sample["x_data"], np.linspace(60.0, 62.0, 9))
    np.testing.assert_allclose(sample["results"]["nested"][0]["weights"], [0.2, 0.3, 0.5])
    assert sample["results"]["signature"] == (("alpha", 1.0),)
    assert save_progress[0][0] == 0 and save_progress[-1][0] == 100
    assert load_progress[0][0] == 0 and load_progress[-1][0] == 100
    assert any("粒径分布" in stage for _value, stage in save_progress)
    assert any("粒径分布" in stage for _value, stage in load_progress)


def test_fingerprints_survive_filename_change(tmp_path):
    first = tmp_path / "before.raw"
    second = tmp_path / "after.raw"
    first.write_bytes(b"same diffraction bytes")
    second.write_bytes(first.read_bytes())

    assert file_sha256(first) == file_sha256(second)
    assert data_sha256([1.0, 2.0], [3.0, 4.0]) == data_sha256([1.0, 2.0], [3.0, 4.0])
    assert data_sha256([1.0, 2.0], [3.0, 4.0]) != data_sha256([1.0, 2.0], [3.0, 4.1])


def test_project_rejects_multiple_samples(tmp_path):
    source = tmp_path / "source.raw"
    source.write_bytes(b"RAW test payload")
    records = [_sample_record(source), {**_sample_record(source), "sample_id": "sample-2"}]

    with pytest.raises(ValueError, match="包含且仅包含 1 个样品"):
        save_project(tmp_path / "multi.xrdproj", records)
