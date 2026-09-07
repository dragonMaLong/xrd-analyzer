"""Versioned, portable single-sample project storage for XRD Analyzer."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


# Schema 2 replaces persisted two-dimensional peak basis matrices with compact
# one-dimensional display-curve snapshots. The loader remains backward
# compatible with schema-1 projects.
PROJECT_SCHEMA_VERSION = 2
ALGORITHM_VERSION = "area-normalized-nnls-rfit-valley-baseline-v3"
PROJECT_EXTENSION = ".xrdproj"
ProgressCallback = Callable[[int, str], None]


class ProjectFormatError(ValueError):
    """Raised when a project is damaged or uses an unsupported schema."""


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash the source bytes without loading a potentially large RAW file at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def data_sha256(x_data, y_data) -> str:
    """Return a filename-independent fingerprint for a parsed diffraction scan."""
    digest = hashlib.sha256(b"XRDAnalyzer:data-fingerprint:v1\0")
    for values in (x_data, y_data):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8").reshape(-1))
        digest.update(int(array.size).to_bytes(8, "little", signed=False))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_state_sha256(value) -> str:
    """Hash JSON-compatible calculation state in a deterministic form."""
    arrays: dict[str, np.ndarray] = {}
    packed = _pack_value(value, arrays, "state")
    array_descriptors = []
    for key, array in arrays.items():
        contiguous = np.ascontiguousarray(array)
        array_descriptors.append(
            {
                "key": key,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            }
        )
    payload = json.dumps(
        {"value": packed, "arrays": array_descriptors},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_project(
    path: str | Path,
    samples: list[dict],
    *,
    active_sample_index: int = 0,
    app_version: str = "",
    project_uuid: str | None = None,
    kind: str = "project",
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Atomically save sample records to one ZIP-based project container."""
    if str(kind) == "project" and len(samples) != 1:
        raise ValueError("工程文件结构无效：应包含且仅包含 1 个样品")
    _report_progress(progress_callback, 0, "准备工程数据")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    project_uuid = str(project_uuid or uuid.uuid4())
    saved_at = datetime.now(timezone.utc).isoformat()
    created_at = saved_at
    if target.is_file():
        try:
            with zipfile.ZipFile(target, "r") as existing_archive:
                existing_manifest = _read_json_member(existing_archive, "manifest.json")
                created_at = str(existing_manifest.get("created_at") or saved_at)
        except Exception:
            created_at = saved_at
    manifest = {
        "format": "XRDAnalyzerProject",
        "schema_version": PROJECT_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "app_version": str(app_version),
        "project_uuid": project_uuid,
        "kind": str(kind),
        "created_at": created_at,
        "saved_at": saved_at,
        "active_sample_index": int(active_sample_index),
        "samples": [],
    }

    descriptor = None
    temp_path = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        os.close(descriptor)
        descriptor = None
        temp_path = Path(temp_name)

        # NumPy payloads are already compressed by np.savez_compressed. Storing
        # those NPZ members directly avoids an expensive second DEFLATE pass
        # that barely changes file size, especially for large fit results.
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED) as archive:
            used_ids: set[str] = set()
            for index, source_record in enumerate(samples):
                _report_progress(progress_callback, 8, "整理样品参数")
                record = dict(source_record)
                sample_id = str(record.get("sample_id") or uuid.uuid4())
                if sample_id in used_ids:
                    sample_id = str(uuid.uuid4())
                used_ids.add(sample_id)
                record["sample_id"] = sample_id
                folder = f"samples/{sample_id}"

                x_data = np.asarray(record.pop("x_data"), dtype=float)
                y_data = np.asarray(record.pop("y_data"), dtype=float)
                results = record.pop("results", {}) or {}

                manifest["samples"].append(
                    {
                        "sample_id": sample_id,
                        "folder": folder,
                        "name": str(record.get("name") or ""),
                        "data_sha256": str(record.get("data_fingerprint") or data_sha256(x_data, y_data)),
                        "has_results": bool(results),
                    }
                )

                state_arrays: dict[str, np.ndarray] = {}
                packed_state = _pack_value(record, state_arrays, "state")
                archive.writestr(
                    f"{folder}/state.json",
                    json.dumps(packed_state, ensure_ascii=False, separators=(",", ":")),
                )
                if state_arrays:
                    _report_progress(progress_callback, 20, "压缩参数与标记数据")
                    archive.writestr(f"{folder}/state_arrays.npz", _npz_bytes(state_arrays))

                _report_progress(progress_callback, 35, "压缩原始 XRD 数据")
                archive.writestr(
                    f"{folder}/data.npz",
                    _npz_bytes({"x_data": x_data, "y_data": y_data}),
                )

                if results:
                    _report_progress(progress_callback, 52, "整理拟合结果")
                    result_arrays: dict[str, np.ndarray] = {}
                    packed_results = _pack_value(results, result_arrays, "result")
                    archive.writestr(
                        f"{folder}/result.json",
                        json.dumps(packed_results, ensure_ascii=False, separators=(",", ":")),
                    )
                    _report_progress(progress_callback, 62, "压缩拟合曲线与粒径分布")
                    archive.writestr(f"{folder}/result.npz", _npz_bytes(result_arrays))
                    _report_progress(progress_callback, 90, "写入工程索引")

            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )

        _report_progress(progress_callback, 96, "完成工程文件写入")
        os.replace(temp_path, target)
        temp_path = None
        _report_progress(progress_callback, 100, "保存完成")
        return manifest
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def materialize_project_snapshot(
    snapshot_path: str | Path,
    target_path: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Atomically save a prepared project without recompressing its NumPy arrays.

    The snapshot already contains the exact single-sample state for one project
    revision.  Materializing it only streams the prepared ZIP members to the
    destination and refreshes the save timestamp.  This keeps normal project
    semantics while moving expensive ``np.savez_compressed`` work off the
    user's explicit Save action.
    """
    source = Path(snapshot_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _report_progress(progress_callback, 0, "检查预生成工程快照")

    saved_at = datetime.now(timezone.utc).isoformat()
    descriptor = None
    temp_path = None
    try:
        with zipfile.ZipFile(source, "r") as source_archive:
            manifest = _read_json_member(source_archive, "manifest.json")
            if manifest.get("format") != "XRDAnalyzerProject":
                raise ProjectFormatError("预生成工程快照无效")
            if str(manifest.get("kind") or "project") != "project":
                raise ProjectFormatError("预生成工程快照类型无效")
            if len(manifest.get("samples", [])) != 1:
                raise ProjectFormatError("预生成工程快照应包含且仅包含 1 个样品")

            # Match save_project(): overwriting an existing valid project keeps
            # its original creation time, while saved_at reflects this Save.
            created_at = str(manifest.get("created_at") or saved_at)
            if target.is_file() and source.resolve() != target.resolve():
                try:
                    with zipfile.ZipFile(target, "r") as existing_archive:
                        existing_manifest = _read_json_member(existing_archive, "manifest.json")
                        created_at = str(existing_manifest.get("created_at") or created_at)
                except Exception:
                    pass
            manifest["created_at"] = created_at
            manifest["saved_at"] = saved_at

            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
            )
            os.close(descriptor)
            descriptor = None
            temp_path = Path(temp_name)

            members = [info for info in source_archive.infolist() if info.filename != "manifest.json"]
            total_size = max(1, sum(max(0, int(info.file_size)) for info in members))
            copied_size = 0
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED) as target_archive:
                for info in members:
                    with source_archive.open(info, "r") as source_member:
                        with target_archive.open(info, "w", force_zip64=True) as target_member:
                            shutil.copyfileobj(source_member, target_member, length=1024 * 1024)
                    copied_size += max(0, int(info.file_size))
                    progress = 10 + int(80 * copied_size / total_size)
                    _report_progress(progress_callback, progress, "写入预压缩工程数据")
                target_archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                )

        _report_progress(progress_callback, 96, "完成工程文件写入")
        os.replace(temp_path, target)
        temp_path = None
        _report_progress(progress_callback, 100, "保存完成")
        return manifest
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, ProjectFormatError):
            raise
        raise ProjectFormatError(f"无法使用预生成工程快照：{exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def load_project(
    path: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Load and validate a single-sample project without extracting members."""
    source = Path(path)
    try:
        _report_progress(progress_callback, 0, "打开工程文件")
        with zipfile.ZipFile(source, "r") as archive:
            _report_progress(progress_callback, 6, "读取工程索引")
            manifest = _read_json_member(archive, "manifest.json")
            if manifest.get("format") != "XRDAnalyzerProject":
                raise ProjectFormatError("不是有效的 XRD Analyzer 工程文件")
            schema_version = int(manifest.get("schema_version", 0))
            if schema_version > PROJECT_SCHEMA_VERSION:
                raise ProjectFormatError(
                    f"工程格式版本 {schema_version} 高于当前支持的版本 {PROJECT_SCHEMA_VERSION}"
                )
            if schema_version < 1:
                raise ProjectFormatError("工程格式版本无效")

            descriptors = list(manifest.get("samples", []))
            if str(manifest.get("kind") or "project") == "project" and len(descriptors) != 1:
                raise ProjectFormatError(
                    "工程文件结构无效：应包含且仅包含 1 个样品"
                )

            records = []
            for descriptor in descriptors:
                sample_id = str(descriptor.get("sample_id") or "")
                folder = str(descriptor.get("folder") or "")
                expected_folder = f"samples/{sample_id}"
                if not sample_id or folder != expected_folder:
                    raise ProjectFormatError("工程中的样品目录无效")

                _report_progress(progress_callback, 18, "读取样品参数与标记")
                state_arrays = _read_npz_member(archive, f"{folder}/state_arrays.npz", optional=True)
                packed_state = _read_json_member(archive, f"{folder}/state.json")
                record = _unpack_value(packed_state, state_arrays)
                _report_progress(progress_callback, 35, "读取原始 XRD 数据")
                data = _read_npz_member(archive, f"{folder}/data.npz")
                if "x_data" not in data or "y_data" not in data:
                    raise ProjectFormatError(f"样品 {sample_id} 缺少原始 X/Y 数据")
                actual_fingerprint = data_sha256(data["x_data"], data["y_data"])
                expected_fingerprint = str(descriptor.get("data_sha256") or "")
                if expected_fingerprint and actual_fingerprint != expected_fingerprint:
                    raise ProjectFormatError(f"样品 {sample_id} 的数据指纹校验失败")
                record["x_data"] = data["x_data"]
                record["y_data"] = data["y_data"]
                record["data_fingerprint"] = actual_fingerprint

                result_name = f"{folder}/result.json"
                if result_name in archive.namelist():
                    _report_progress(progress_callback, 55, "读取拟合结果索引")
                    packed_results = _read_json_member(archive, result_name)
                    _report_progress(progress_callback, 65, "读取拟合曲线与粒径分布")
                    result_arrays = _read_npz_member(archive, f"{folder}/result.npz")
                    record["results"] = _unpack_value(packed_results, result_arrays)
                else:
                    record["results"] = {}
                _report_progress(progress_callback, 94, "校验工程数据")
                records.append(record)
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, ProjectFormatError):
            raise
        raise ProjectFormatError(f"无法读取工程文件：{exc}") from exc

    _report_progress(progress_callback, 100, "读取完成")
    return {**manifest, "samples": records}


def _report_progress(callback: ProgressCallback | None, value: int, stage: str) -> None:
    if callback is None:
        return
    try:
        callback(max(0, min(100, int(value))), str(stage))
    except Exception:
        # A display callback must never make a valid save/load operation fail.
        pass


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _read_json_member(archive: zipfile.ZipFile, name: str):
    try:
        raw = archive.read(name)
    except KeyError as exc:
        raise ProjectFormatError(f"工程缺少 {name}") from exc
    return json.loads(raw.decode("utf-8"))


def _read_npz_member(archive: zipfile.ZipFile, name: str, *, optional: bool = False) -> dict[str, np.ndarray]:
    if name not in archive.namelist():
        if optional:
            return {}
        raise ProjectFormatError(f"工程缺少 {name}")
    try:
        with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as loaded:
            return {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    except (OSError, ValueError) as exc:
        raise ProjectFormatError(f"无法读取数组数据 {name}：{exc}") from exc


def _pack_value(value, arrays: dict[str, np.ndarray], prefix: str):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("工程文件不支持 object 类型 NumPy 数组")
        for existing_key, existing_array in arrays.items():
            if existing_array is value:
                return {"__xrd_type__": "ndarray", "key": existing_key}
        key = f"a{len(arrays):06d}"
        arrays[key] = np.asarray(value)
        return {"__xrd_type__": "ndarray", "key": key}
    if isinstance(value, np.generic):
        return _pack_value(value.item(), arrays, prefix)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__xrd_type__": "float", "value": "nan"}
        if math.isinf(value):
            return {"__xrd_type__": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            "__xrd_type__": "dict",
            "items": [
                [_pack_value(key, arrays, prefix), _pack_value(item, arrays, prefix)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, tuple):
        return {"__xrd_type__": "tuple", "items": [_pack_value(item, arrays, prefix) for item in value]}
    if isinstance(value, set):
        return {"__xrd_type__": "set", "items": [_pack_value(item, arrays, prefix) for item in value]}
    if isinstance(value, list):
        return [_pack_value(item, arrays, prefix) for item in value]
    raise TypeError(f"工程文件不支持的数据类型：{type(value).__name__}")


def _unpack_value(value, arrays: dict[str, np.ndarray]):
    if isinstance(value, list):
        return [_unpack_value(item, arrays) for item in value]
    if not isinstance(value, dict) or "__xrd_type__" not in value:
        return value
    kind = value.get("__xrd_type__")
    if kind == "ndarray":
        key = str(value.get("key") or "")
        if key not in arrays:
            raise ProjectFormatError(f"工程缺少数组 {key}")
        return arrays[key]
    if kind == "float":
        return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}[value["value"]]
    if kind == "dict":
        return {
            _unpack_value(key, arrays): _unpack_value(item, arrays)
            for key, item in value.get("items", [])
        }
    if kind == "tuple":
        return tuple(_unpack_value(item, arrays) for item in value.get("items", []))
    if kind == "set":
        return set(_unpack_value(item, arrays) for item in value.get("items", []))
    raise ProjectFormatError(f"未知的工程数据类型：{kind}")
