from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from .version import __version__


DEFAULT_UPDATE_REPOSITORY = "dragonMaLong/xrd-analyzer"
DEFAULT_GITEE_MANIFEST_URL = (
    "https://gitee.com/dragonMalong/xrd-analyzer/raw/main/updates/latest.json"
)
DEFAULT_GITHUB_MANIFEST_URL = (
    "https://raw.githubusercontent.com/dragonMaLong/xrd-analyzer/main/updates/latest.json"
)
DEFAULT_UPDATE_MANIFEST_URLS = (
    DEFAULT_GITEE_MANIFEST_URL,
    DEFAULT_GITHUB_MANIFEST_URL,
)
GITHUB_LATEST_RELEASE_URL = "https://api.github.com/repos/{repo}/releases/latest"


@dataclass(frozen=True)
class UpdateInfo:
    current_version: str
    latest_version: str
    update_available: bool
    release_url: str
    download_url: str
    asset_name: str
    release_name: str
    release_notes: str
    published_at: str
    source_name: str = ""
    source_url: str = ""
    sha256: str = ""


class UpdateCheckError(RuntimeError):
    """Raised when the update endpoint cannot be reached or parsed."""


def check_for_update(
    current_version: str = __version__,
    *,
    repository: str = DEFAULT_UPDATE_REPOSITORY,
    manifest_urls: Iterable[str] = DEFAULT_UPDATE_MANIFEST_URLS,
    timeout: float = 8.0,
) -> UpdateInfo:
    errors: list[str] = []
    candidates: list[UpdateInfo] = []

    for url in manifest_urls:
        url = str(url or "").strip()
        if not url:
            continue
        try:
            payload = _fetch_json(url, timeout=timeout, api=False)
            candidates.append(_info_from_manifest(payload, current_version, url))
        except UpdateCheckError as exc:
            errors.append(f"{_source_name_from_url(url)}: {exc}")

    if candidates:
        best = _newest_info(candidates)
        if best.update_available:
            return best

    try:
        release_info = _info_from_github_release(
            _fetch_latest_release(repository, timeout=timeout),
            current_version,
            repository,
        )
    except UpdateCheckError as exc:
        errors.append(f"GitHub Releases: {exc}")
        if candidates:
            return _newest_info(candidates)
        raise UpdateCheckError("; ".join(errors)) from exc

    candidates.append(release_info)
    return _newest_info(candidates)


def compare_versions(left: str, right: str) -> int:
    left_parts = _version_parts(left)
    right_parts = _version_parts(right)
    max_len = max(len(left_parts), len(right_parts), 3)
    left_parts.extend([0] * (max_len - len(left_parts)))
    right_parts.extend([0] * (max_len - len(right_parts)))
    if left_parts > right_parts:
        return 1
    if left_parts < right_parts:
        return -1
    return 0


def _newest_info(candidates: Iterable[UpdateInfo]) -> UpdateInfo:
    return max(candidates, key=lambda info: _version_parts(info.latest_version))


def _info_from_manifest(payload: dict[str, Any], current_version: str, source_url: str) -> UpdateInfo:
    version = str(
        payload.get("version")
        or payload.get("latest_version")
        or payload.get("tag_name")
        or ""
    ).strip()
    if not version:
        raise UpdateCheckError("更新清单缺少 version 字段。")

    latest_version = _clean_version(version)
    detected_source = _source_name_from_url(source_url)
    source_prefix = "gitee" if detected_source == "Gitee" else "github" if detected_source == "GitHub" else ""
    release_url = _first_manifest_value(
        payload,
        _manifest_keys(
            source_prefix,
            "release_url",
            legacy_keys=("release_url", "html_url", "gitee_url", "github_url"),
        ),
    )
    download_url = _first_manifest_value(
        payload,
        _manifest_keys(
            source_prefix,
            "download_url",
            legacy_keys=("download_url", "browser_download_url"),
        ),
    )
    asset_name = str(payload.get("asset_name") or "").strip()
    if not asset_name and download_url:
        asset_name = download_url.rstrip("/").rsplit("/", 1)[-1]

    return UpdateInfo(
        current_version=_clean_version(current_version),
        latest_version=latest_version,
        update_available=compare_versions(latest_version, current_version) > 0,
        release_url=release_url,
        download_url=download_url,
        asset_name=asset_name,
        release_name=str(payload.get("release_name") or payload.get("name") or f"v{latest_version}").strip(),
        release_notes=str(payload.get("release_notes") or payload.get("notes") or payload.get("body") or "").strip(),
        published_at=str(payload.get("published_at") or "").strip(),
        source_name=str(payload.get(f"{source_prefix}_source_name") or detected_source or payload.get("source_name") or "").strip(),
        source_url=source_url,
        sha256=str(payload.get("sha256") or payload.get("checksum") or "").strip(),
    )


def _manifest_keys(source_prefix: str, key: str, *, legacy_keys: tuple[str, ...]) -> tuple[str, ...]:
    if not source_prefix:
        return legacy_keys
    return (f"{source_prefix}_{key}", *legacy_keys)


def _first_manifest_value(payload: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _info_from_github_release(payload: dict[str, Any], current_version: str, repository: str) -> UpdateInfo:
    tag_name = str(payload.get("tag_name") or "").strip()
    if not tag_name:
        raise UpdateCheckError("GitHub Release 没有版本标签。")

    latest_version = _clean_version(tag_name)
    release_url = str(payload.get("html_url") or "").strip()
    assets = payload.get("assets") if isinstance(payload.get("assets"), list) else []
    asset = _choose_download_asset(assets)

    return UpdateInfo(
        current_version=_clean_version(current_version),
        latest_version=latest_version,
        update_available=compare_versions(latest_version, current_version) > 0,
        release_url=release_url,
        download_url=str(asset.get("browser_download_url") or "").strip() if asset else "",
        asset_name=str(asset.get("name") or "").strip() if asset else "",
        release_name=str(payload.get("name") or tag_name).strip(),
        release_notes=str(payload.get("body") or "").strip(),
        published_at=str(payload.get("published_at") or "").strip(),
        source_name="GitHub Releases",
        source_url=GITHUB_LATEST_RELEASE_URL.format(repo=repository.strip()),
        sha256="",
    )


def _fetch_latest_release(repository: str, *, timeout: float) -> dict[str, Any]:
    url = GITHUB_LATEST_RELEASE_URL.format(repo=repository.strip())
    return _fetch_json(url, timeout=timeout, api=True)


def _fetch_json(url: str, *, timeout: float, api: bool) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json" if api else "application/json, text/plain, */*",
        "User-Agent": f"XRD-Analyzer/{__version__}",
    }
    if api:
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateCheckError("没有找到更新信息。") from exc
        raise UpdateCheckError(f"服务器返回 HTTP {exc.code}。") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise UpdateCheckError(f"无法连接：{reason}") from exc
    except TimeoutError as exc:
        raise UpdateCheckError("连接超时。") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateCheckError(f"无法读取更新信息：{exc}") from exc
    if not isinstance(payload, dict):
        raise UpdateCheckError("更新信息格式不是 JSON 对象。")
    return payload


def _choose_download_asset(assets: list[Any]) -> dict[str, Any] | None:
    candidates = [asset for asset in assets if isinstance(asset, dict)]
    if not candidates:
        return None

    preferred_suffixes = (".exe", ".msi", ".zip", ".7z")
    for suffix in preferred_suffixes:
        for asset in candidates:
            name = str(asset.get("name") or "").lower()
            if name.endswith(suffix) and asset.get("browser_download_url"):
                return asset

    for asset in candidates:
        if asset.get("browser_download_url"):
            return asset
    return None


def _source_name_from_url(url: str) -> str:
    text = str(url).lower()
    if "gitee.com" in text:
        return "Gitee"
    if "github.com" in text or "githubusercontent.com" in text:
        return "GitHub"
    return "更新源"


def _clean_version(version: str) -> str:
    text = str(version or "").strip()
    if text.lower().startswith("version "):
        text = text[8:].strip()
    if text[:1].lower() == "v":
        text = text[1:]
    return text


def _version_parts(version: str) -> list[int]:
    text = _clean_version(version)
    match = re.match(r"(\d+(?:\.\d+)*)", text)
    if not match:
        return [0]
    return [int(part) for part in match.group(1).split(".")]
