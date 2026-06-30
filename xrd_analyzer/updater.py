from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .update_checker import UpdateInfo


ProgressCallback = Callable[[int, int], None]


class UpdateDownloadError(RuntimeError):
    """Raised when an update package cannot be downloaded or verified."""


def download_update(info: UpdateInfo, *, progress_callback: ProgressCallback | None = None) -> Path:
    url = (info.download_url or "").strip()
    part_urls = tuple(part.strip() for part in getattr(info, "download_parts", ()) if str(part).strip())
    if not url and not part_urls:
        raise UpdateDownloadError("更新信息中没有可下载的安装包链接。")

    download_dir = _download_dir()
    download_dir.mkdir(parents=True, exist_ok=True)
    target = download_dir / _download_filename(info)
    partial = target.with_name(target.name + ".part")
    expected_sha256 = (info.sha256 or "").strip().lower()

    if target.exists() and expected_sha256 and _sha256_file(target) == expected_sha256:
        size = target.stat().st_size
        if progress_callback is not None:
            progress_callback(size, size)
        return target

    if partial.exists():
        partial.unlink()

    urls = part_urls or (url,)
    try:
        digest = hashlib.sha256()
        downloaded = 0
        with partial.open("wb") as handle:
            for index, item_url in enumerate(urls, start=1):
                request = urllib.request.Request(
                    item_url,
                    headers={
                        "Accept": "application/octet-stream, */*",
                        "User-Agent": f"XRD-Analyzer-Updater/{info.current_version}",
                    },
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    part_total = int(response.headers.get("Content-Length") or 0)
                    progress_total = part_total if len(urls) == 1 else 0
                    if progress_callback is not None and len(urls) > 1:
                        progress_callback(downloaded, 0)

                    expected_part = _part_index_from_url(item_url)
                    if expected_part is not None and expected_part != index:
                        raise UpdateDownloadError("更新分片顺序不正确，请重新检查更新清单。")

                    while True:
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if progress_callback is not None:
                            progress_callback(downloaded, progress_total)
    except urllib.error.HTTPError as exc:
        _remove_quietly(partial)
        raise UpdateDownloadError(f"下载安装包失败：服务器返回 HTTP {exc.code}。") from exc
    except urllib.error.URLError as exc:
        _remove_quietly(partial)
        reason = getattr(exc, "reason", exc)
        raise UpdateDownloadError(f"下载安装包失败：无法连接更新源：{reason}") from exc
    except TimeoutError as exc:
        _remove_quietly(partial)
        raise UpdateDownloadError("下载安装包失败：连接超时。") from exc
    except UpdateDownloadError:
        _remove_quietly(partial)
        raise
    except OSError as exc:
        _remove_quietly(partial)
        raise UpdateDownloadError(f"下载安装包失败：{exc}") from exc

    actual_sha256 = digest.hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        _remove_quietly(partial)
        raise UpdateDownloadError("安装包校验失败：下载文件的 SHA256 与更新清单不一致。")

    if target.exists():
        target.unlink()
    partial.replace(target)
    return target


def launch_update_and_exit(downloaded_exe: Path) -> None:
    exe_path = Path(downloaded_exe).resolve()
    if not exe_path.exists():
        raise UpdateDownloadError(f"找不到已下载的安装包：{exe_path}")

    if getattr(sys, "frozen", False):
        _launch_replacement_script(exe_path)
    else:
        subprocess.Popen(
            [str(exe_path)],
            cwd=str(exe_path.parent),
            close_fds=True,
            creationflags=_windows_detached_flags(),
        )


def _download_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "XRDAnalyzer" / "updates"
    return Path.home() / ".xrd_analyzer" / "updates"


def _download_filename(info: UpdateInfo) -> str:
    raw_name = (info.asset_name or "").strip() or "XRD-Analyzer.exe"
    suffix = Path(raw_name).suffix or ".exe"
    version = re.sub(r"[^0-9A-Za-z._-]+", "_", info.latest_version or "update")
    return f"XRD-Analyzer-v{version}{suffix}"


def _part_index_from_url(url: str) -> int | None:
    match = re.search(r"\.part(\d+)(?:$|[?#])", str(url), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_quietly(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _append_update_log(path: Path, message: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def _launch_replacement_script(downloaded_exe: Path) -> None:
    current_exe = Path(sys.executable).resolve()
    current_pid = os.getpid()
    script_dir = _download_dir()
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "apply_update.ps1"
    log_path = script_dir / "apply_update.log"
    backup_path = current_exe.with_name(current_exe.name + ".old")
    script_path.write_text(
        _replacement_script_text(current_exe, downloaded_exe, backup_path, log_path, current_pid),
        encoding="utf-8-sig",
    )
    _append_update_log(
        log_path,
        f"launcher prepared update script; old={current_exe} new={downloaded_exe} pid={current_pid}",
    )
    try:
        subprocess.Popen(
            [
                _powershell_executable(),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ],
            cwd=str(script_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=_windows_update_script_flags(),
        )
    except OSError as exc:
        _append_update_log(log_path, f"launcher failed to start update script: {exc}")
        raise UpdateDownloadError(f"Failed to launch update replacement script: {exc}") from exc


def _replacement_script_text(
    current_exe: Path,
    downloaded_exe: Path,
    backup_path: Path,
    log_path: Path,
    current_pid: int,
) -> str:
    return f"""$ErrorActionPreference = 'Stop'
$OldPath = {_ps_single_quoted(str(current_exe))}
$NewPath = {_ps_single_quoted(str(downloaded_exe))}
$BackupPath = {_ps_single_quoted(str(backup_path))}
$LogPath = {_ps_single_quoted(str(log_path))}
$OldProcessId = {int(current_pid)}
$TargetDir = Split-Path -Parent $OldPath

function Write-UpdateLog($Message) {{
    try {{
        $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'
        Add-Content -LiteralPath $LogPath -Value "$stamp $Message" -Encoding UTF8
    }} catch {{}}
}}

function Try-UnblockFile($Path) {{
    try {{
        if (Test-Path -LiteralPath $Path) {{
            Unblock-File -LiteralPath $Path -ErrorAction SilentlyContinue
            Write-UpdateLog ("unblocked " + $Path)
        }}
    }} catch {{
        Write-UpdateLog ("unblock failed for " + $Path + ": " + $_.Exception.Message)
    }}
}}

function Show-UpdateMessage($Message, $Title, $IconName) {{
    try {{
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction SilentlyContinue
        $icon = [System.Windows.Forms.MessageBoxIcon]::Information
        if ($IconName -eq "Warning") {{
            $icon = [System.Windows.Forms.MessageBoxIcon]::Warning
        }} elseif ($IconName -eq "Error") {{
            $icon = [System.Windows.Forms.MessageBoxIcon]::Error
        }}
        [System.Windows.Forms.MessageBox]::Show(
            $Message,
            $Title,
            [System.Windows.Forms.MessageBoxButtons]::OK,
            $icon
        ) | Out-Null
    }} catch {{
        Write-UpdateLog "message box failed: $($_.Exception.Message)"
        try {{ msg * $Message 2>$null }} catch {{}}
    }}
}}

function Set-PyInstallerRestartEnvironment() {{
    try {{
        $env:PYINSTALLER_RESET_ENVIRONMENT = "1"
        Get-ChildItem Env: | Where-Object {{ $_.Name -like "_PYI_*" }} | ForEach-Object {{
            try {{
                Remove-Item -LiteralPath ("Env:" + $_.Name) -ErrorAction SilentlyContinue
            }} catch {{}}
        }}
        Write-UpdateLog "prepared clean PyInstaller restart environment"
    }} catch {{
        Write-UpdateLog "prepare PyInstaller restart environment failed: $($_.Exception.Message)"
    }}
}}

function Start-UpdatedExecutable($Path, $WorkingDir) {{
    Set-PyInstallerRestartEnvironment
    Start-Sleep -Milliseconds 1200
    Start-Process -FilePath $Path -WorkingDirectory $WorkingDir
}}

Write-UpdateLog "apply update started; old=$OldPath new=$NewPath pid=$OldProcessId"

try {{
    $process = Get-Process -Id $OldProcessId -ErrorAction SilentlyContinue
    if ($process) {{
        Write-UpdateLog "waiting for old process to exit"
        Wait-Process -Id $OldProcessId -Timeout 60 -ErrorAction SilentlyContinue
    }}
}} catch {{
    Write-UpdateLog "wait failed: $($_.Exception.Message)"
}}

for ($i = 0; $i -lt 120; $i++) {{
    Start-Sleep -Milliseconds 500
    try {{
        if (-not (Test-Path -LiteralPath $NewPath)) {{
            throw "downloaded update is missing: $NewPath"
        }}
        Try-UnblockFile $NewPath
        if (Test-Path -LiteralPath $BackupPath) {{
            Remove-Item -LiteralPath $BackupPath -Force -ErrorAction SilentlyContinue
        }}
        if (Test-Path -LiteralPath $OldPath) {{
            Move-Item -LiteralPath $OldPath -Destination $BackupPath -Force
        }}
        Copy-Item -LiteralPath $NewPath -Destination $OldPath -Force
        Try-UnblockFile $OldPath
        Write-UpdateLog "replacement succeeded"
        try {{
            Show-UpdateMessage "更新成功！即将重启软件。" "软件更新" "Information"
            Write-UpdateLog "success restart message confirmed"
            Start-UpdatedExecutable $OldPath $TargetDir
            Write-UpdateLog "replacement launch started after confirmation"
        }} catch {{
            Write-UpdateLog "confirmed launch failed: $($_.Exception.Message)"
            throw
        }}
        Remove-Item -LiteralPath $BackupPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $NewPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue
        exit 0
    }} catch {{
        Write-UpdateLog "attempt $i failed: $($_.Exception.Message)"
        if ((-not (Test-Path -LiteralPath $OldPath)) -and (Test-Path -LiteralPath $BackupPath)) {{
            try {{
                Move-Item -LiteralPath $BackupPath -Destination $OldPath -Force
            }} catch {{}}
        }}
    }}
}}

Write-UpdateLog "replacement failed; asking user to run downloaded exe manually"
Try-UnblockFile $NewPath
try {{
    Show-UpdateMessage "自动更新失败，请手动运行已下载的更新文件：`n$NewPath" "软件更新失败" "Warning"
    Write-UpdateLog "fallback message shown"
    try {{
        Start-Process -FilePath explorer.exe -ArgumentList "/select,`"$NewPath`""
    }} catch {{}}
}} catch {{
    Write-UpdateLog "fallback message failed: $($_.Exception.Message)"
    try {{
        Start-Process -FilePath explorer.exe -ArgumentList "/select,`"$NewPath`""
    }} catch {{}}
    try {{ msg * "Update failed. Please manually run: $NewPath" 2>$null }} catch {{}}
}}
"""


def _ps_single_quoted(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _windows_detached_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0
    )


def _windows_update_script_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "CREATE_NO_WINDOW", 0
    )


def _powershell_executable() -> str:
    if os.name != "nt":
        return "powershell"
    system_root = os.environ.get("SystemRoot") or r"C:\Windows"
    path = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if path.exists():
        return str(path)
    return "powershell.exe"
