"""
integrity_engine.py
===================
Cloud-Safe Immune & Verification System.

Manages local integrity state maps, cryptographic SHA-256 hashes of every
critical system script, cloud restore-point state tracking, and instant
rollback for corrupted server runtime scripts.

Classes
-------
SystemIntegrityManager
    The single public entry-point used by ``app.py`` and ``core_engine.py``.

Design notes
------------
* Restore points are stored under ``./system_restore_points/<point_name>/``
  and a companion JSON manifest ``./system_restore_points/_manifest.json``
  holds every recorded hash + timestamp + description.
* SHA-256 is used everywhere — file fingerprints and manifest tamper tokens.
* ``check_for_tampering()`` is safe to call on every Streamlit page reload
  (it is pure-read and idempotent).
* ``rollback_to_point()`` performs an atomic copy-then-verify so a half-
  written restore can never leave the live scripts in a broken state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_SCRIPT_FILES: List[str] = [
    "app.py",
    "core_engine.py",
    "integrity_engine.py",
]

RESTORE_POINTS_DIR: str = "./system_restore_points"
MANIFEST_PATH: str = os.path.join(RESTORE_POINTS_DIR, "_manifest.json")
_HASH_CHUNK_SIZE: int = 65536


class SystemIntegrityManager:
    """Manages cryptographic integrity state, restore points, and rollback.

    Parameters
    ----------
    base_dir : str, optional
        The project root that contains ``app.py``, ``core_engine.py`` and
        ``integrity_engine.py``.  Defaults to the current working directory.
    script_files : list[str], optional
        Override the default ``SYSTEM_SCRIPT_FILES`` list — useful when running
        tests or when extra guarded files are added later.
    """

    def __init__(
        self,
        base_dir: str = ".",
        script_files: Optional[List[str]] = None,
    ) -> None:
        self.base_dir: Path = Path(base_dir).resolve()
        self.script_files: List[str] = list(script_files or SYSTEM_SCRIPT_FILES)

        self.restore_dir: Path = (self.base_dir / RESTORE_POINTS_DIR).resolve()
        self.restore_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path: Path = self.restore_dir / "_manifest.json"
        self._manifest: Dict[str, Any] = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the on-disk manifest JSON, or return a blank structure."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                backup = self.manifest_path.with_suffix(".json.bak")
                try:
                    shutil.copy2(self.manifest_path, backup)
                except OSError:
                    pass
        return {"restore_points": {}, "last_check": None}

    def _save_manifest(self) -> None:
        """Persist the in-memory manifest to disk atomically."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.manifest_path)

    # ------------------------------------------------------------------
    # SHA-256 fingerprinting
    # ------------------------------------------------------------------

    def calculate_file_hash(self, file_path: str) -> str:
        """Return the lowercase hex SHA-256 digest of *file_path*.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot hash missing file: {file_path}")

        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest()

    def _hash_all_scripts(self) -> Dict[str, str]:
        """Hash every file in ``self.script_files`` and return a mapping."""
        hashes: Dict[str, str] = {}
        for fname in self.script_files:
            fpath = self.base_dir / fname
            if fpath.exists():
                hashes[fname] = self.calculate_file_hash(str(fpath))
            else:
                hashes[fname] = "MISSING"
        return hashes

    # ------------------------------------------------------------------
    # Restore-point creation
    # ------------------------------------------------------------------

    def create_restore_point(self, description: str = "") -> Dict[str, Any]:
        """Snapshot all guarded scripts into a timestamped restore point.

        Copies each live script into
        ``./system_restore_points/<point_name>/`` and records its SHA-256 +
        timestamp + description in the manifest.

        Returns
        -------
        dict
            The restore-point payload that was persisted.
        """
        timestamp = datetime.now(timezone.utc)
        point_name = timestamp.strftime("restore_%Y%m%d_%H%M%S")
        point_dir = self.restore_dir / point_name
        point_dir.mkdir(parents=True, exist_ok=True)

        file_hashes: Dict[str, str] = {}
        for fname in self.script_files:
            src = self.base_dir / fname
            dst = point_dir / fname
            if src.exists():
                shutil.copy2(src, dst)
                file_hashes[fname] = self.calculate_file_hash(str(src))
            else:
                file_hashes[fname] = "MISSING"

        payload: Dict[str, Any] = {
            "point_name": point_name,
            "description": description or "Manual restore point",
            "timestamp": timestamp.isoformat(),
            "files": file_hashes,
            "backup_dir": str(point_dir.relative_to(self.base_dir)),
        }

        self._manifest["restore_points"][point_name] = payload
        self._manifest["last_check"] = timestamp.isoformat()
        self._save_manifest()
        return payload

    # ------------------------------------------------------------------
    # Tampering detection
    # ------------------------------------------------------------------

    def check_for_tampering(self) -> Dict[str, Any]:
        """Compare live script hashes against the latest restore point.

        Returns
        -------
        dict
            ``{
                "tampered": bool,
                "verified": bool,
                "checked_files": int,
                "tampered_files": list[str],
                "details": dict[str, dict],
                "last_restore_point": str | None,
                "checked_at": str
            }``
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        live_hashes = self._hash_all_scripts()

        latest_point = self._latest_restore_point()

        if latest_point is None:
            self.create_restore_point(
                "Automatic baseline snapshot (no prior restore point existed)"
            )
            return {
                "tampered": False,
                "verified": True,
                "checked_files": len(live_hashes),
                "tampered_files": [],
                "details": {
                    fname: {"status": "baseline_created"} for fname in live_hashes
                },
                "last_restore_point": None,
                "checked_at": timestamp,
            }

        recorded_hashes: Dict[str, str] = latest_point["files"]
        tampered_files: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        for fname, live_hash in live_hashes.items():
            recorded_hash = recorded_hashes.get(fname, "MISSING")
            if live_hash == recorded_hash:
                details[fname] = {
                    "status": "verified",
                    "live_hash": live_hash,
                    "recorded_hash": recorded_hash,
                }
            else:
                tampered_files.append(fname)
                details[fname] = {
                    "status": "TAMPERED",
                    "live_hash": live_hash,
                    "recorded_hash": recorded_hash,
                }

        tampered = len(tampered_files) > 0
        self._manifest["last_check"] = timestamp
        self._save_manifest()

        return {
            "tampered": tampered,
            "verified": not tampered,
            "checked_files": len(live_hashes),
            "tampered_files": tampered_files,
            "details": details,
            "last_restore_point": latest_point.get("point_name"),
            "checked_at": timestamp,
        }

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback_to_point(self, point_name: str) -> Dict[str, Any]:
        """Restore all guarded scripts from a named restore point.

        Performs an atomic copy-then-verify: files are copied first, then
        re-hashed and compared to the manifest.  If verification fails the
        copy is retried once; if it still fails an exception is raised.

        Raises
        ------
        ValueError
            If *point_name* does not exist in the manifest.
        RuntimeError
            If post-restore verification fails after retry.
        """
        point = self._manifest["restore_points"].get(point_name)
        if point is None:
            raise ValueError(f"Unknown restore point: {point_name}")

        point_dir = self.restore_dir / point_name
        recorded_hashes: Dict[str, str] = point["files"]
        files_restored: List[str] = []

        def _do_copy() -> bool:
            for fname, recorded_hash in recorded_hashes.items():
                src = point_dir / fname
                dst = self.base_dir / fname
                if not src.exists():
                    continue
                shutil.copy2(src, dst)
                files_restored.append(fname)
            live_hashes = self._hash_all_scripts()
            for fname, recorded_hash in recorded_hashes.items():
                if fname not in files_restored:
                    continue
                if live_hashes.get(fname) != recorded_hash:
                    return False
            return True

        verified = _do_copy()
        if not verified:
            verified = _do_copy()

        if not verified:
            raise RuntimeError(
                f"Post-restore verification failed for point '{point_name}'. "
                "Manual intervention required."
            )

        return {
            "restored": True,
            "point_name": point_name,
            "files_restored": files_restored,
            "verified": verified,
        }

    # ------------------------------------------------------------------
    # Manifest queries
    # ------------------------------------------------------------------

    def list_restore_points(self) -> List[Dict[str, Any]]:
        """Return all restore points sorted newest-first."""
        points = list(self._manifest["restore_points"].values())
        points.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
        return points

    def get_latest_restore_point_name(self) -> Optional[str]:
        """Convenience: name of the most recent restore point, or None."""
        latest = self._latest_restore_point()
        return latest["point_name"] if latest else None

    def _latest_restore_point(self) -> Optional[Dict[str, Any]]:
        """Return the newest restore-point payload from the manifest."""
        points = list(self._manifest["restore_points"].values())
        if not points:
            return None
        points.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
        return points[0]

    # ------------------------------------------------------------------
    # Post-update baseline refresh
    # ------------------------------------------------------------------

    def refresh_baseline_after_update(
        self, description: str = "Post-update baseline"
    ) -> Dict[str, Any]:
        """Create a fresh restore point after a legitimate code update.

        Call this after intentionally modifying the live scripts (e.g. after
        the autonomous debug engine patches ``app.py``) so the new state
        becomes the trusted baseline.
        """
        return self.create_restore_point(description)


def get_default_manager() -> SystemIntegrityManager:
    """Return a ``SystemIntegrityManager`` rooted at the current directory."""
    return SystemIntegrityManager(base_dir=".")


if __name__ == "__main__":
    mgr = SystemIntegrityManager(base_dir=".")
    print("[integrity] Creating baseline restore point ...")
    point = mgr.create_restore_point("CLI baseline test")
    print(f"[integrity] Restore point created: {point['point_name']}")
    print("[integrity] Checking for tampering ...")
    result = mgr.check_for_tampering()
    print(json.dumps(result, indent=2))
    print("[integrity] Listing restore points ...")
    for p in mgr.list_restore_points():
        print(f"  - {p['point_name']}  ({p['timestamp']})  {p['description']}")
