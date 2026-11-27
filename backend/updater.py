import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


DEFAULT_REPOSITORY = "https://github.com/bleu37-sudo/Lampyre"
DEFAULT_ASSET = "lampyre.zip"
DEFAULT_BRANCH = "main"


class Updater:
    def __init__(self, app_root: str, repository: str = DEFAULT_REPOSITORY, asset_name: str = DEFAULT_ASSET, branch: str = DEFAULT_BRANCH):
        self.app_root = Path(app_root).resolve()
        self.repository = repository.rstrip("/")
        self.asset_name = asset_name
        self.branch = branch
        self.owner, self.repo = self._parse_repository(repository)
        self.lock = threading.Lock()
        self.downloading = False
        self.pending_restart = False
        self.last_error = ""
        self.latest_version = ""
        self.release_url = ""
        self.download_url = ""

    @staticmethod
    def _parse_repository(repository: str):
        parts = repository.rstrip("/").split("/")
        if len(parts) < 2:
            raise ValueError("URL de dépôt GitHub invalide")
        return parts[-2], parts[-1].removesuffix(".git")

    def current_version(self) -> str:
        version_file = self.app_root / "VERSION"
        if version_file.exists():
            return version_file.read_text(encoding="utf-8").strip() or "0.0.0"
        return "0.0.0"

    @staticmethod
    def _normalize_version(version: str) -> str:
        value = (version or "").strip()
        if value.lower().startswith("v"):
            value = value[1:]
        return value

    def _urlopen(self, url: str):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Lampyre-Updater/1.0",
            },
        )
        context = ssl.create_default_context()
        return urllib.request.urlopen(request, timeout=30, context=context)

    def check(self) -> dict:
        with self.lock:
            self.last_error = ""
            self.downloading = False
        try:
            version_url = (
                f"https://raw.githubusercontent.com/"
                f"{self.owner}/{self.repo}/{self.branch}/VERSION"
            )
            with self._urlopen(version_url) as response:
                latest = self._normalize_version(
                    response.read().decode("utf-8")
                )
            if not latest:
                raise RuntimeError("Le fichier VERSION distant est vide.")

            api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/releases/latest"
            with self._urlopen(api_url) as response:
                release = json.load(response)

            download_url = ""
            for asset in release.get("assets") or []:
                if asset.get("name") == self.asset_name:
                    download_url = str(asset.get("browser_download_url") or "")
                    break
            if not download_url:
                download_url = str(release.get("zipball_url") or "")
            if not download_url:
                raise RuntimeError("Aucune archive téléchargeable dans la dernière release.")

            current = self._normalize_version(self.current_version())
            update_available = latest != current and bool(latest)
            with self.lock:
                self.latest_version = latest
                self.release_url = str(release.get("html_url") or f"{self.repository}/releases")
                self.download_url = download_url
            return self.status_locked(update_available)
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
            return self.status_locked(False)

    def status_locked(self, update_available: Optional[bool] = None) -> dict:
        if update_available is None:
            current = self.current_version()
            update_available = bool(self.latest_version) and (
                self._normalize_version(self.latest_version) != self._normalize_version(current)
            )
        return {
            "current_version": self.current_version(),
            "latest_version": self.latest_version,
            "update_available": update_available,
            "downloading": self.downloading,
            "pending_restart": self.pending_restart,
            "release_url": self.release_url,
            "error": self.last_error,
        }

    def status(self) -> dict:
        with self.lock:
            return self.status_locked()

    def download_and_apply(self) -> dict:
        with self.lock:
            if self.downloading:
                return {"ok": False, "error": "Une mise à jour est déjà en cours."}
            if self.pending_restart:
                return {"ok": True, "message": "La mise à jour est prête. Redémarrage en cours."}
            self.downloading = True
            self.last_error = ""
            download_url = self.download_url

        if not download_url:
            checked = self.check()
            with self.lock:
                download_url = self.download_url
            if checked.get("error"):
                with self.lock:
                    self.downloading = False
                return {"ok": False, "error": checked["error"]}

        staging = self.app_root.parent / ".lampyre-update"
        archive = staging.parent / ".lampyre-update.zip"
        try:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            if archive.exists():
                archive.unlink()

            with self._urlopen(download_url) as response:
                with open(archive, "wb") as target:
                    shutil.copyfileobj(response, target)

            if archive.stat().st_size > 1024 * 1024 * 1024:
                raise RuntimeError("Archive de mise à jour trop volumineuse.")

            with zipfile.ZipFile(archive, "r") as package:
                for member in package.infolist():
                    destination = (staging / member.filename).resolve()
                    if not str(destination).startswith(str(staging.resolve()) + os.sep):
                        raise RuntimeError("Archive contenant un chemin invalide.")
                package.extractall(staging)

            package_root = staging
            if not (package_root / "launcher.py").exists():
                children = [child for child in staging.iterdir() if child.is_dir()]
                matches = [
                    child for child in children
                    if (child / "launcher.py").exists()
                ]
                if len(matches) == 1:
                    package_root = matches[0]

            required = [
                package_root / "launcher.py",
                package_root / "backend" / "server.py",
                package_root / "frontend" / "index.html",
                package_root / "Lancer_Lampyre.bat",
            ]
            missing = [str(path.relative_to(package_root)) for path in required if not path.exists()]
            if missing:
                raise RuntimeError("Archive incomplète : " + ", ".join(missing))

            protected = {"runtime", "data", ".git", ".lampyre-update"}
            for source in package_root.rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(package_root)
                if relative.parts and relative.parts[0] in protected:
                    continue
                destination = self.app_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

            version_file = package_root / "VERSION"
            if version_file.exists():
                shutil.copy2(version_file, self.app_root / "VERSION")

            with self.lock:
                self.pending_restart = True
                self.downloading = False
            return {
                "ok": True,
                "message": f"Lampyre {self.latest_version} a été installée. Redémarrage nécessaire.",
                "restart_required": True,
            }
        except Exception as exc:
            with self.lock:
                self.downloading = False
                self.last_error = str(exc)
            return {"ok": False, "error": str(exc)}
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if archive.exists():
                archive.unlink(missing_ok=True)

    def restart(self) -> dict:
        if not self.pending_restart:
            return {"ok": False, "error": "Aucune mise à jour installée en attente."}

        def stop_and_restart():
            try:
                subprocess.Popen(
                    [sys.executable, str(self.app_root / "launcher.py")],
                    cwd=str(self.app_root),
                    close_fds=True,
                )
            except Exception:
                pass
            os._exit(0)

        threading.Timer(0.8, stop_and_restart).start()
        return {"ok": True, "message": "Lampyre va redémarrer."}
