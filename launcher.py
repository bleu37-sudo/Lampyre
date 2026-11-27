import os
import sys
import socket
import threading
import time
import urllib.request
import webbrowser

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOG_FILE = os.path.join(DATA_DIR, "lampyre.log")


class TeeLogger:
    def __init__(self, filepath, stream):
        self.file = open(filepath, "a", encoding="utf-8", buffering=1)
        self.stream = stream
        self.encoding = getattr(stream, "encoding", None) or "utf-8"

    def write(self, msg):
        if not isinstance(msg, str):
            msg = str(msg)
        try:
            self.file.write(msg)
        except Exception:
            pass
        if self.stream is not None:
            try:
                self.stream.write(msg)
            except UnicodeEncodeError:
                try:
                    self.stream.write(msg.encode(self.encoding, errors="replace").decode(self.encoding, errors="replace"))
                except Exception:
                    pass
            except Exception:
                pass

    def flush(self):
        try:
            self.file.flush()
        except Exception:
            pass
        if self.stream is not None:
            try:
                self.stream.flush()
            except Exception:
                pass

    def isatty(self):
        return bool(self.stream and hasattr(self.stream, "isatty") and self.stream.isatty())

    def fileno(self):
        if self.stream is not None and hasattr(self.stream, "fileno"):
            return self.stream.fileno()
        raise OSError("TeeLogger has no fileno")


_tee = TeeLogger(LOG_FILE, sys.stdout)
sys.stdout = _tee
sys.stderr = TeeLogger(LOG_FILE, sys.stderr)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ["LAMPYRE_USER_DATA"] = DATA_DIR
os.environ["LAMPYRE_BUNDLE_DIR"] = APP_ROOT
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

import uvicorn
from backend.server import app


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_server(port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uvicorn.Server(config).run()


def wait_for_server(url: str, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/status", timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


if __name__ == "__main__":
    port = get_free_port()
    url = f"http://127.0.0.1:{port}"
    threading.Thread(target=run_server, args=(port,), daemon=True).start()
    if wait_for_server(url):
        print(f"[LAMPYRE] Application disponible sur : {url}", flush=True)
        print("[LAMPYRE] Cette adresse reste accessible si le navigateur ne s'ouvre pas automatiquement.", flush=True)
        webbrowser.open(url)
    while True:
        time.sleep(1)
