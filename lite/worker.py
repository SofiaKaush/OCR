"""Single PDF worker per data directory; state survives a closed browser tab."""
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from filelock import FileLock, Timeout
from .storage import data_dir, db, documents, update


def ensure_worker():
    root = data_dir()
    if not any(d["status"] in ("queued", "processing") for d in documents(root)):
        return
    # Try the lock first so Streamlit reruns do not keep spawning contenders.
    lock = FileLock(str(root / "worker.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return
    lock.release()
    with (root / "worker.log").open("ab") as log:
        subprocess.Popen([sys.executable, "-m", "lite.worker"], cwd=Path(__file__).resolve().parents[1],
                         env=os.environ | {"DATA_DIR": str(root)}, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=log, start_new_session=True)


def work():
    root = data_dir()
    with db(root):
        pass
    try:
        with FileLock(str(root / "worker.lock"), timeout=0):
            # Only the lock owner can recover work abandoned after a restart.
            with db(root) as conn:
                conn.execute("UPDATE documents SET status='queued', error=NULL WHERE status='processing'")
            while True:
                queued = next((d for d in reversed(documents(root)) if d["status"] == "queued"), None)
                if not queued:
                    return
                sha = queued["sha"]
                update(sha, status="processing", page=0)
                try:
                    from .parser import parse_pdf
                    result = parse_pdf((root / "pdf" / f"{sha}.pdf").read_bytes(), queued["filename"],
                                       force_ocr=bool(queued["force_ocr"]),
                                       progress=lambda page, pages: update(sha, page=page, pages=pages))
                    update(sha, status="done", result=json.dumps(result, ensure_ascii=False), error=None)
                except Exception as exc:
                    logging.exception("Document processing failed: %s", sha)
                    update(sha, status="failed", error=f"{type(exc).__name__}: {str(exc)[:400]}")
    except Timeout:
        return


if __name__ == "__main__":
    work()
