"""SQLite registry. Atomic result writes; PDF bytes addressed by SHA-256."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile
from io import BytesIO

MAX_FILE = 100 * 1024 * 1024
MAX_BATCH = 500 * 1024 * 1024
MAX_FILES = 100


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data")).resolve()


@contextmanager
def db(root: Path | None = None):
    root = root or data_dir()
    root.mkdir(parents=True, exist_ok=True)
    (root / "pdf").mkdir(exist_ok=True)
    conn = sqlite3.connect(root / "registry.sqlite3", timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS documents (sha TEXT PRIMARY KEY, filename TEXT NOT NULL, uploaded_at TEXT NOT NULL, status TEXT NOT NULL, force_ocr INTEGER DEFAULT 0, page INTEGER DEFAULT 0, pages INTEGER DEFAULT 0, error TEXT, result TEXT)")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def unpack_uploads(files: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    """Validate the entire batch before adding files. ZIP entries never touch disk."""
    output, total = [], 0
    def add(name, content):
        nonlocal total
        if not content[:1024].lstrip().startswith(b"%PDF-"):
            raise ValueError(f"{name}: файл не является PDF.")
        total += len(content)
        if len(content) > MAX_FILE or total > MAX_BATCH or len(output) >= MAX_FILES:
            raise ValueError("Лимит: 100 PDF, 100 МБ на PDF и 500 МБ на одну загрузку.")
        output.append((Path(name.replace("\\", "/")).name, content))
    for name, content in files:
        if len(content) > MAX_FILE:
            raise ValueError(f"{name}: файл больше 100 МБ.")
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(BytesIO(content)) as z:
                    entries = z.infolist()
                    if len(entries) > 1000:
                        raise ValueError("В ZIP слишком много файлов.")
                    pdfs = [i for i in entries if not i.is_dir() and i.filename.lower().endswith(".pdf") and not i.filename.startswith("__MACOSX/")]
                    if not pdfs:
                        raise ValueError(f"{name}: внутри ZIP нет PDF.")
                    for entry in pdfs:
                        parts = entry.filename.replace("\\", "/").split("/")
                        if ".." in parts or entry.filename.startswith(("/", "\\")):
                            raise ValueError("ZIP содержит недопустимые пути.")
                        if entry.flag_bits & 1:
                            raise ValueError("ZIP с паролем не поддерживается.")
                        if entry.file_size > MAX_FILE or total + entry.file_size > MAX_BATCH or entry.file_size > max(entry.compress_size, 1) * 250:
                            raise ValueError("ZIP превышает лимит распакованного объёма.")
                        add(entry.filename, z.read(entry))
            except zipfile.BadZipFile as exc:
                raise ValueError(f"{name}: повреждённый ZIP.") from exc
        elif name.lower().endswith(".pdf"):
            add(name, content)
        else:
            raise ValueError("Допустимы только PDF и ZIP с PDF.")
    return output


def enqueue(files, force_ocr=False, root=None):
    root = root or data_dir()
    added = skipped = 0
    with db(root) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for name, content in files:
            sha = hashlib.sha256(content).hexdigest()
            if conn.execute("SELECT 1 FROM documents WHERE sha=?", (sha,)).fetchone():
                skipped += 1
                continue
            target = root / "pdf" / f"{sha}.pdf"
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, target)
            conn.execute("INSERT INTO documents (sha, filename, uploaded_at, status, force_ocr) VALUES (?,?,?,?,?)",
                         (sha, name, datetime.now(timezone.utc).isoformat(), "queued", int(force_ocr)))
            added += 1
    return added, skipped


def documents(root=None):
    with db(root) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")]


def update(sha, root=None, **values):
    allowed = {"status", "page", "pages", "error", "result", "force_ocr"}
    if not values or not set(values) <= allowed:
        raise ValueError("Invalid update")
    with db(root) as conn:
        conn.execute("UPDATE documents SET " + ",".join(f"{k}=?" for k in values) + " WHERE sha=?", (*values.values(), sha))


def retry(sha, force_ocr=False, root=None):
    with db(root) as conn:
        # Do not race a running parser or enqueue an already queued document.
        conn.execute("UPDATE documents SET status='queued', page=0, pages=0, error=NULL, result=NULL, force_ocr=? WHERE sha=? AND status IN ('done','failed')", (int(force_ocr), sha))


def rows(docs, latest=True):
    parsed = [(d, json.loads(d["result"])) for d in docs if d["status"] == "done" and d["result"]]
    groups = {}
    for d, result in parsed:
        number = result["header"].get("declaration_number")
        # A declaration number carries the region. INN helps guard against unrelated documents.
        key = (result.get("inn") or "", number) if number else ("file", d["sha"])
        groups.setdefault(key, []).append((d, result))
    output = []
    for members in groups.values():
        dates = [r["header"].get("date_iso") for _, r in members if r["header"].get("date_iso")]
        max_date = max(dates) if dates else None
        newest = [(d,r) for d,r in members if r["header"].get("date_iso") == max_date] if max_date else members
        # Duplicate digital / scanned copies do not double-count identical object datasets.
        def fingerprint(r):
            return json.dumps([{k:v for k,v in o.items() if k not in {"evidence", "has_ocr", "issues", "status"}} for o in r["objects"]], sort_keys=True, ensure_ascii=False)
        same_date_conflict = len({fingerprint(r) for _,r in newest}) > 1
        unique = set()
        for d, result in members:
            date = result["header"].get("date_iso")
            current = bool(max_date and date == max_date) or (max_date is None and len(members) == 1)
            if latest and not current and date is not None:
                continue
            signature = (date, fingerprint(result))
            duplicate_revision = signature in unique
            unique.add(signature)
            for obj in result["objects"]:
                conflict = same_date_conflict and date == max_date
                eligible = bool(current and date and not conflict and not duplicate_revision)
                # History is descriptive; old revisions never inflate portfolio totals.
                version = "Конфликт одной даты" if conflict else "Дата не найдена" if not date else "Дубликат версии" if duplicate_revision else "Актуальная" if current else "Архивная"
                output.append(obj | dict(sha=d["sha"], filename=d["filename"], developer=result["developer"], inn=result["inn"],
                    declaration_number=result["header"].get("declaration_number"), declaration_date=date,
                    version=version, eligible=eligible, document_warnings=result["warnings"]))
    return output


def summary(items):
    active = [r for r in items if r.get("eligible")]
    cost_rows = [r for r in active if r.get("planned_cost") is not None]
    def weighted(field, ratio):
        valid = [r for r in active if r.get(ratio) is not None and r.get(field) and r.get("planned_cost") is not None]
        return (sum(r["planned_cost"] for r in valid) / sum(r[field] for r in valid), len(valid)) if valid else (None, 0)
    return dict(objects=len(active), cost=sum(r["planned_cost"] for r in cost_rows) if cost_rows else None,
                cost_coverage=len(cost_rows), gross=weighted("gross_area", "cost_per_gross"), saleable=weighted("saleable_area", "cost_per_saleable"))
