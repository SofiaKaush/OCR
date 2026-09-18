"""Evidence-preserving parser for Russian project declarations: PDF text, tables and OCR.

No language model, invented costs, external calls or allocation of project totals.
OCR results follow the same object-scoped field mapping as native PDF text.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from io import BytesIO
import math
import re
import time
from typing import Callable

import fitz
import cv2
import numpy as np
import pytesseract
from PIL import Image

from .vendor.pdf_layout import (
    LayoutDeclarationExtractor, PipelineConfig, normalize_text, parse_ru_number,
)

VERSION = "lite-1.0.0"
CODES = {
    "1.1.2", "1.1.3", "2.1.1", "2.1.2", "9.2.1", "9.2.2", "9.2.3",
    "9.1.1", "9.2.6", "9.2.17", "9.2.19", "9.2.20", "9.2.21", "9.3.1", "9.3.2",
    "9.3.3", "10.6.1", "11.1.1", "11.1.2", "17.2.1", "17.2.2", "18.1.1",
}
MARKER = re.compile(r"Объект\s*(?:№|N[oо]?\.?|#)\s*(\d+)", re.I)
UNITS = re.compile(r"(?:тыс\.?|млн\.?|млрд\.?)?\s*(?:руб\.?|₽)", re.I)


def iso_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%d.%m.%Y").date().isoformat()
    except ValueError:
        return None


def amount(value: str, label: str = "") -> float | None:
    """Accept one monetary amount, with units explicitly in value or label."""
    if not value or re.search(r"\d\s*[-–—]\s*\d|\b(?:от|до)\s+\d", value, re.I):
        return None
    unit_match = UNITS.search(value) or UNITS.search(label)
    if not unit_match:
        return None
    number = parse_ru_number(value)
    if number is None or not math.isfinite(number) or number < 0:
        return None
    unit = unit_match.group().lower()
    multiplier = 1e9 if "млрд" in unit else 1e6 if "млн" in unit else 1e3 if "тыс" in unit else 1
    return number * multiplier


def positive(value: str) -> float | None:
    if not value or re.search(r"\d\s*[-–—]\s*\d|\b(?:от|до)\s+\d", value, re.I):
        return None
    number = parse_ru_number(value)
    return number if number is not None and math.isfinite(number) and number > 0 else None


def _split_cell(text: str) -> tuple[str, str]:
    text = normalize_text(text)
    if ":" in text:
        label, value = text.split(":", 1)
        return label + ":", value.strip()
    return text, ""


def _native_fields(page, spans, helper) -> list[dict]:
    found = []
    try:
        tables = page.find_tables().tables
    except Exception:
        return found
    for table in tables:
        for row, values in zip(table.rows, table.extract()):
            for idx, raw in enumerate(values):
                code = normalize_text(raw or "")
                if code not in CODES or row.cells[idx] is None:
                    continue
                candidates = [tuple(c) for c in row.cells[idx + 1:] if c is not None]
                candidates = list(dict.fromkeys(candidates))
                if not candidates:
                    continue
                box = candidates[0]
                cell_spans = [s for s in spans if helper._span_in_bbox(s, box)]
                bold = [s for s in cell_spans if s.bold]
                if bold:
                    label = helper._join_spans([s for s in cell_spans if not s.bold])
                    value = helper._join_spans(bold)
                else:
                    text = helper._join_spans(cell_spans)
                    label, value = _split_cell(text)
                if len(candidates) == 2 and not value:
                    label = helper._join_spans(cell_spans)
                    box = candidates[1]
                    value = helper._join_spans([s for s in spans if helper._span_in_bbox(s, box)])
                found.append(dict(code=code, label=label, value=value, bbox=list(box),
                                  y=row.cells[idx][1], method="PDF / таблица"))
    return found


def _spatial_fields(page, words, method: str, row_lines=None) -> list[dict]:
    """Read the answer zone to the right of a field code; never across pages."""
    anchors = []
    # Other official field codes delimit the answer even if we do not export them.
    for w in words:
        token = w[4].strip(" :;")
        if re.fullmatch(r"\d{1,2}(?:\.\d{1,3}){2,7}", token) and w[0] < page.rect.width * .6:
            anchors.append((w, token))
    anchors.sort(key=lambda x: (x[0][1], x[0][0]))
    found = []
    for i, (w, code) in enumerate(anchors):
        if code not in CODES:
            continue
        top = w[1] - 4
        bottom = anchors[i + 1][0][1] - 1 if i + 1 < len(anchors) else min(page.rect.height - 18, w[1] + 210)
        row_center = (w[1] + w[3]) / 2
        borders = [line[1] for line in (row_lines or []) if line[0] - 3 <= w[0] <= line[2] + 3]
        above = [y for y in borders if y < row_center - 3]
        below = [y for y in borders if y > row_center + 3]
        if above and below:
            top, bottom = max(above), min(below)
        selected = [z for z in words if z[0] >= w[2] + 2 and z[1] >= top - 2 and z[1] < bottom - 1]
        if not selected:
            continue
        lines = defaultdict(list)
        for z in selected:
            lines[(z[5], z[6])].append(z)
        text = " ".join(" ".join(z[4] for z in sorted(line, key=lambda z: z[0]))
                        for line in sorted(lines.values(), key=lambda line: min(z[1] for z in line)))
        label, value = _split_cell(text)
        if not value:
            continue
        found.append(dict(code=code, label=label, value=value,
                          bbox=[min(z[0] for z in selected), min(z[1] for z in selected),
                                max(z[2] for z in selected), max(z[3] for z in selected)],
                          y=w[1], method=method))
    return found


def _ocr(page, dpi=260):
    # Bound raster size to avoid exhausting memory on oversized architectural pages.
    if not {"rus", "eng"} <= set(pytesseract.get_languages()):
        raise RuntimeError("Установите языки Tesseract rus и eng")
    scale = min(dpi / 72, (18_000_000 / max(page.rect.width * page.rect.height, 1)) ** .5)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    image = Image.open(BytesIO(pix.tobytes("png")))
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(35*scale)), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35,int(25*scale)))))
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = [cv2.boundingRect(c) for c in contours]
    lines = [(x, y+h/2, x+w) for x,y,w,h in lines if w > pix.width*.2]
    # Table rules otherwise cause Tesseract to discard the narrow field-code column.
    mask = cv2.bitwise_or(horizontal, vertical)
    clean = gray.copy()
    clean[cv2.dilate(mask, np.ones((2,2),np.uint8)) > 0] = 255
    searchable = pytesseract.image_to_pdf_or_hocr(Image.fromarray(clean), extension="pdf", lang="rus+eng", config=f"--psm 6 --dpi {round(scale*72)}", timeout=150)
    doc = fitz.open(stream=searchable, filetype="pdf")
    sx, sy = doc[0].rect.width / pix.width, doc[0].rect.height / pix.height
    # Read narrow code cells separately. Whole-page OCR can omit dots or entire codes.
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = [cv2.boundingRect(c) for c in contours]
    grid_fields = []
    for x,y,w,h in sorted(cells, key=lambda b:(b[1],b[0])):
        if not (pix.width*.20 < x < pix.width*.43 and 20*scale < w < 90*scale and 9*scale < h < 240*scale):
            continue
        code_image = clean[y+2:y+h-2, x+2:x+w-2]
        if not code_image.size:
            continue
        code_image = cv2.copyMakeBorder(code_image,8,8,8,8,cv2.BORDER_CONSTANT,value=255)
        code = pytesseract.image_to_string(code_image, lang="eng", config="--psm 6 -c tessedit_char_whitelist=0123456789.", timeout=25).strip()
        if code not in CODES:
            continue
        answers = [b for b in cells if x+w-4 <= b[0] < x+w+12 and b[1]-4 <= y+h/2 <= b[1]+b[3]+4 and b[2] > 90*scale]
        if not answers:
            continue
        ax,ay,aw,ah = min(answers, key=lambda b:b[2]*b[3])
        answer_image = clean[ay+2:ay+ah-2, ax+2:ax+aw-2]
        text = pytesseract.image_to_string(answer_image, lang="rus+eng", config="--psm 6", timeout=30)
        label, value = _split_cell(text)
        grid_fields.append(dict(code=code,label=label,value=value,bbox=[ax*sx,ay*sy,(ax+aw)*sx,(ay+ah)*sy],
                                y=y*sy,method="OCR / ячейка таблицы"))
    return doc, [(x*sx,y*sy,right*sx) for x,y,right in lines], grid_fields


def parse_pdf(pdf: bytes, filename: str, force_ocr=False,
              progress: Callable[[int, int], None] | None = None) -> dict:
    started = time.monotonic()
    helper = LayoutDeclarationExtractor(PipelineConfig(enable_ocr=False, use_cache=False))
    warnings, fields, first_lines = [], [], []
    scope, overview_no, area_no, ocr_pages = None, 0, 0, 0
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        if doc.needs_pass:
            raise ValueError("PDF защищён паролем. Загрузите копию без пароля.")
        if not len(doc) or len(doc) > 600:
            raise ValueError("Поддерживаются PDF от 1 до 600 страниц.")
        total_pages = len(doc)
        for pageno, original in enumerate(doc):
            if progress:
                progress(pageno, total_pages)
            page, ocr_doc, method = original, None, "PDF / расположение текста"
            row_lines, grid_fields = [], []
            text = original.get_text()
            scan = force_ocr or len(re.sub(r"\s", "", text)) < 80
            # Also OCR scanned pages with a small added text footer/header.
            if not scan and len(text) < 500:
                image_rects = [fitz.Rect(i["bbox"]) for i in original.get_image_info()]
                scan = any(r.get_area() > original.rect.get_area() * .65 for r in image_rects)
            if scan:
                try:
                    ocr_doc, row_lines, grid_fields = _ocr(original)
                    page, method = ocr_doc[0], "OCR / расположение текста"
                    ocr_pages += 1
                except Exception as exc:
                    warnings.append(f"Страница {pageno + 1}: OCR не выполнен ({type(exc).__name__}: {str(exc)[:140]}).")
            try:
                spans = helper._spans_from_page(page)
                lines = helper._group_span_lines(spans)
                if pageno == 0:
                    first_lines = lines
                words = page.get_text("words", sort=True)
                candidates = grid_fields if ocr_doc else _native_fields(page, spans, helper)
                fallback = _spatial_fields(page, words, method, row_lines)
                for item in fallback:
                    match = next((c for c in candidates if c["code"] == item["code"] and (c["bbox"][1] - 5 <= item["y"] <= c["bbox"][3] + 2)), None)
                    if match is None:
                        candidates.append(item)
                    elif not match["value"]:
                        match.update(item)
                # Read object markers in geometric order, including multiple objects on a page.
                events = [(c["y"], "field", c) for c in candidates]
                line_groups = defaultdict(list)
                for w in words:
                    line_groups[(w[5], w[6])].append(w)
                for group in line_groups.values():
                    line = " ".join(w[4] for w in sorted(group, key=lambda w: w[0]))
                    marker = MARKER.search(line)
                    if marker:
                        events.append((min(w[1] for w in group), "marker", int(marker.group(1))))
                seen = set()
                for _, kind, item in sorted(events, key=lambda e: e[0]):
                    if kind == "marker":
                        scope = item
                        continue
                    code = item["code"]
                    dedup = (code, round(item["y"] / 8), item["value"])
                    if dedup in seen:
                        continue
                    seen.add(dedup)
                    if code == "9.2.1":
                        overview_no += 1
                    if code == "9.3.1":
                        area_no += 1
                    obj = ((overview_no or 1) if code.startswith("9.2.") else
                           (area_no or 1) if code.startswith("9.3.") else
                           scope if int(code.split(".")[0]) >= 10 else None)
                    box = item["bbox"]
                    if ocr_doc:
                        sx, sy = original.rect.width / page.rect.width, original.rect.height / page.rect.height
                        box = [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]
                    fields.append({k: v for k, v in item.items() if k not in ("bbox", "y")} |
                                  dict(page=pageno + 1, object_no=obj, bbox=box))
            finally:
                if ocr_doc:
                    ocr_doc.close()
        if progress:
            progress(total_pages, total_pages)
    finally:
        doc.close()
    header = helper._parse_header(first_lines, " ".join(first_lines)).model_dump()
    # OCR commonly prints N instead of №. Only accept an explicit declaration header.
    if not header.get("declaration_number"):
        m = re.search(r"(?:№|No|N)\s*([\d-]{4,})\s+от\s+(\d{2}\.\d{2}\.\d{4})", " ".join(first_lines), re.I)
        if m:
            header.update(declaration_number=m[1], declaration_date=m[2])
    header["date_iso"] = iso_date(header.get("declaration_date"))
    def global_value(code):
        return next((f["value"] for f in fields if f["code"] == code and f["value"]), None)
    developer = global_value("1.1.3") or global_value("1.1.2")
    inn = re.sub(r"\D", "", global_value("2.1.1") or "") or None
    expected = positive(global_value("9.1.1") or "")
    scoped = {f["object_no"] for f in fields if f["object_no"] is not None and int(f["code"].split(".")[0]) >= 10}
    expected = int(expected) if expected and expected.is_integer() else len(scoped) or None
    # Missing grouping anchors must not shift another corpus's area into a cost ratio.
    for prefix, count in [("9.2.", overview_no), ("9.3.", area_no)]:
        if expected and count != expected:
            warnings.append(f"Раздел {prefix[:-1]}: найдено групп {count} из {expected}; привязка этих полей к корпусам не подтверждена.")
            for f in fields:
                if f["code"].startswith(prefix):
                    f["object_no"] = None
    object_nos = sorted({f["object_no"] for f in fields if f["object_no"] is not None})
    if not object_nos and any(f["code"] == "18.1.1" for f in fields):
        # No object marker => keep the value in evidence, do not allocate it.
        warnings.append("Стоимость найдена без привязки к объекту; распределение не выполнялось.")
    if len(object_nos) == 1:
        for f in fields:
            if f["object_no"] is None and int(f["code"].split(".")[0]) >= 10:
                f["object_no"] = object_nos[0]
    objects = []
    for no in object_nos:
        issues, evidence = [], {}
        subset = [f for f in fields if f["object_no"] == no]
        def read(code, numeric=None):
            matches = [f for f in subset if f["code"] == code and f["value"]]
            # Older and newer formats reuse 18.1.1 for different payments.
            if code == "18.1.1":
                matches = [f for f in matches if re.search(r"стоимост.*строител", f["label"], re.I)]
            if not matches:
                return None
            vals = [(numeric(f["value"], f["label"]) if numeric is amount else numeric(f["value"]) if numeric else f["value"]) for f in matches]
            distinct = {v for v in vals if v is not None}
            if len(distinct) > 1:
                issues.append(f"Поле {code}: противоречивые значения")
                return None
            evidence[code] = matches[0]
            if numeric and not distinct:
                issues.append(f"Поле {code}: число или единица измерения не распознаны")
            return next(iter(distinct), None)
        gross, sale = read("9.2.21", positive), read("9.3.3", positive)
        cost = read("18.1.1", amount)
        if gross and sale and sale > gross * 1.02:
            issues.append("Продаваемая площадь больше ОСП; удельные показатели не рассчитаны")
        bad_area = bool(gross and sale and sale > gross * 1.02)
        missing = [label for value, label in [(gross,"ОСП"),(sale,"продаваемая площадь"),(cost,"стоимость")] if value is None]
        if missing:
            issues.append("Не найдены: " + ", ".join(missing))
        row = dict(object_no=no, project=read("10.6.1") or header.get("title"),
                   name=read("9.2.2") or f"Объект №{no}", address=read("9.2.17"),
                   region=read("9.2.3"), settlement=read("9.2.6"),
                   floors_min=read("9.2.19", positive), floors_max=read("9.2.20", positive),
                   residential_area=read("9.3.1", positive), nonresidential_area=read("9.3.2", positive),
                   gross_area=gross, saleable_area=sale, planned_cost=cost,
                   cost_per_gross=cost/gross if cost is not None and gross and not bad_area else None,
                   cost_per_saleable=cost/sale if cost is not None and sale and not bad_area else None,
                   saleable_share=sale/gross if gross and sale and not bad_area else None,
                   permit=read("11.1.1"), transfer_date=read("17.2.2") or read("17.2.1"),
                   issues=issues, evidence=evidence,
                   status="Данные получены" if not issues else "Часть данных отсутствует",
                   has_ocr=any(f["method"].startswith("OCR") for f in subset))
        objects.append(row)
    if not objects:
        warnings.append("Объекты не найдены. Нужна проектная декларация с полями 9.2, 9.3 и 18.1.1.")
    if not header.get("date_iso"):
        warnings.append("Дата декларации не распознана; документ не заменяет датированную версию.")
    return dict(parser_version=VERSION, filename=filename, header=header, developer=developer,
                inn=inn, objects=objects, fields=fields, warnings=warnings,
                pages=total_pages, ocr_pages=ocr_pages, elapsed=round(time.monotonic()-started, 2))
