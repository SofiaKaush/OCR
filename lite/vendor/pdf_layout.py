"""
PDF layout and text-extraction pipeline.
===========================================

Гибридный layout-aware конвейер для проектных деклараций по 214-ФЗ.

Ключевые свойства:
- извлекает ВСЕ пронумерованные поля официального шаблона, а не только 15–20 regex-полей;
- разделяет подпись поля и значение по геометрии таблицы и начертанию шрифта;
- сохраняет доказательство каждого значения: страница, код пункта, bbox, способ и confidence;
- корректно обрабатывает декларации с несколькими объектами / корпусами;
- извлекает большие таблицы квартир, нежилых помещений, общего имущества и оборудования;
- строит объектный и проектный набор признаков для RLV / ML-калибровки;
- выполняет перекрёстные проверки и формирует отчёт качества;
- поддерживает PDF, папку и ZIP, кеширование и изоляцию ошибок.

Основная зависимость: PyMuPDF >= 1.24 (fitz). Для OCR требуется системный Tesseract.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import statistics
import tempfile
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Optional, Sequence

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .economics import (
    ConstructionCostConfig,
    aggregate_finish,
    classify_finish_texts,
    derive_construction_cost,
    derive_saleable_components,
)

try:
    import orjson
except ImportError:  # pragma: no cover - обычный json полностью поддерживается
    orjson = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


PARSER_VERSION = "3.3.0-mvp"
LOGGER = logging.getLogger("pdf_layout")

# Код ячейки официальной формы: 1.1.2, 19.7.3.1.2.3 и т.п.
FIELD_CODE_RE = re.compile(r"^\d{1,2}(?:\.\d{1,3}){1,8}$")
OBJECT_MARKER_RE = re.compile(r"Объект\s*№\s*(\d+)", re.IGNORECASE)
DATE_RE = re.compile(r"(?<!\d)(\d{2}\.\d{2}\.\d{4})(?!\d)")
CADASTRAL_RE = re.compile(r"(?<!\d)(\d{2}:\d{2}:\d{6,7}:\d+)(?!\d)")
NUMBER_RE = re.compile(r"[-+]?\d[\d\s\u00a0\u202f]*(?:[.,]\d+)?")

TABLE_SECTION_BY_CODE: dict[str, str] = {
    "15.2.1": "apartments",
    "15.3.1": "non_residential",
    "16.1.1": "common_property",
    "16.2.1": "engineering_equipment",
}


# -----------------------------------------------------------------------------
# Нормализация и типизация русских чисел / единиц
# -----------------------------------------------------------------------------

def _strip_invisible(text: str) -> str:
    return (
        text.replace("\xad", "")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )


def normalize_text(text: Any) -> str:
    """Нормализует пробелы, мягкие переносы и частый артефакт ``Санкт- Петербург``."""
    if text is None:
        return ""
    value = _strip_invisible(str(text))
    value = re.sub(r"\s+", " ", value).strip()
    # Здесь соединяем только пробел после дефиса. Удаление самого дефиса при переносе
    # выполняется в join_layout_lines, где известна граница строки.
    value = re.sub(r"-\s+(?=[А-ЯA-ZЁ])", "-", value)
    return value


def normalize_table_cell(value: Any) -> str:
    """Нормализует текст ячейки, сохраняя информацию о переносах строк.

    PyMuPDF возвращает содержимое ячейки с ``\n``. Большинство дефисных
    переносов являются разрывами слова (``строитель-\nства``), однако
    ``Квартира-студия`` — устойчивый термин, где дефис должен сохраниться.
    """
    if value is None:
        return ""
    raw = _strip_invisible(str(value))
    normalized = join_layout_lines(raw.splitlines())
    # PyMuPDF часто разбивает этот термин как ``Квартира-\nстудия``.
    normalized = re.sub(
        r"\bквартира[\s-]*студия\b",
        "Квартира-студия",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized


def join_layout_lines(lines: Sequence[str]) -> str:
    """Склеивает строки с учётом переносов внутри слова.

    - ``строитель-`` + ``ства`` -> ``строительства``;
    - ``Санкт-`` + ``Петербург`` -> ``Санкт-Петербург``;
    - обычные строки соединяются пробелом.
    """
    cleaned = [normalize_text(line) for line in lines if normalize_text(line)]
    if not cleaned:
        return ""
    out = cleaned[0]
    for nxt in cleaned[1:]:
        if out.endswith("-") and nxt:
            first = nxt[0]
            if first.islower():
                out = out[:-1] + nxt
            else:
                out = out + nxt
        else:
            out += " " + nxt
    return normalize_text(out)


def parse_ru_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = normalize_text(value)
    match = NUMBER_RE.search(text)
    if not match:
        return None
    token = match.group(0).replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
    token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def parse_int(value: Any) -> Optional[int]:
    number = parse_ru_number(value)
    if number is None:
        return None
    if abs(number - round(number)) > 1e-9:
        return None
    return int(round(number))


def parse_money(value: Any) -> Optional[float]:
    text = normalize_text(value).lower()
    if not text or "руб" not in text:
        return None
    number = parse_ru_number(text)
    if number is None:
        return None
    multiplier = 1.0
    if re.search(r"\bтыс\.?\s*руб", text):
        multiplier = 1_000.0
    elif re.search(r"\bмлн\.?\s*руб", text):
        multiplier = 1_000_000.0
    elif re.search(r"\bмлрд\.?\s*руб", text):
        multiplier = 1_000_000_000.0
    return number * multiplier


def parse_area(value: Any) -> Optional[float]:
    text = normalize_text(value).lower()
    if not text:
        return None
    if not re.search(r"(?:м\s*[²2]|кв\.?\s*м)", text):
        return None
    return parse_ru_number(text)


def parse_percent(value: Any) -> Optional[float]:
    text = normalize_text(value)
    if "%" not in text:
        return None
    return parse_ru_number(text)


def parse_date(value: Any) -> Optional[str]:
    text = normalize_text(value)
    match = DATE_RE.search(text)
    return match.group(1) if match else None


def parse_quarter(value: Any) -> Optional[str]:
    text = normalize_text(value)
    match = re.search(r"([1-4])\s*квартал\s*(\d{4})", text, re.IGNORECASE)
    return f"{match.group(1)} квартал {match.group(2)}" if match else None


def parse_typed_value(value: str) -> tuple[Any, Optional[str]]:
    """Возвращает наиболее полезное типизированное значение и единицу."""
    text = normalize_text(value)
    if not text:
        return None, None
    if DATE_RE.search(text):
        return parse_date(text), "date"
    if "руб" in text.lower():
        return parse_money(text), "RUB"
    if "%" in text:
        return parse_percent(text), "%"
    if re.search(r"(?:м\s*[²2]|кв\.?\s*м)", text.lower()):
        return parse_area(text), "m2"
    if re.fullmatch(r"[-+]?\d[\d\s]*(?:[.,]\d+)?", text):
        number = parse_ru_number(text)
        if number is not None and abs(number - round(number)) < 1e-9:
            return int(round(number)), None
        return number, None
    return text, None


def safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        if numerator is None or denominator in (None, 0) or pd.isna(numerator) or pd.isna(denominator):
            return None
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def pct_delta(actual: Any, expected: Any) -> Optional[float]:
    ratio = safe_div((float(actual) - float(expected)) if actual is not None and expected is not None else None, expected)
    return ratio * 100.0 if ratio is not None else None


def months_between(start: Optional[str], end: Optional[str]) -> Optional[float]:
    if not start or not end:
        return None
    try:
        d1 = datetime.strptime(start, "%d.%m.%Y")
        d2 = datetime.strptime(end, "%d.%m.%Y")
    except ValueError:
        return None
    return round((d2 - d1).days / 30.4375, 2)


def json_dumps(data: Any, *, indent: int = 2) -> str:
    if orjson is not None:
        option = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
        if indent:
            option |= orjson.OPT_INDENT_2
        return orjson.dumps(data, option=option, default=str).decode("utf-8")
    return json.dumps(data, ensure_ascii=False, indent=indent, default=str)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -----------------------------------------------------------------------------
# Pydantic-схемы: строгий типизированный контракт результата
# -----------------------------------------------------------------------------

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PipelineConfig(StrictModel):
    output_dir: Path = Path("declaration_output")
    cache_dir: Optional[Path] = Path(".declaration_cache")
    use_cache: bool = True
    parse_tables: bool = True
    store_raw_table_rows: bool = False
    enable_ocr: bool = False
    ocr_language: str = "rus+eng"
    ocr_dpi: int = 300
    ocr_min_text_chars: int = 40
    include_empty_fields: bool = True
    max_workers: int = 1
    base_cost_rub_per_sqm: float = 167_200.0
    deflator: float = 1.0
    quality_area_tolerance_pct: float = 0.5
    quality_count_tolerance: int = 0
    export_html_report: bool = True
    export_csv: bool = True
    export_json: bool = True
    deduct_utility_fees_from_investment_cost: bool = True
    additional_nonconstruction_share_pct: float = 0.0

    @field_validator("max_workers")
    @classmethod
    def validate_workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_workers должен быть >= 1")
        return value

    @field_validator("additional_nonconstruction_share_pct")
    @classmethod
    def validate_nonconstruction_share(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 45.0:
            raise ValueError("additional_nonconstruction_share_pct должен быть в диапазоне 0–45")
        return float(value)


class Evidence(StrictModel):
    page: int
    object_no: Optional[int] = None
    code: Optional[str] = None
    label: Optional[str] = None
    raw_value: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    method: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedField(StrictModel):
    ordinal: int
    page: int
    object_no: Optional[int] = None
    code: str
    occurrence: int
    label: str
    value_text: str
    typed_value: Any = None
    unit: Optional[str] = None
    method: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]

    def evidence(self) -> Evidence:
        return Evidence(
            page=self.page,
            object_no=self.object_no,
            code=self.code,
            label=self.label,
            raw_value=self.value_text,
            bbox=self.bbox,
            method=self.method,
            confidence=self.confidence,
        )


class HeaderInfo(StrictModel):
    declaration_number: Optional[str] = None
    declaration_date: Optional[str] = None
    first_publication_date: Optional[str] = None
    title: Optional[str] = None
    address: Optional[str] = None
    cadastral_number: Optional[str] = None


class DeveloperInfo(StrictModel):
    legal_form: Optional[str] = None
    full_name: Optional[str] = None
    short_name: Optional[str] = None
    inn: Optional[str] = None
    ogrn: Optional[str] = None
    registration_date: Optional[str] = None
    legal_address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    ceo_full_name: Optional[str] = None
    ceo_position: Optional[str] = None
    reporting_date: Optional[str] = None
    net_profit_rub: Optional[float] = None
    accounts_payable_rub: Optional[float] = None
    accounts_receivable_rub: Optional[float] = None
    paid_up_capital_rub: Optional[float] = None


class ObjectSummary(StrictModel):
    object_no: int
    name: Optional[str] = None
    object_type: Optional[str] = None
    address: Optional[str] = None
    region: Optional[str] = None
    settlement: Optional[str] = None
    purpose: Optional[str] = None
    residential_complex_name: Optional[str] = None
    min_floors: Optional[int] = None
    max_floors: Optional[int] = None
    gross_area_sqm: Optional[float] = None
    residential_area_sqm: Optional[float] = None
    non_residential_area_sqm: Optional[float] = None
    saleable_area_sqm: Optional[float] = None
    gross_construction_area_sqm: Optional[float] = None
    saleable_housing_area_sqm: Optional[float] = None
    saleable_commercial_area_sqm: Optional[float] = None
    saleable_storage_area_sqm: Optional[float] = None
    saleable_parking_area_sqm: Optional[float] = None
    saleable_other_nonres_area_sqm: Optional[float] = None
    saleable_nonresidential_area_sqm: Optional[float] = None
    saleable_total_area_sqm: Optional[float] = None
    official_saleable_area_sqm: Optional[float] = None
    saleable_area_for_cost_sqm: Optional[float] = None
    saleable_area_for_cost_source: Optional[str] = None
    saleable_area_for_cost_confidence: Optional[float] = None
    nonresidential_component_coverage_pct: Optional[float] = None
    gross_to_saleable_ratio: Optional[float] = None
    official_gross_to_saleable_ratio: Optional[float] = None
    official_saleable_efficiency_pct: Optional[float] = None
    non_saleable_area_sqm: Optional[float] = None
    saleable_components_source: Optional[str] = None
    saleable_components_confidence: Optional[float] = None
    finish_type: Optional[str] = None
    finish_confidence: Optional[float] = None
    finish_evidence: Optional[str] = None
    finish_source: Optional[str] = None
    wall_material: Optional[str] = None
    floor_material: Optional[str] = None
    energy_class: Optional[str] = None
    passenger_lifts: Optional[int] = None
    freight_lifts: Optional[int] = None
    passenger_freight_lifts: Optional[int] = None
    accessibility_lifts: Optional[int] = None
    apartment_count_declared: Optional[int] = None
    apartment_count_parsed: Optional[int] = None
    non_residential_count_declared: Optional[int] = None
    non_residential_count_parsed: Optional[int] = None
    parsed_apartment_area_sqm: Optional[float] = None
    parsed_apartment_living_area_sqm: Optional[float] = None
    parsed_non_residential_area_sqm: Optional[float] = None
    parsed_apartment_area_coverage_pct: Optional[float] = None
    parsed_non_residential_area_coverage_pct: Optional[float] = None
    common_property_count_parsed: Optional[int] = None
    common_property_area_sqm: Optional[float] = None
    common_property_share_pct: Optional[float] = None
    engineering_equipment_count_parsed: Optional[int] = None
    parking_spaces_declared: Optional[int] = None
    parking_spaces_per_100_apartments: Optional[float] = None
    lifts_per_100_apartments: Optional[float] = None
    median_ceiling_height_m: Optional[float] = None
    average_apartment_area_sqm: Optional[float] = None
    median_apartment_area_sqm: Optional[float] = None
    average_non_residential_unit_area_sqm: Optional[float] = None
    gross_area_per_apartment_sqm: Optional[float] = None
    saleable_area_per_apartment_sqm: Optional[float] = None
    studio_count: Optional[int] = None
    one_room_count: Optional[int] = None
    two_room_count: Optional[int] = None
    three_room_count: Optional[int] = None
    four_plus_room_count: Optional[int] = None
    construction_permit_number: Optional[str] = None
    construction_permit_date: Optional[str] = None
    permit_valid_until: Optional[str] = None
    planned_transfer_date: Optional[str] = None
    planned_commissioning_period: Optional[str] = None
    planned_construction_cost_rub: Optional[float] = None
    investment_cost_rub: Optional[float] = None
    investment_cost_per_gross_sqm: Optional[float] = None
    investment_cost_per_saleable_sqm: Optional[float] = None
    investment_cost_per_official_saleable_sqm: Optional[float] = None
    disclosed_nonconstruction_cost_rub: Optional[float] = None
    additional_nonconstruction_cost_rub: Optional[float] = None
    construction_cost_rub: Optional[float] = None
    construction_cost_per_gross_sqm: Optional[float] = None
    construction_cost_per_saleable_sqm: Optional[float] = None
    construction_cost_per_official_saleable_sqm: Optional[float] = None
    construction_share_of_investment_pct: Optional[float] = None
    investment_to_construction_delta_rub: Optional[float] = None
    construction_cost_bridge_method: Optional[str] = None
    construction_cost_confidence: Optional[float] = None
    construction_cost_confidence_label: Optional[str] = None
    construction_cost_components_json: Optional[str] = None
    utility_connection_fees_rub: Optional[float] = None
    utility_fee_per_gross_sqm: Optional[float] = None
    loan_amount_rub: Optional[float] = None
    loan_debt_rub: Optional[float] = None
    loan_unused_rub: Optional[float] = None
    loan_utilization_pct: Optional[float] = None
    loan_to_cost_pct: Optional[float] = None
    loan_debt_to_cost_pct: Optional[float] = None
    sold_apartment_contracts: Optional[int] = None
    sold_nonres_contracts: Optional[int] = None
    sold_apartment_area_sqm: Optional[float] = None
    sold_nonres_area_sqm: Optional[float] = None
    sold_apartment_revenue_rub: Optional[float] = None
    sold_nonres_revenue_rub: Optional[float] = None
    total_sold_revenue_rub: Optional[float] = None
    avg_apartment_sale_price_rub_per_sqm: Optional[float] = None
    avg_nonres_sale_price_rub_per_sqm: Optional[float] = None
    estimated_unsold_apartment_units: Optional[int] = None
    estimated_unsold_apartment_area_sqm: Optional[float] = None
    sold_apartment_unit_share_pct: Optional[float] = None
    sold_apartment_area_share_pct: Optional[float] = None
    floor_zone: Optional[str] = None
    k_fl: Optional[float] = None
    class_proxy: Optional[str] = None
    class_proxy_confidence: Optional[float] = None
    k_cl: Optional[float] = None
    theoretical_cost_rub_per_sqm: Optional[float] = None
    cost_gap_to_parametric_pct: Optional[float] = None
    saleable_efficiency_pct: Optional[float] = None
    construction_duration_months: Optional[float] = None


class ProjectSummary(StrictModel):
    file_name: str
    sha256: str
    declaration_number: Optional[str] = None
    declaration_date: Optional[str] = None
    first_publication_date: Optional[str] = None
    title: Optional[str] = None
    address: Optional[str] = None
    cadastral_number: Optional[str] = None
    residential_complex_name: Optional[str] = None
    developer_name: Optional[str] = None
    developer_inn: Optional[str] = None
    object_count_declared: Optional[int] = None
    object_count_parsed: int = 0
    gross_area_sqm: Optional[float] = None
    residential_area_sqm: Optional[float] = None
    non_residential_area_sqm: Optional[float] = None
    saleable_area_sqm: Optional[float] = None
    gross_construction_area_sqm: Optional[float] = None
    saleable_housing_area_sqm: Optional[float] = None
    saleable_commercial_area_sqm: Optional[float] = None
    saleable_storage_area_sqm: Optional[float] = None
    saleable_parking_area_sqm: Optional[float] = None
    saleable_other_nonres_area_sqm: Optional[float] = None
    saleable_nonresidential_area_sqm: Optional[float] = None
    saleable_total_area_sqm: Optional[float] = None
    official_saleable_area_sqm: Optional[float] = None
    saleable_area_for_cost_sqm: Optional[float] = None
    saleable_area_for_cost_source: Optional[str] = None
    saleable_area_for_cost_confidence: Optional[float] = None
    nonresidential_component_coverage_pct: Optional[float] = None
    gross_to_saleable_ratio: Optional[float] = None
    official_gross_to_saleable_ratio: Optional[float] = None
    official_saleable_efficiency_pct: Optional[float] = None
    non_saleable_area_sqm: Optional[float] = None
    saleable_components_source: Optional[str] = None
    saleable_components_confidence: Optional[float] = None
    finish_type: Optional[str] = None
    finish_confidence: Optional[float] = None
    finish_evidence: Optional[str] = None
    finish_source: Optional[str] = None
    land_area_sqm: Optional[float] = None
    floor_area_ratio: Optional[float] = None
    saleable_efficiency_pct: Optional[float] = None
    apartments_declared: Optional[int] = None
    apartments_parsed: Optional[int] = None
    non_residential_units_declared: Optional[int] = None
    non_residential_units_parsed: Optional[int] = None
    parsed_apartment_area_sqm: Optional[float] = None
    parsed_apartment_living_area_sqm: Optional[float] = None
    parsed_non_residential_area_sqm: Optional[float] = None
    parsed_apartment_area_coverage_pct: Optional[float] = None
    parsed_non_residential_area_coverage_pct: Optional[float] = None
    common_property_count_parsed: Optional[int] = None
    common_property_area_sqm: Optional[float] = None
    common_property_share_pct: Optional[float] = None
    engineering_equipment_count_parsed: Optional[int] = None
    passenger_lifts: Optional[int] = None
    parking_spaces_declared: Optional[int] = None
    lifts_per_100_apartments: Optional[float] = None
    parking_spaces_per_100_apartments: Optional[float] = None
    max_floors: Optional[int] = None
    median_ceiling_height_m: Optional[float] = None
    average_apartment_area_sqm: Optional[float] = None
    average_non_residential_unit_area_sqm: Optional[float] = None
    gross_area_per_apartment_sqm: Optional[float] = None
    saleable_area_per_apartment_sqm: Optional[float] = None
    studio_count: Optional[int] = None
    one_room_count: Optional[int] = None
    two_room_count: Optional[int] = None
    three_room_count: Optional[int] = None
    four_plus_room_count: Optional[int] = None
    planned_construction_cost_rub: Optional[float] = None
    investment_cost_rub: Optional[float] = None
    investment_cost_per_gross_sqm: Optional[float] = None
    investment_cost_per_saleable_sqm: Optional[float] = None
    investment_cost_per_official_saleable_sqm: Optional[float] = None
    disclosed_nonconstruction_cost_rub: Optional[float] = None
    additional_nonconstruction_cost_rub: Optional[float] = None
    construction_cost_rub: Optional[float] = None
    planned_cost_per_apartment_rub: Optional[float] = None
    construction_cost_per_gross_sqm: Optional[float] = None
    construction_cost_per_saleable_sqm: Optional[float] = None
    construction_cost_per_official_saleable_sqm: Optional[float] = None
    construction_share_of_investment_pct: Optional[float] = None
    investment_to_construction_delta_rub: Optional[float] = None
    construction_cost_bridge_method: Optional[str] = None
    construction_cost_confidence: Optional[float] = None
    construction_cost_confidence_label: Optional[str] = None
    construction_cost_components_json: Optional[str] = None
    parametric_cost_rub_per_saleable_sqm: Optional[float] = None
    parametric_total_cost_rub: Optional[float] = None
    cost_gap_to_parametric_pct: Optional[float] = None
    utility_connection_fees_rub: Optional[float] = None
    utility_fee_per_gross_sqm: Optional[float] = None
    total_loan_amount_rub: Optional[float] = None
    total_loan_debt_rub: Optional[float] = None
    total_loan_unused_rub: Optional[float] = None
    loan_utilization_pct: Optional[float] = None
    loan_to_cost_pct: Optional[float] = None
    loan_debt_to_cost_pct: Optional[float] = None
    loan_unused_to_cost_pct: Optional[float] = None
    sold_apartment_contracts: Optional[int] = None
    sold_nonres_contracts: Optional[int] = None
    sold_apartment_area_sqm: Optional[float] = None
    sold_nonres_area_sqm: Optional[float] = None
    sold_apartment_revenue_rub: Optional[float] = None
    sold_nonres_revenue_rub: Optional[float] = None
    total_sold_revenue_rub: Optional[float] = None
    total_sold_area_sqm: Optional[float] = None
    avg_apartment_sale_price_rub_per_sqm: Optional[float] = None
    avg_nonres_sale_price_rub_per_sqm: Optional[float] = None
    blended_avg_sale_price_rub_per_sqm: Optional[float] = None
    estimated_unsold_apartment_units: Optional[int] = None
    estimated_unsold_apartment_area_sqm: Optional[float] = None
    sold_apartment_unit_share_pct: Optional[float] = None
    sold_apartment_area_share_pct: Optional[float] = None
    developer_net_profit_to_cost_pct: Optional[float] = None
    developer_payables_to_cost_pct: Optional[float] = None
    developer_receivables_to_cost_pct: Optional[float] = None
    land_area_per_saleable_sqm: Optional[float] = None
    raw_field_count: int = 0
    filled_field_count: int = 0
    parsed_table_row_count: int = 0
    quality_score: Optional[float] = None


class QualityCheck(StrictModel):
    scope: str
    check: str
    status: Literal["PASS", "WARN", "FAIL", "INFO"]
    actual: Any = None
    expected: Any = None
    delta: Optional[float] = None
    message: str
    evidence_pages: list[int] = Field(default_factory=list)


class Diagnostics(StrictModel):
    parser_version: str = PARSER_VERSION
    file_name: str
    sha256: str
    page_count: int
    text_pages: int
    ocr_pages: int
    scanned_page_candidates: int
    raw_field_count: int
    filled_field_count: int
    raw_table_row_count: int
    apartment_rows: int
    non_residential_rows: int
    common_property_rows: int
    engineering_equipment_rows: int
    elapsed_seconds: float
    cache_hit: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Внутренние layout-структуры
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class Span:
    text: str
    bbox: tuple[float, float, float, float]
    font: str
    size: float
    flags: int

    @property
    def bold(self) -> bool:
        return "bold" in self.font.lower() or bool(self.flags & 16)

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass(slots=True)
class InputPdf:
    file_name: str
    pdf_bytes: bytes
    source: str


# -----------------------------------------------------------------------------
# Файловые источники: PDF / папка / ZIP
# -----------------------------------------------------------------------------

def iter_input_pdfs(input_path: str | Path, *, recursive: bool = True) -> Iterator[InputPdf]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Не найден входной путь: {path}")

    if path.is_file() and path.suffix.lower() == ".pdf":
        yield InputPdf(path.name, path.read_bytes(), str(path))
        return

    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            for member in sorted(archive.namelist()):
                if member.lower().endswith(".pdf") and not member.endswith("/"):
                    yield InputPdf(Path(member).name, archive.read(member), f"{path}!{member}")
        return

    if path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        for pdf_path in sorted(path.glob(pattern)):
            if pdf_path.is_file():
                yield InputPdf(pdf_path.name, pdf_path.read_bytes(), str(pdf_path))
        return

    raise ValueError(f"Поддерживаются PDF, ZIP или папка с PDF: {path}")


# -----------------------------------------------------------------------------
# Layout-aware извлечение
# -----------------------------------------------------------------------------

class LayoutDeclarationExtractor:
    """Извлекает поля и таблицы за один проход по страницам."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._tesseract_available = bool(
            self.config.enable_ocr
            and shutil.which("tesseract")
            and (pytesseract is not None or hasattr(fitz.Page, "get_textpage_ocr"))
        )

    @staticmethod
    def _spans_from_page(page: fitz.Page, *, textpage: Any = None) -> list[Span]:
        data = page.get_text("dict", textpage=textpage, sort=True)
        spans: list[Span] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for raw in line.get("spans", []):
                    text = normalize_text(raw.get("text", ""))
                    if not text:
                        continue
                    spans.append(
                        Span(
                            text=text,
                            bbox=tuple(float(v) for v in raw["bbox"]),
                            font=str(raw.get("font", "")),
                            size=float(raw.get("size", 0.0)),
                            flags=int(raw.get("flags", 0)),
                        )
                    )
        return spans

    @staticmethod
    def _group_span_lines(spans: Sequence[Span], y_tolerance: float = 2.6) -> list[str]:
        groups: list[list[Any]] = []
        for span in sorted(spans, key=lambda item: (item.center[1], item.bbox[0])):
            cy = span.center[1]
            group = next((candidate for candidate in groups if abs(candidate[0] - cy) <= y_tolerance), None)
            if group is None:
                group = [cy, []]
                groups.append(group)
            group[1].append(span)

        lines: list[str] = []
        for _, items in sorted(groups, key=lambda item: item[0]):
            items = sorted(items, key=lambda item: item.bbox[0])
            lines.append(normalize_text(" ".join(item.text for item in items)))
        return [line for line in lines if line]

    @classmethod
    def _join_spans(cls, spans: Sequence[Span]) -> str:
        return join_layout_lines(cls._group_span_lines(spans))

    @staticmethod
    def _span_in_bbox(span: Span, bbox: tuple[float, float, float, float], margin: float = 0.8) -> bool:
        x, y = span.center
        x0, y0, x1, y1 = bbox
        return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin

    @staticmethod
    def _is_official_code_cell(bbox: tuple[float, float, float, float]) -> bool:
        x0, _, x1, _ = bbox
        return 100.0 <= x0 <= 270.0 and (x1 - x0) <= 100.0

    def _ocr_textpage(self, page: fitz.Page, warnings: list[str]) -> Any:
        if not self._tesseract_available:
            return None
        try:
            return page.get_textpage_ocr(
                language=self.config.ocr_language,
                dpi=self.config.ocr_dpi,
                full=True,
            )
        except Exception as exc:  # pragma: no cover - зависит от локальной установки OCR
            warnings.append(f"OCR страницы {page.number + 1} не выполнен: {exc}")
            return None

    def extract(self, pdf_bytes: bytes, file_name: str) -> dict[str, Any]:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        raw_fields: list[ExtractedField] = []
        raw_table_rows: list[dict[str, Any]] = []
        apartments: list[dict[str, Any]] = []
        non_residential: list[dict[str, Any]] = []
        non_residential_parts: list[dict[str, Any]] = []
        common_property: list[dict[str, Any]] = []
        engineering_equipment: list[dict[str, Any]] = []
        page_texts: list[str] = []
        page_line_lists: list[list[str]] = []
        warnings: list[str] = []

        current_object_no: Optional[int] = None
        table_section: Optional[str] = None
        current_nonres_unit: dict[int, str] = {}
        occurrence_counter: Counter[tuple[Optional[int], str]] = Counter()
        ordinal = 0
        ocr_pages = 0
        scanned_candidates = 0
        text_pages = 0

        for page_index, page in enumerate(document):
            spans = self._spans_from_page(page)
            page_chars = sum(len(span.text) for span in spans)
            if page_chars < self.config.ocr_min_text_chars:
                scanned_candidates += 1
                textpage = self._ocr_textpage(page, warnings)
                if textpage is not None:
                    ocr_spans = self._spans_from_page(page, textpage=textpage)
                    if sum(len(span.text) for span in ocr_spans) > page_chars:
                        spans = ocr_spans
                        ocr_pages += 1
                        page_chars = sum(len(span.text) for span in spans)
            if page_chars >= self.config.ocr_min_text_chars:
                text_pages += 1

            lines = self._group_span_lines(spans)
            page_text = join_layout_lines(lines)
            page_texts.append(page_text)
            page_line_lists.append(lines)

            object_match = OBJECT_MARKER_RE.search(page_text)
            if object_match:
                current_object_no = int(object_match.group(1))
                table_section = None

            if not self.config.parse_tables:
                continue

            try:
                tables = page.find_tables().tables
            except Exception as exc:
                warnings.append(f"Таблицы страницы {page_index + 1} не распознаны: {exc}")
                tables = []

            for table_index, table in enumerate(tables):
                extracted_rows = table.extract()
                for row_index, (row, row_values) in enumerate(zip(table.rows, extracted_rows)):
                    values = [normalize_table_cell(value) if value is not None else None for value in row_values]
                    flattened = [value for value in values if value]
                    if self.config.store_raw_table_rows and flattened:
                        raw_table_rows.append(
                            {
                                "page": page_index + 1,
                                "object_no": current_object_no,
                                "table_index": table_index,
                                "row_index": row_index,
                                "section_state": table_section,
                                "cells": flattened,
                            }
                        )

                    official_codes: list[tuple[int, str, tuple[float, float, float, float]]] = []
                    for cell_index, value in enumerate(values):
                        if not value or not FIELD_CODE_RE.fullmatch(value):
                            continue
                        cell_bbox = row.cells[cell_index] if cell_index < len(row.cells) else None
                        if cell_bbox is None or not self._is_official_code_cell(cell_bbox):
                            continue
                        official_codes.append((cell_index, value, tuple(float(v) for v in cell_bbox)))

                    # Сначала извлекаем официальные поля из строки.
                    for cell_index, code, code_bbox in official_codes:
                        right_candidates: list[tuple[int, tuple[float, float, float, float]]] = []
                        seen_bboxes: set[tuple[float, float, float, float]] = set()
                        for right_index in range(cell_index + 1, len(row.cells)):
                            bbox = row.cells[right_index]
                            if bbox is None:
                                continue
                            bbox_tuple = tuple(float(v) for v in bbox)
                            rounded = tuple(round(v, 2) for v in bbox_tuple)
                            if rounded in seen_bboxes:
                                continue
                            seen_bboxes.add(rounded)
                            if bbox_tuple[0] >= code_bbox[2] - 2.0:
                                right_candidates.append((right_index, bbox_tuple))
                        if not right_candidates:
                            continue

                        right_index, value_bbox = right_candidates[0]
                        cell_spans = [span for span in spans if self._span_in_bbox(span, value_bbox)]
                        label_spans = [span for span in cell_spans if not span.bold]
                        value_spans = [span for span in cell_spans if span.bold]
                        label = self._join_spans(label_spans)
                        value_text = self._join_spans(value_spans)
                        method = "layout_table+bold"
                        confidence = 0.995

                        # Резерв для PDF, где ответ напечатан обычным шрифтом после двоеточия.
                        if not value_text:
                            cell_text = values[right_index] if right_index < len(values) else ""
                            if cell_text and ":" in cell_text:
                                colon = cell_text.rfind(":")
                                tail = normalize_text(cell_text[colon + 1 :])
                                if tail:
                                    label = normalize_text(cell_text[: colon + 1])
                                    value_text = tail
                                    method = "layout_table+after_colon"
                                    confidence = 0.84
                                else:
                                    method = "layout_table+empty"
                                    confidence = 0.99
                            else:
                                method = "layout_table+empty"
                                confidence = 0.99

                        if not self.config.include_empty_fields and not value_text:
                            continue

                        typed_value, unit = parse_typed_value(value_text)
                        ordinal += 1
                        occurrence_counter[(current_object_no, code)] += 1
                        raw_fields.append(
                            ExtractedField(
                                ordinal=ordinal,
                                page=page_index + 1,
                                object_no=current_object_no,
                                code=code,
                                occurrence=occurrence_counter[(current_object_no, code)],
                                label=label,
                                value_text=value_text,
                                typed_value=typed_value,
                                unit=unit,
                                method=method,
                                confidence=confidence,
                                bbox=value_bbox,
                            )
                        )

                    # Затем обновляем состояние большой таблицы.
                    for _, code, _ in official_codes:
                        if code in TABLE_SECTION_BY_CODE:
                            table_section = TABLE_SECTION_BY_CODE[code]
                        elif table_section and code.startswith(("17.", "18.", "19.", "20.", "21.", "22.", "23.")):
                            table_section = None

                    if official_codes or current_object_no is None:
                        continue

                    # Специализированный разбор строк больших таблиц.
                    if table_section == "apartments":
                        record = self._parse_apartment_row(flattened, current_object_no, page_index + 1)
                        if record:
                            apartments.append(record)
                    elif table_section == "non_residential":
                        record, part = self._parse_nonres_row(flattened, current_object_no, page_index + 1)
                        if record:
                            non_residential.append(record)
                            current_nonres_unit[current_object_no] = record["unit_id"]
                        if part:
                            non_residential_parts.append(part)
                        elif not record and current_object_no in current_nonres_unit:
                            continuation = self._parse_nonres_part_continuation(
                                flattened,
                                current_object_no,
                                current_nonres_unit[current_object_no],
                                page_index + 1,
                            )
                            if continuation:
                                non_residential_parts.append(continuation)
                    elif table_section == "common_property":
                        record = self._parse_common_property_row(flattened, current_object_no, page_index + 1)
                        if record:
                            common_property.append(record)
                    elif table_section == "engineering_equipment":
                        record = self._parse_equipment_row(flattened, current_object_no, page_index + 1)
                        if record:
                            engineering_equipment.append(record)

        document.close()
        header = self._parse_header(page_line_lists[0] if page_line_lists else [], page_texts[0] if page_texts else "")
        return {
            "header": header,
            "raw_fields": raw_fields,
            "raw_table_rows": raw_table_rows,
            "apartments": apartments,
            "non_residential": non_residential,
            "non_residential_parts": non_residential_parts,
            "common_property": common_property,
            "engineering_equipment": engineering_equipment,
            "page_texts": page_texts,
            "warnings": warnings,
            "page_count": len(page_texts),
            "text_pages": text_pages,
            "ocr_pages": ocr_pages,
            "scanned_candidates": scanned_candidates,
        }

    @staticmethod
    def _parse_header(lines: Sequence[str], page_text: str) -> HeaderInfo:
        declaration_number = None
        declaration_date = None
        first_publication_date = None
        title_lines: list[str] = []

        declaration_line_index: Optional[int] = None
        publication_line_index: Optional[int] = None
        for index, line in enumerate(lines):
            match = re.search(r"№\s*([^\s]+)\s+от\s+(\d{2}\.\d{2}\.\d{4})", line)
            if match and declaration_number is None:
                declaration_number = match.group(1)
                declaration_date = match.group(2)
                declaration_line_index = index
            if "Дата первичного размещения" in line:
                first_publication_date = parse_date(line)
                publication_line_index = index
                break

        if declaration_line_index is not None and publication_line_index is not None:
            title_lines = list(lines[declaration_line_index + 1 : publication_line_index])
        title = join_layout_lines(title_lines) or None
        address = None
        if title:
            match = re.search(r"по адресу\s*:\s*(.+)$", title, re.IGNORECASE)
            if match:
                address = normalize_text(match.group(1))
        cadastral_match = CADASTRAL_RE.search(title or page_text)
        cadastral_number = cadastral_match.group(1) if cadastral_match else None
        return HeaderInfo(
            declaration_number=declaration_number,
            declaration_date=declaration_date,
            first_publication_date=first_publication_date,
            title=title,
            address=address,
            cadastral_number=cadastral_number,
        )

    @staticmethod
    def _parse_apartment_row(cells: Sequence[str], object_no: int, page: int) -> Optional[dict[str, Any]]:
        def canonical_unit_type(cell: Any) -> str:
            text = normalize_table_cell(cell)
            compact = re.sub(r"[\s-]+", "", text).lower()
            if compact == "квартирастудия":
                return "Квартира-студия"
            if compact == "квартира":
                return "Квартира"
            return text

        canonical_cells = [canonical_unit_type(cell) for cell in cells]
        apartment_index = next(
            (
                index
                for index, cell in enumerate(canonical_cells)
                if cell in {"Квартира", "Квартира-студия"}
            ),
            None,
        )
        if apartment_index is None or apartment_index < 1 or len(cells) < apartment_index + 7:
            return None
        unit_id = normalize_text(cells[apartment_index - 1])
        if not re.fullmatch(r"\d+(?:\.\d+){0,4}", unit_id):
            return None
        values = cells[apartment_index + 1 : apartment_index + 7]
        floor = parse_int(values[0])
        entrance = parse_int(values[1])
        area = parse_ru_number(values[2])
        rooms = parse_int(values[3])
        living_area = parse_ru_number(values[4])
        ceiling = parse_ru_number(values[5])
        if None in (floor, entrance, area, rooms, living_area, ceiling):
            return None
        return {
            "object_no": object_no,
            "page": page,
            "unit_id": unit_id,
            "unit_type": canonical_cells[apartment_index],
            "floor": floor,
            "entrance": entrance,
            "area_sqm": area,
            "rooms": rooms,
            "living_area_sqm": living_area,
            "ceiling_height_m": ceiling,
        }

    @staticmethod
    def _parse_nonres_row(
        cells: Sequence[str], object_no: int, page: int
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        if len(cells) < 5 or not re.fullmatch(r"\d+(?:\.\d+){0,4}", cells[0]):
            return None, None
        floor = parse_int(cells[2]) if len(cells) > 2 else None
        entrance = parse_int(cells[3]) if len(cells) > 3 else None
        area = parse_ru_number(cells[4]) if len(cells) > 4 else None
        if floor is None or area is None:
            return None, None
        record = {
            "object_no": object_no,
            "page": page,
            "unit_id": cells[0],
            "purpose": cells[1] if len(cells) > 1 else None,
            "floor": floor,
            "entrance": entrance,
            "area_sqm": area,
            "ceiling_height_m": None,
        }
        part = None
        if len(cells) >= 8:
            record["ceiling_height_m"] = parse_ru_number(cells[-1])
            part_area = parse_ru_number(cells[6])
            if part_area is not None:
                part = {
                    "object_no": object_no,
                    "unit_id": cells[0],
                    "page": page,
                    "part_name": cells[5],
                    "part_area_sqm": part_area,
                }
        elif len(cells) >= 7:
            part_area = parse_ru_number(cells[6])
            if part_area is not None:
                part = {
                    "object_no": object_no,
                    "unit_id": cells[0],
                    "page": page,
                    "part_name": cells[5],
                    "part_area_sqm": part_area,
                }
        return record, part

    @staticmethod
    def _parse_nonres_part_continuation(
        cells: Sequence[str], object_no: int, unit_id: str, page: int
    ) -> Optional[dict[str, Any]]:
        if len(cells) < 2:
            return None
        area = parse_ru_number(cells[-1])
        if area is None:
            return None
        name = normalize_text(" ".join(cells[:-1]))
        if not name:
            return None
        return {
            "object_no": object_no,
            "unit_id": unit_id,
            "page": page,
            "part_name": name,
            "part_area_sqm": area,
        }

    @staticmethod
    def _parse_common_property_row(cells: Sequence[str], object_no: int, page: int) -> Optional[dict[str, Any]]:
        if len(cells) < 5 or not re.fullmatch(r"\d+", cells[0]):
            return None
        area = parse_ru_number(cells[4])
        if area is None:
            return None
        return {
            "object_no": object_no,
            "page": page,
            "item_no": int(cells[0]),
            "room_type": cells[1],
            "location": cells[2],
            "purpose": cells[3],
            "area_sqm": area,
        }

    @staticmethod
    def _parse_equipment_row(cells: Sequence[str], object_no: int, page: int) -> Optional[dict[str, Any]]:
        if len(cells) < 4 or not re.fullmatch(r"\d+", cells[0]):
            return None
        return {
            "object_no": object_no,
            "page": page,
            "item_no": int(cells[0]),
            "location": cells[1],
            "equipment": cells[2],
            "purpose": cells[3],
        }


# -----------------------------------------------------------------------------
# Индекс полей и групп повторяющихся секций
# -----------------------------------------------------------------------------

_ANY = object()


class FieldIndex:
    def __init__(self, fields: Sequence[ExtractedField]):
        self.fields = sorted(fields, key=lambda item: item.ordinal)
        self.by_code: dict[str, list[ExtractedField]] = defaultdict(list)
        for field in self.fields:
            self.by_code[field.code].append(field)

    def find(
        self,
        code: str,
        *,
        object_no: Any = _ANY,
        label_contains: Optional[str | Sequence[str]] = None,
        nonempty: bool = True,
    ) -> list[ExtractedField]:
        candidates = self.by_code.get(code, [])
        if object_no is not _ANY:
            candidates = [field for field in candidates if field.object_no == object_no]
        if label_contains:
            needles = [label_contains] if isinstance(label_contains, str) else list(label_contains)
            candidates = [
                field
                for field in candidates
                if all(normalize_text(needle).lower() in field.label.lower() for needle in needles)
            ]
        if nonempty:
            candidates = [field for field in candidates if field.value_text]
        return candidates

    def first_field(self, code: str, **kwargs: Any) -> Optional[ExtractedField]:
        values = self.find(code, **kwargs)
        return values[0] if values else None

    def first_text(self, code: str, **kwargs: Any) -> Optional[str]:
        field = self.first_field(code, **kwargs)
        return field.value_text if field else None

    def first_int(self, code: str, **kwargs: Any) -> Optional[int]:
        return parse_int(self.first_text(code, **kwargs))

    def first_number(self, code: str, **kwargs: Any) -> Optional[float]:
        return parse_ru_number(self.first_text(code, **kwargs))

    def first_money(self, code: str, **kwargs: Any) -> Optional[float]:
        return parse_money(self.first_text(code, **kwargs))

    def first_area(self, code: str, **kwargs: Any) -> Optional[float]:
        return parse_area(self.first_text(code, **kwargs))

    def first_date(self, code: str, **kwargs: Any) -> Optional[str]:
        return parse_date(self.first_text(code, **kwargs))

    def values_text(self, code: str, **kwargs: Any) -> list[str]:
        return [field.value_text for field in self.find(code, **kwargs)]

    def groups(
        self,
        *,
        anchor_code: str,
        prefix: str,
        object_no: Optional[int],
    ) -> list[dict[str, list[ExtractedField]]]:
        """Группирует повторяющиеся блоки: 9.2, 14.1, 17.1, 19.6 и т.д."""
        scoped = [field for field in self.fields if field.object_no == object_no]
        groups: list[dict[str, list[ExtractedField]]] = []
        current: Optional[dict[str, list[ExtractedField]]] = None
        for field in scoped:
            if field.code == anchor_code:
                current = defaultdict(list)
                groups.append(current)
            if current is None:
                continue
            if field.code.startswith(prefix):
                current[field.code].append(field)
            elif field.code != anchor_code:
                current = None
        return [dict(group) for group in groups]

    @staticmethod
    def group_text(group: Mapping[str, Sequence[ExtractedField]], code: str) -> Optional[str]:
        values = group.get(code, [])
        return values[0].value_text if values and values[0].value_text else None

    @classmethod
    def group_int(cls, group: Mapping[str, Sequence[ExtractedField]], code: str) -> Optional[int]:
        return parse_int(cls.group_text(group, code))

    @classmethod
    def group_area(cls, group: Mapping[str, Sequence[ExtractedField]], code: str) -> Optional[float]:
        return parse_area(cls.group_text(group, code))

    @classmethod
    def group_money(cls, group: Mapping[str, Sequence[ExtractedField]], code: str) -> Optional[float]:
        return parse_money(cls.group_text(group, code))


# -----------------------------------------------------------------------------
# Извлечение бизнес-сущностей и признаков
# -----------------------------------------------------------------------------

FLOOR_BANDS: list[tuple[int, int, str, float]] = [
    (1, 12, "Среднеэтажная (1–12)", 1.00),
    (13, 18, "Многоэтажная (13–18)", 1.10),
    (19, 25, "Высотная (19–25)", 1.25),
    (26, 30, "Доминанта (26–30)", 1.33),
]
CLASS_BANDS: list[tuple[float, str, float]] = [
    (3.00, "Бизнес-класс (proxy)", 1.25),
    (2.75, "Комфорт-класс (proxy)", 1.09),
    (0.00, "Стандарт (proxy)", 1.00),
]


def floor_band(max_floors: Optional[int]) -> tuple[Optional[str], Optional[float]]:
    if max_floors is None:
        return None, None
    for low, high, label, coefficient in FLOOR_BANDS:
        if low <= max_floors <= high:
            return label, coefficient
    if max_floors > FLOOR_BANDS[-1][1]:
        return FLOOR_BANDS[-1][2] + ", экстраполяция", FLOOR_BANDS[-1][3]
    return FLOOR_BANDS[0][2], FLOOR_BANDS[0][3]


def class_proxy(ceiling_height: Optional[float], project_text: str = "") -> tuple[Optional[str], Optional[float], Optional[float]]:
    """Proxy, а не юридически достоверный класс проекта.

    Сначала учитываются явные слова в названии, затем высота потолка. Confidence намеренно
    ограничен: проектная декларация обычно не содержит официального поля «класс жилья».
    """
    text = normalize_text(project_text).lower()
    explicit = [
        (("премиум", "deluxe", "элит"), "Премиум/элитный (по названию)", 1.35, 0.82),
        (("бизнес",), "Бизнес-класс (по названию)", 1.25, 0.82),
        (("комфорт",), "Комфорт-класс (по названию)", 1.09, 0.78),
        (("стандарт", "эконом"), "Стандарт (по названию)", 1.00, 0.75),
    ]
    for keywords, label, coefficient, confidence in explicit:
        if any(keyword in text for keyword in keywords):
            return label, coefficient, confidence
    if ceiling_height is None:
        return None, None, None
    for minimum, label, coefficient in CLASS_BANDS:
        if ceiling_height >= minimum:
            return label, coefficient, 0.56
    return None, None, None


class BusinessEntityBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def build(
        self,
        *,
        file_name: str,
        sha256: str,
        header: HeaderInfo,
        fields: Sequence[ExtractedField],
        apartments: Sequence[Mapping[str, Any]],
        non_residential: Sequence[Mapping[str, Any]],
        common_property: Sequence[Mapping[str, Any]],
        engineering_equipment: Sequence[Mapping[str, Any]],
        raw_table_rows: Sequence[Mapping[str, Any]],
    ) -> tuple[DeveloperInfo, list[ObjectSummary], ProjectSummary, dict[str, Any]]:
        index = FieldIndex(fields)
        developer = self._build_developer(index)
        objects, collections = self._build_objects(
            index, apartments, non_residential, common_property, engineering_equipment, header
        )
        project = self._build_project(
            file_name=file_name,
            sha256=sha256,
            header=header,
            developer=developer,
            index=index,
            objects=objects,
            fields=fields,
            raw_table_rows=raw_table_rows,
        )
        return developer, objects, project, collections

    @staticmethod
    def _build_developer(index: FieldIndex) -> DeveloperInfo:
        address_parts = [
            index.first_text(code, object_no=None)
            for code in ("1.2.1", "1.2.2", "1.2.3", "1.2.4", "1.2.5", "1.2.6", "1.2.7", "1.2.8", "1.2.9", "1.2.10")
        ]
        address = ", ".join(part for part in address_parts if part) or None
        ceo = " ".join(
            part
            for part in (
                index.first_text("1.5.1", object_no=None),
                index.first_text("1.5.2", object_no=None),
                index.first_text("1.5.3", object_no=None),
            )
            if part
        ) or None
        return DeveloperInfo(
            legal_form=index.first_text("1.1.1", object_no=None),
            full_name=index.first_text("1.1.2", object_no=None),
            short_name=index.first_text("1.1.3", object_no=None),
            inn=index.first_text("2.1.1", object_no=None),
            ogrn=index.first_text("2.1.2", object_no=None),
            registration_date=index.first_date("2.1.3", object_no=None),
            legal_address=address,
            phone=index.first_text("1.4.1", object_no=None),
            email=index.first_text("1.4.2", object_no=None),
            website=index.first_text("1.4.3", object_no=None),
            ceo_full_name=ceo,
            ceo_position=index.first_text("1.5.4", object_no=None),
            reporting_date=index.first_date("6.1.1", object_no=None),
            net_profit_rub=index.first_money("6.1.2", object_no=None),
            accounts_payable_rub=index.first_money("6.1.3", object_no=None),
            accounts_receivable_rub=index.first_money("6.1.4", object_no=None),
            paid_up_capital_rub=index.first_money("21.1.1", object_no=_ANY),
        )

    def _build_objects(
        self,
        index: FieldIndex,
        apartments: Sequence[Mapping[str, Any]],
        non_residential: Sequence[Mapping[str, Any]],
        common_property: Sequence[Mapping[str, Any]],
        engineering_equipment: Sequence[Mapping[str, Any]],
        header: HeaderInfo,
    ) -> tuple[list[ObjectSummary], dict[str, Any]]:
        overview_groups = index.groups(anchor_code="9.2.1", prefix="9.2.", object_no=None)
        area_groups = index.groups(anchor_code="9.3.1", prefix="9.3.", object_no=None)
        lift_groups = index.groups(anchor_code="9.4.1", prefix="9.4.", object_no=None)
        parsed_object_numbers = sorted({field.object_no for field in index.fields if field.object_no is not None})
        object_count = max(len(overview_groups), len(parsed_object_numbers))

        collections: dict[str, Any] = {
            "beneficial_owners": self._extract_beneficial_owners(index),
            "group_companies": self._extract_group_companies(index),
            "objects": {},
        }
        objects: list[ObjectSummary] = []

        for object_no in range(1, object_count + 1):
            overview = overview_groups[object_no - 1] if object_no <= len(overview_groups) else {}
            area_group = area_groups[object_no - 1] if object_no <= len(area_groups) else {}
            lift_group = lift_groups[object_no - 1] if object_no <= len(lift_groups) else {}
            object_apartments = [row for row in apartments if row.get("object_no") == object_no]
            object_nonres = [row for row in non_residential if row.get("object_no") == object_no]
            object_common = [row for row in common_property if row.get("object_no") == object_no]
            object_equipment = [row for row in engineering_equipment if row.get("object_no") == object_no]

            declared_apartments = index.first_int("15.1.1", object_no=object_no)
            declared_nonres = index.first_int("15.1.2", object_no=object_no)
            apartment_areas = [float(row["area_sqm"]) for row in object_apartments if row.get("area_sqm") is not None]
            apartment_living_areas = [float(row["living_area_sqm"]) for row in object_apartments if row.get("living_area_sqm") is not None]
            parsed_nonres_areas = [float(row["area_sqm"]) for row in object_nonres if row.get("area_sqm") is not None]
            common_areas = [float(row["area_sqm"]) for row in object_common if row.get("area_sqm") is not None]
            ceilings = [float(row["ceiling_height_m"]) for row in object_apartments if row.get("ceiling_height_m") is not None]
            rooms = [int(row["rooms"]) for row in object_apartments if row.get("rooms") is not None]
            if object_apartments:
                studio_count = sum(
                    "студия" in normalize_text(row.get("unit_type", "")).lower()
                    for row in object_apartments
                )
                one_room = sum(
                    row.get("rooms") == 1
                    and "студия" not in normalize_text(row.get("unit_type", "")).lower()
                    for row in object_apartments
                )
                two_room = sum(row.get("rooms") == 2 for row in object_apartments)
                three_room = sum(row.get("rooms") == 3 for row in object_apartments)
                four_plus = sum(
                    row.get("rooms") is not None and int(row["rooms"]) >= 4
                    for row in object_apartments
                )
            else:
                studio_count = one_room = two_room = three_room = four_plus = None

            gross_area = FieldIndex.group_area(overview, "9.2.21")
            residential_area = FieldIndex.group_area(area_group, "9.3.1")
            nonres_area = FieldIndex.group_area(area_group, "9.3.2")
            saleable_area = FieldIndex.group_area(area_group, "9.3.3")
            area_bridge = derive_saleable_components(
                gross_area_sqm=gross_area,
                residential_area_sqm=residential_area,
                non_residential_area_sqm=nonres_area,
                saleable_area_sqm=saleable_area,
                non_residential_rows=object_nonres,
            )
            # Keep the official 9.3.3 total for sales/reconciliation.  Cost and
            # OSP analytics use the requested denominator: housing + commercial
            # premises + parking, with an explicit 9.3.3 fallback when table 15.3
            # is not sufficiently complete.
            saleable_area = area_bridge.get("saleable_total_area_sqm") or saleable_area
            saleable_area_for_cost = area_bridge.get("saleable_area_for_cost_sqm") or saleable_area
            max_floors = FieldIndex.group_int(overview, "9.2.20")
            median_ceiling = round(statistics.median(ceilings), 3) if ceilings else None

            cost_field = index.first_field(
                "18.1.1",
                object_no=object_no,
                label_contains="Планируемая стоимость строительства",
            )
            planned_cost = parse_money(cost_field.value_text) if cost_field else None
            permit_number = index.first_text("11.1.1", object_no=object_no)
            permit_date = index.first_date("11.1.2", object_no=object_no)
            permit_until = index.first_date("11.1.3", object_no=object_no)
            transfer_date = index.first_date("17.2.2", object_no=object_no) or index.first_date("17.2.1", object_no=object_no)
            commissioning = self._planned_commissioning(index, object_no)

            utility_groups = index.groups(anchor_code="14.1.1", prefix="14.1.", object_no=object_no)
            communication_groups = index.groups(anchor_code="14.2.1", prefix="14.2.", object_no=object_no)
            expertise_groups = index.groups(anchor_code="10.4.1", prefix="10.4.", object_no=object_no)
            surveyor_groups = index.groups(anchor_code="10.2.1", prefix="10.2.", object_no=object_no)
            schedule_groups = index.groups(anchor_code="17.1.1", prefix="17.1.", object_no=object_no)
            loan_groups = index.groups(anchor_code="19.6.1.1", prefix="19.6.1.", object_no=object_no)

            utility_fees = sum(
                fee
                for fee in (FieldIndex.group_money(group, "14.1.8") for group in utility_groups)
                if fee is not None
            ) or None
            cost_bridge = derive_construction_cost(
                investment_cost_rub=planned_cost,
                gross_area_sqm=gross_area,
                saleable_area_sqm=saleable_area_for_cost,
                official_saleable_area_sqm=saleable_area,
                utility_connection_fees_rub=utility_fees,
                config=ConstructionCostConfig(
                    deduct_utility_fees=self.config.deduct_utility_fees_from_investment_cost,
                    additional_nonconstruction_share_pct=self.config.additional_nonconstruction_share_pct,
                ),
            )
            finish = classify_finish_texts(
                f"{field.label}: {field.value_text}"
                for field in index.fields
                if field.object_no in (None, object_no)
            )
            loan_amounts = [FieldIndex.group_money(group, "19.6.1.4") for group in loan_groups]
            loan_debts = [FieldIndex.group_money(group, "19.6.1.5") for group in loan_groups]
            loan_unused = [FieldIndex.group_money(group, "19.6.1.6") for group in loan_groups]
            loan_amount = sum(value for value in loan_amounts if value is not None) or None
            loan_debt = sum(value for value in loan_debts if value is not None) or None
            unused = sum(value for value in loan_unused if value is not None) or None

            sold_apartment_contracts = index.first_int("19.7.1.1.1.1", object_no=object_no)
            sold_nonres_contracts = index.first_int("19.7.1.1.2.1", object_no=object_no)
            sold_apartment_area = index.first_area("19.7.2.1.1.1", object_no=object_no)
            sold_nonres_area = index.first_area("19.7.2.1.2.1", object_no=object_no)
            sold_apartment_revenue = index.first_money("19.7.3.1.1.1", object_no=object_no)
            sold_nonres_revenue = index.first_money("19.7.3.1.2.1", object_no=object_no)

            floor_label, kfl = floor_band(max_floors)
            class_label, kcl, class_confidence = class_proxy(median_ceiling, f"{header.title or ''} {FieldIndex.group_text(overview, '9.2.2') or ''}")
            theoretical_cost = None
            if kfl is not None and kcl is not None:
                theoretical_cost = round(self.config.base_cost_rub_per_sqm * kfl * kcl * self.config.deflator, 2)

            parsed_apartment_area = round(sum(apartment_areas), 3) if apartment_areas else None
            parsed_apartment_living_area = round(sum(apartment_living_areas), 3) if apartment_living_areas else None
            parsed_nonres_area = round(sum(parsed_nonres_areas), 3) if parsed_nonres_areas else None
            common_area = round(sum(common_areas), 3) if common_areas else None
            parking_spaces = self._parking_spaces(index, object_no)
            total_sold_revenue = (sold_apartment_revenue or 0.0) + (sold_nonres_revenue or 0.0)
            if sold_apartment_revenue is None and sold_nonres_revenue is None:
                total_sold_revenue = None
            unsold_units = (declared_apartments - sold_apartment_contracts) if declared_apartments is not None and sold_apartment_contracts is not None else None
            unsold_area = (residential_area - sold_apartment_area) if residential_area is not None and sold_apartment_area is not None else None
            actual_saleable_cost = cost_bridge.get("construction_cost_per_saleable_sqm")

            obj = ObjectSummary(
                object_no=object_no,
                name=FieldIndex.group_text(overview, "9.2.2"),
                object_type=FieldIndex.group_text(overview, "9.2.1"),
                address=FieldIndex.group_text(overview, "9.2.17"),
                region=FieldIndex.group_text(overview, "9.2.3"),
                settlement=FieldIndex.group_text(overview, "9.2.6"),
                purpose=FieldIndex.group_text(overview, "9.2.18"),
                residential_complex_name=index.first_text("10.6.1", object_no=object_no),
                min_floors=FieldIndex.group_int(overview, "9.2.19"),
                max_floors=max_floors,
                gross_area_sqm=gross_area,
                residential_area_sqm=residential_area,
                non_residential_area_sqm=nonres_area,
                saleable_area_sqm=saleable_area,
                gross_construction_area_sqm=area_bridge.get("gross_construction_area_sqm"),
                saleable_housing_area_sqm=area_bridge.get("saleable_housing_area_sqm"),
                saleable_commercial_area_sqm=area_bridge.get("saleable_commercial_area_sqm"),
                saleable_storage_area_sqm=area_bridge.get("saleable_storage_area_sqm"),
                saleable_parking_area_sqm=area_bridge.get("saleable_parking_area_sqm"),
                saleable_other_nonres_area_sqm=area_bridge.get("saleable_other_nonres_area_sqm"),
                saleable_nonresidential_area_sqm=area_bridge.get("saleable_nonresidential_area_sqm"),
                saleable_total_area_sqm=area_bridge.get("saleable_total_area_sqm"),
                official_saleable_area_sqm=area_bridge.get("official_saleable_area_sqm"),
                saleable_area_for_cost_sqm=area_bridge.get("saleable_area_for_cost_sqm"),
                saleable_area_for_cost_source=area_bridge.get("saleable_area_for_cost_source"),
                saleable_area_for_cost_confidence=area_bridge.get("saleable_area_for_cost_confidence"),
                nonresidential_component_coverage_pct=area_bridge.get("nonresidential_component_coverage_pct"),
                gross_to_saleable_ratio=area_bridge.get("gross_to_saleable_ratio"),
                official_gross_to_saleable_ratio=area_bridge.get("official_gross_to_saleable_ratio"),
                official_saleable_efficiency_pct=area_bridge.get("official_saleable_efficiency_pct"),
                non_saleable_area_sqm=area_bridge.get("non_saleable_area_sqm"),
                saleable_components_source=area_bridge.get("saleable_components_source"),
                saleable_components_confidence=area_bridge.get("saleable_components_confidence"),
                finish_type=finish.get("finish_type"),
                finish_confidence=finish.get("finish_confidence"),
                finish_evidence=finish.get("finish_evidence"),
                finish_source=finish.get("finish_source"),
                wall_material=FieldIndex.group_text(overview, "9.2.22"),
                floor_material=FieldIndex.group_text(overview, "9.2.23"),
                energy_class=FieldIndex.group_text(overview, "9.2.24"),
                passenger_lifts=FieldIndex.group_int(lift_group, "9.4.1"),
                freight_lifts=FieldIndex.group_int(lift_group, "9.4.2"),
                passenger_freight_lifts=FieldIndex.group_int(lift_group, "9.4.3"),
                accessibility_lifts=FieldIndex.group_int(lift_group, "9.4.4"),
                apartment_count_declared=declared_apartments,
                apartment_count_parsed=len(object_apartments),
                non_residential_count_declared=declared_nonres,
                non_residential_count_parsed=len(object_nonres),
                parsed_apartment_area_sqm=parsed_apartment_area,
                parsed_apartment_living_area_sqm=parsed_apartment_living_area,
                parsed_non_residential_area_sqm=parsed_nonres_area,
                parsed_apartment_area_coverage_pct=round(safe_div(parsed_apartment_area, residential_area) * 100.0, 3) if safe_div(parsed_apartment_area, residential_area) is not None else None,
                parsed_non_residential_area_coverage_pct=round(safe_div(parsed_nonres_area, nonres_area) * 100.0, 3) if safe_div(parsed_nonres_area, nonres_area) is not None else None,
                common_property_count_parsed=len(object_common),
                common_property_area_sqm=common_area,
                common_property_share_pct=round(safe_div(common_area, gross_area) * 100.0, 2) if safe_div(common_area, gross_area) is not None else None,
                engineering_equipment_count_parsed=len(object_equipment),
                parking_spaces_declared=parking_spaces,
                parking_spaces_per_100_apartments=round(safe_div(parking_spaces, declared_apartments) * 100.0, 2) if safe_div(parking_spaces, declared_apartments) is not None else None,
                lifts_per_100_apartments=round(safe_div(FieldIndex.group_int(lift_group, "9.4.1"), declared_apartments) * 100.0, 2) if safe_div(FieldIndex.group_int(lift_group, "9.4.1"), declared_apartments) is not None else None,
                median_ceiling_height_m=median_ceiling,
                average_apartment_area_sqm=round(float(np.mean(apartment_areas)), 3) if apartment_areas else None,
                median_apartment_area_sqm=round(float(np.median(apartment_areas)), 3) if apartment_areas else None,
                average_non_residential_unit_area_sqm=round(float(np.mean(parsed_nonres_areas)), 3) if parsed_nonres_areas else None,
                gross_area_per_apartment_sqm=round(safe_div(gross_area, declared_apartments), 3) if safe_div(gross_area, declared_apartments) is not None else None,
                saleable_area_per_apartment_sqm=round(safe_div(saleable_area, declared_apartments), 3) if safe_div(saleable_area, declared_apartments) is not None else None,
                studio_count=studio_count,
                one_room_count=one_room,
                two_room_count=two_room,
                three_room_count=three_room,
                four_plus_room_count=four_plus,
                construction_permit_number=permit_number,
                construction_permit_date=permit_date,
                permit_valid_until=permit_until,
                planned_transfer_date=transfer_date,
                planned_commissioning_period=commissioning,
                planned_construction_cost_rub=planned_cost,
                investment_cost_rub=cost_bridge.get("investment_cost_rub"),
                investment_cost_per_gross_sqm=cost_bridge.get("investment_cost_per_gross_sqm"),
                investment_cost_per_saleable_sqm=cost_bridge.get("investment_cost_per_saleable_sqm"),
                investment_cost_per_official_saleable_sqm=cost_bridge.get("investment_cost_per_official_saleable_sqm"),
                disclosed_nonconstruction_cost_rub=cost_bridge.get("disclosed_nonconstruction_cost_rub"),
                additional_nonconstruction_cost_rub=cost_bridge.get("additional_nonconstruction_cost_rub"),
                construction_cost_rub=cost_bridge.get("construction_cost_rub"),
                construction_cost_per_gross_sqm=cost_bridge.get("construction_cost_per_gross_sqm"),
                construction_cost_per_saleable_sqm=cost_bridge.get("construction_cost_per_saleable_sqm"),
                construction_cost_per_official_saleable_sqm=cost_bridge.get("construction_cost_per_official_saleable_sqm"),
                construction_share_of_investment_pct=cost_bridge.get("construction_share_of_investment_pct"),
                investment_to_construction_delta_rub=cost_bridge.get("investment_to_construction_delta_rub"),
                construction_cost_bridge_method=cost_bridge.get("construction_cost_bridge_method"),
                construction_cost_confidence=cost_bridge.get("construction_cost_confidence"),
                construction_cost_confidence_label=cost_bridge.get("construction_cost_confidence_label"),
                construction_cost_components_json=cost_bridge.get("construction_cost_components_json"),
                utility_connection_fees_rub=utility_fees,
                utility_fee_per_gross_sqm=round(safe_div(utility_fees, gross_area), 2) if safe_div(utility_fees, gross_area) is not None else None,
                loan_amount_rub=loan_amount,
                loan_debt_rub=loan_debt,
                loan_unused_rub=unused,
                loan_utilization_pct=round(safe_div(loan_debt, loan_amount) * 100.0, 2) if safe_div(loan_debt, loan_amount) is not None else None,
                loan_to_cost_pct=round(safe_div(loan_amount, cost_bridge.get("construction_cost_rub")) * 100.0, 2) if safe_div(loan_amount, cost_bridge.get("construction_cost_rub")) is not None else None,
                loan_debt_to_cost_pct=round(safe_div(loan_debt, cost_bridge.get("construction_cost_rub")) * 100.0, 2) if safe_div(loan_debt, cost_bridge.get("construction_cost_rub")) is not None else None,
                sold_apartment_contracts=sold_apartment_contracts,
                sold_nonres_contracts=sold_nonres_contracts,
                sold_apartment_area_sqm=sold_apartment_area,
                sold_nonres_area_sqm=sold_nonres_area,
                sold_apartment_revenue_rub=sold_apartment_revenue,
                sold_nonres_revenue_rub=sold_nonres_revenue,
                total_sold_revenue_rub=total_sold_revenue,
                avg_apartment_sale_price_rub_per_sqm=round(safe_div(sold_apartment_revenue, sold_apartment_area), 2) if safe_div(sold_apartment_revenue, sold_apartment_area) is not None else None,
                avg_nonres_sale_price_rub_per_sqm=round(safe_div(sold_nonres_revenue, sold_nonres_area), 2) if safe_div(sold_nonres_revenue, sold_nonres_area) is not None else None,
                estimated_unsold_apartment_units=unsold_units,
                estimated_unsold_apartment_area_sqm=round(unsold_area, 3) if unsold_area is not None else None,
                sold_apartment_unit_share_pct=round(safe_div(sold_apartment_contracts, declared_apartments) * 100.0, 2) if safe_div(sold_apartment_contracts, declared_apartments) is not None else None,
                sold_apartment_area_share_pct=round(safe_div(sold_apartment_area, residential_area) * 100.0, 2) if safe_div(sold_apartment_area, residential_area) is not None else None,
                floor_zone=floor_label,
                k_fl=kfl,
                class_proxy=class_label,
                class_proxy_confidence=class_confidence,
                k_cl=kcl,
                theoretical_cost_rub_per_sqm=theoretical_cost,
                cost_gap_to_parametric_pct=round(pct_delta(actual_saleable_cost, theoretical_cost), 2) if pct_delta(actual_saleable_cost, theoretical_cost) is not None else None,
                saleable_efficiency_pct=round(safe_div(saleable_area_for_cost, gross_area) * 100.0, 2) if safe_div(saleable_area_for_cost, gross_area) is not None else None,
                construction_duration_months=months_between(permit_date, transfer_date),
            )
            objects.append(obj)

            collections["objects"][str(object_no)] = {
                "engineering_connections": [self._group_to_values(group) for group in utility_groups],
                "communication_connections": [self._group_to_values(group) for group in communication_groups],
                "expertise_conclusions": [self._group_to_values(group) for group in expertise_groups],
                "surveyors": [self._group_to_values(group) for group in surveyor_groups],
                "schedule": [self._group_to_values(group) for group in schedule_groups],
                "loans": [self._group_to_values(group) for group in loan_groups],
                "general_contractor": {
                    "legal_form": index.first_text("10.7.1", object_no=object_no),
                    "name": index.first_text("10.7.2", object_no=object_no),
                    "inn": index.first_text("10.7.6", object_no=object_no),
                },
                "land": {
                    "right_type": index.first_text("12.1.1", object_no=object_no),
                    "basis_document": index.first_text("12.1.2", object_no=object_no),
                    "basis_date": index.first_date("12.1.4", object_no=object_no),
                    "registration_date": index.first_date("12.1.11", object_no=object_no),
                    "cadastral_number": index.first_text("12.3.1", object_no=object_no),
                    "land_area_sqm": index.first_area("12.3.2", object_no=object_no),
                },
                "escrow": {
                    "method": index.first_text("19.1.1", object_no=object_no),
                    "bank_legal_form": index.first_text("19.2.1", object_no=object_no),
                    "bank_name": index.first_text("19.2.2", object_no=object_no),
                    "bank_inn": index.first_text("19.2.3", object_no=object_no),
                },
            }

        return objects, collections

    @staticmethod
    def _group_to_values(group: Mapping[str, Sequence[ExtractedField]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for code, fields in group.items():
            values = [field.value_text for field in fields if field.value_text]
            if not values:
                continue
            result[code] = values[0] if len(values) == 1 else values
        return result

    @staticmethod
    def _planned_commissioning(index: FieldIndex, object_no: int) -> Optional[str]:
        groups = index.groups(anchor_code="17.1.1", prefix="17.1.", object_no=object_no)
        for group in groups:
            stage = FieldIndex.group_text(group, "17.1.1") or ""
            if "разрешения на ввод" in stage.lower():
                return FieldIndex.group_text(group, "17.1.2")
        return None

    @staticmethod
    def _parking_spaces(index: FieldIndex, object_no: int) -> Optional[int]:
        values = []
        for code in ("13.1.2.1", "13.1.2.2"):
            value = index.first_int(code, object_no=object_no)
            if value is not None:
                values.append(value)
        return sum(values) if values else None

    @staticmethod
    def _extract_beneficial_owners(index: FieldIndex) -> list[dict[str, Any]]:
        groups = index.groups(anchor_code="3.4.1", prefix="3.4.", object_no=None)
        owners = []
        for group in groups:
            surname = FieldIndex.group_text(group, "3.4.1")
            if not surname:
                continue
            owners.append(
                {
                    "surname": surname,
                    "name": FieldIndex.group_text(group, "3.4.2"),
                    "patronymic": FieldIndex.group_text(group, "3.4.3"),
                    "citizenship": FieldIndex.group_text(group, "3.4.4"),
                    "share_pct": parse_percent(FieldIndex.group_text(group, "3.4.5")),
                    "snils": FieldIndex.group_text(group, "3.4.6"),
                    "inn": FieldIndex.group_text(group, "3.4.7"),
                }
            )
        return owners

    @staticmethod
    def _extract_group_companies(index: FieldIndex) -> list[dict[str, Any]]:
        groups = index.groups(anchor_code="3.1.2.1", prefix="3.1.2.", object_no=None)
        companies = []
        for group in groups:
            name = FieldIndex.group_text(group, "3.1.2.2")
            if not name:
                continue
            companies.append(
                {
                    "legal_form": FieldIndex.group_text(group, "3.1.2.1"),
                    "name": name,
                    "inn": FieldIndex.group_text(group, "3.1.2.3"),
                    "ogrn": FieldIndex.group_text(group, "3.1.2.4"),
                    "basis": FieldIndex.group_text(group, "3.1.2.5"),
                }
            )
        return companies

    @staticmethod
    def _build_project(
        *,
        file_name: str,
        sha256: str,
        header: HeaderInfo,
        developer: DeveloperInfo,
        index: FieldIndex,
        objects: Sequence[ObjectSummary],
        fields: Sequence[ExtractedField],
        raw_table_rows: Sequence[Mapping[str, Any]],
    ) -> ProjectSummary:
        def sum_attr(name: str, ndigits: Optional[int] = None) -> Optional[float]:
            values = [getattr(obj, name) for obj in objects if getattr(obj, name) is not None]
            if not values:
                return None
            total = float(sum(values))
            return round(total, ndigits) if ndigits is not None else total

        gross_area = sum_attr("gross_area_sqm", 3)
        residential_area = sum_attr("residential_area_sqm", 3)
        nonres_area = sum_attr("non_residential_area_sqm", 3)
        saleable_area = sum_attr("saleable_area_sqm", 3)
        saleable_area_for_cost = sum_attr("saleable_area_for_cost_sqm", 3) or saleable_area
        official_saleable_area = sum_attr("official_saleable_area_sqm", 3) or saleable_area
        planned_cost = sum_attr("planned_construction_cost_rub", 2)
        investment_cost = sum_attr("investment_cost_rub", 2) or planned_cost
        construction_cost = sum_attr("construction_cost_rub", 2)
        disclosed_nonconstruction_cost = sum_attr("disclosed_nonconstruction_cost_rub", 2)
        additional_nonconstruction_cost = sum_attr("additional_nonconstruction_cost_rub", 2)
        saleable_housing_area = sum_attr("saleable_housing_area_sqm", 3)
        saleable_commercial_area = sum_attr("saleable_commercial_area_sqm", 3)
        saleable_storage_area = sum_attr("saleable_storage_area_sqm", 3)
        saleable_parking_area = sum_attr("saleable_parking_area_sqm", 3)
        saleable_other_nonres_area = sum_attr("saleable_other_nonres_area_sqm", 3)
        finish_summary = aggregate_finish(obj.finish_type for obj in objects)
        land_areas = [
            index.first_area("12.3.2", object_no=obj.object_no)
            for obj in objects
        ]
        # Один и тот же участок повторяется в каждом объектном блоке — не суммируем дубликаты.
        unique_land = sorted({round(value, 6) for value in land_areas if value is not None})
        land_area = unique_land[0] if len(unique_land) == 1 else (sum(unique_land) if unique_land else None)

        apartments_declared = int(sum(obj.apartment_count_declared or 0 for obj in objects)) or None
        apartments_parsed = int(sum(obj.apartment_count_parsed or 0 for obj in objects)) or None
        nonres_declared = int(sum(obj.non_residential_count_declared or 0 for obj in objects)) or None
        nonres_parsed = int(sum(obj.non_residential_count_parsed or 0 for obj in objects)) or None
        parsed_apartment_area = sum_attr("parsed_apartment_area_sqm", 3)
        parsed_apartment_living_area = sum_attr("parsed_apartment_living_area_sqm", 3)
        parsed_nonres_area = sum_attr("parsed_non_residential_area_sqm", 3)
        common_property_count = int(sum(obj.common_property_count_parsed or 0 for obj in objects)) or None
        common_property_area = sum_attr("common_property_area_sqm", 3)
        equipment_count = int(sum(obj.engineering_equipment_count_parsed or 0 for obj in objects)) or None
        passenger_lifts = int(sum(obj.passenger_lifts or 0 for obj in objects)) or None
        parking_spaces = int(sum(obj.parking_spaces_declared or 0 for obj in objects))
        if apartments_parsed is not None:
            studio_count = int(sum(obj.studio_count or 0 for obj in objects))
            one_room_count = int(sum(obj.one_room_count or 0 for obj in objects))
            two_room_count = int(sum(obj.two_room_count or 0 for obj in objects))
            three_room_count = int(sum(obj.three_room_count or 0 for obj in objects))
            four_plus_room_count = int(sum(obj.four_plus_room_count or 0 for obj in objects))
        else:
            studio_count = one_room_count = two_room_count = three_room_count = four_plus_room_count = None
        ceilings = [obj.median_ceiling_height_m for obj in objects if obj.median_ceiling_height_m is not None]

        sold_contracts = int(sum(obj.sold_apartment_contracts or 0 for obj in objects)) or None
        sold_nonres_contracts = int(sum(obj.sold_nonres_contracts or 0 for obj in objects)) or None
        sold_area = sum_attr("sold_apartment_area_sqm", 3)
        sold_nonres_area = sum_attr("sold_nonres_area_sqm", 3)
        sold_revenue = sum_attr("sold_apartment_revenue_rub", 2)
        sold_nonres_revenue = sum_attr("sold_nonres_revenue_rub", 2)
        loan_amount = sum_attr("loan_amount_rub", 2)
        loan_debt = sum_attr("loan_debt_rub", 2)
        loan_unused = sum_attr("loan_unused_rub", 2)
        utility_fees = sum_attr("utility_connection_fees_rub", 2)

        parametric_total = sum(
            (obj.theoretical_cost_rub_per_sqm or 0.0) * (obj.saleable_area_for_cost_sqm or obj.saleable_area_sqm or 0.0)
            for obj in objects
            if obj.theoretical_cost_rub_per_sqm is not None and (obj.saleable_area_for_cost_sqm is not None or obj.saleable_area_sqm is not None)
        ) or None
        if parametric_total is not None:
            parametric_total = round(float(parametric_total), 2)
        parametric_unit = safe_div(parametric_total, saleable_area_for_cost)
        actual_saleable_cost = safe_div(construction_cost, saleable_area_for_cost)
        total_sold_revenue = (sold_revenue or 0.0) + (sold_nonres_revenue or 0.0)
        if sold_revenue is None and sold_nonres_revenue is None:
            total_sold_revenue = None
        total_sold_area = (sold_area or 0.0) + (sold_nonres_area or 0.0)
        if sold_area is None and sold_nonres_area is None:
            total_sold_area = None
        unsold_units = (apartments_declared - sold_contracts) if apartments_declared is not None and sold_contracts is not None else None
        unsold_area = (residential_area - sold_area) if residential_area is not None and sold_area is not None else None

        return ProjectSummary(
            file_name=file_name,
            sha256=sha256,
            declaration_number=header.declaration_number,
            declaration_date=header.declaration_date,
            first_publication_date=header.first_publication_date,
            title=header.title,
            address=header.address,
            cadastral_number=header.cadastral_number,
            residential_complex_name=next((obj.residential_complex_name for obj in objects if obj.residential_complex_name), None),
            developer_name=developer.full_name,
            developer_inn=developer.inn,
            object_count_declared=index.first_int("9.1.1", object_no=None),
            object_count_parsed=len(objects),
            gross_area_sqm=gross_area,
            residential_area_sqm=residential_area,
            non_residential_area_sqm=nonres_area,
            saleable_area_sqm=saleable_area,
            gross_construction_area_sqm=gross_area,
            saleable_housing_area_sqm=saleable_housing_area or residential_area,
            saleable_commercial_area_sqm=saleable_commercial_area,
            saleable_storage_area_sqm=saleable_storage_area,
            saleable_parking_area_sqm=saleable_parking_area,
            saleable_other_nonres_area_sqm=saleable_other_nonres_area,
            saleable_nonresidential_area_sqm=nonres_area,
            saleable_total_area_sqm=saleable_area,
            official_saleable_area_sqm=official_saleable_area,
            saleable_area_for_cost_sqm=saleable_area_for_cost,
            saleable_area_for_cost_source="Агрегация: жильё + коммерческие помещения + паркинг; fallback 9.3.3",
            saleable_area_for_cost_confidence=min((obj.saleable_area_for_cost_confidence or 0.0 for obj in objects), default=0.0),
            nonresidential_component_coverage_pct=round(float(np.nanmean([obj.nonresidential_component_coverage_pct for obj in objects if obj.nonresidential_component_coverage_pct is not None])), 2) if any(obj.nonresidential_component_coverage_pct is not None for obj in objects) else None,
            gross_to_saleable_ratio=round(safe_div(gross_area, saleable_area_for_cost), 4) if safe_div(gross_area, saleable_area_for_cost) is not None else None,
            official_gross_to_saleable_ratio=round(safe_div(gross_area, official_saleable_area), 4) if safe_div(gross_area, official_saleable_area) is not None else None,
            official_saleable_efficiency_pct=round(safe_div(official_saleable_area, gross_area) * 100.0, 2) if safe_div(official_saleable_area, gross_area) is not None else None,
            non_saleable_area_sqm=round(max(0.0, gross_area - saleable_area_for_cost), 3) if gross_area is not None and saleable_area_for_cost is not None else None,
            saleable_components_source="Агрегация объектных полей 9.2/9.3 и таблиц 15.3",
            saleable_components_confidence=min((obj.saleable_components_confidence or 0.0 for obj in objects), default=0.0),
            finish_type=finish_summary.get("finish_type"),
            finish_confidence=finish_summary.get("finish_confidence"),
            finish_source="Агрегация по объектам декларации",
            land_area_sqm=land_area,
            floor_area_ratio=round(safe_div(gross_area, land_area), 4) if safe_div(gross_area, land_area) is not None else None,
            saleable_efficiency_pct=round(safe_div(saleable_area, gross_area) * 100.0, 2) if safe_div(saleable_area, gross_area) is not None else None,
            apartments_declared=apartments_declared,
            apartments_parsed=apartments_parsed,
            non_residential_units_declared=nonres_declared,
            non_residential_units_parsed=nonres_parsed,
            parsed_apartment_area_sqm=parsed_apartment_area,
            parsed_apartment_living_area_sqm=parsed_apartment_living_area,
            parsed_non_residential_area_sqm=parsed_nonres_area,
            parsed_apartment_area_coverage_pct=round(safe_div(parsed_apartment_area, residential_area) * 100.0, 3) if safe_div(parsed_apartment_area, residential_area) is not None else None,
            parsed_non_residential_area_coverage_pct=round(safe_div(parsed_nonres_area, nonres_area) * 100.0, 3) if safe_div(parsed_nonres_area, nonres_area) is not None else None,
            common_property_count_parsed=common_property_count,
            common_property_area_sqm=common_property_area,
            common_property_share_pct=round(safe_div(common_property_area, gross_area) * 100.0, 2) if safe_div(common_property_area, gross_area) is not None else None,
            engineering_equipment_count_parsed=equipment_count,
            passenger_lifts=passenger_lifts,
            parking_spaces_declared=parking_spaces,
            lifts_per_100_apartments=round(safe_div(passenger_lifts, apartments_declared) * 100.0, 2) if safe_div(passenger_lifts, apartments_declared) is not None else None,
            parking_spaces_per_100_apartments=round(safe_div(parking_spaces, apartments_declared) * 100.0, 2) if safe_div(parking_spaces, apartments_declared) is not None else None,
            max_floors=max((obj.max_floors for obj in objects if obj.max_floors is not None), default=None),
            median_ceiling_height_m=round(float(statistics.median(ceilings)), 3) if ceilings else None,
            average_apartment_area_sqm=round(safe_div(residential_area, apartments_declared), 3) if safe_div(residential_area, apartments_declared) is not None else None,
            average_non_residential_unit_area_sqm=round(safe_div(nonres_area, nonres_declared), 3) if safe_div(nonres_area, nonres_declared) is not None else None,
            gross_area_per_apartment_sqm=round(safe_div(gross_area, apartments_declared), 3) if safe_div(gross_area, apartments_declared) is not None else None,
            saleable_area_per_apartment_sqm=round(safe_div(saleable_area, apartments_declared), 3) if safe_div(saleable_area, apartments_declared) is not None else None,
            studio_count=studio_count,
            one_room_count=one_room_count,
            two_room_count=two_room_count,
            three_room_count=three_room_count,
            four_plus_room_count=four_plus_room_count,
            planned_construction_cost_rub=planned_cost,
            investment_cost_rub=investment_cost,
            investment_cost_per_gross_sqm=round(safe_div(investment_cost, gross_area), 2) if safe_div(investment_cost, gross_area) is not None else None,
            investment_cost_per_saleable_sqm=round(safe_div(investment_cost, saleable_area_for_cost), 2) if safe_div(investment_cost, saleable_area_for_cost) is not None else None,
            investment_cost_per_official_saleable_sqm=round(safe_div(investment_cost, official_saleable_area), 2) if safe_div(investment_cost, official_saleable_area) is not None else None,
            disclosed_nonconstruction_cost_rub=disclosed_nonconstruction_cost,
            additional_nonconstruction_cost_rub=additional_nonconstruction_cost,
            construction_cost_rub=construction_cost,
            planned_cost_per_apartment_rub=round(safe_div(investment_cost, apartments_declared), 2) if safe_div(investment_cost, apartments_declared) is not None else None,
            construction_cost_per_gross_sqm=round(safe_div(construction_cost, gross_area), 2) if safe_div(construction_cost, gross_area) is not None else None,
            construction_cost_per_saleable_sqm=round(safe_div(construction_cost, saleable_area_for_cost), 2) if safe_div(construction_cost, saleable_area_for_cost) is not None else None,
            construction_cost_per_official_saleable_sqm=round(safe_div(construction_cost, official_saleable_area), 2) if safe_div(construction_cost, official_saleable_area) is not None else None,
            construction_share_of_investment_pct=round(safe_div(construction_cost, investment_cost) * 100.0, 2) if safe_div(construction_cost, investment_cost) is not None else None,
            investment_to_construction_delta_rub=round(investment_cost - construction_cost, 2) if investment_cost is not None and construction_cost is not None else None,
            construction_cost_bridge_method="Сумма объектных мостов инвестиционная → строительная",
            construction_cost_confidence=min((obj.construction_cost_confidence or 0.0 for obj in objects), default=0.0),
            construction_cost_confidence_label="Средняя" if objects and min((obj.construction_cost_confidence or 0.0 for obj in objects), default=0.0) >= 0.55 else "Ограниченная",
            parametric_cost_rub_per_saleable_sqm=round(parametric_unit, 2) if parametric_unit is not None else None,
            parametric_total_cost_rub=parametric_total,
            cost_gap_to_parametric_pct=round(pct_delta(actual_saleable_cost, parametric_unit), 2) if pct_delta(actual_saleable_cost, parametric_unit) is not None else None,
            utility_connection_fees_rub=utility_fees,
            utility_fee_per_gross_sqm=round(safe_div(utility_fees, gross_area), 2) if safe_div(utility_fees, gross_area) is not None else None,
            total_loan_amount_rub=loan_amount,
            total_loan_debt_rub=loan_debt,
            total_loan_unused_rub=loan_unused,
            loan_utilization_pct=round(safe_div(loan_debt, loan_amount) * 100.0, 2) if safe_div(loan_debt, loan_amount) is not None else None,
            loan_to_cost_pct=round(safe_div(loan_amount, construction_cost) * 100.0, 2) if safe_div(loan_amount, construction_cost) is not None else None,
            loan_debt_to_cost_pct=round(safe_div(loan_debt, construction_cost) * 100.0, 2) if safe_div(loan_debt, construction_cost) is not None else None,
            loan_unused_to_cost_pct=round(safe_div(loan_unused, construction_cost) * 100.0, 2) if safe_div(loan_unused, construction_cost) is not None else None,
            sold_apartment_contracts=sold_contracts,
            sold_nonres_contracts=sold_nonres_contracts,
            sold_apartment_area_sqm=sold_area,
            sold_nonres_area_sqm=sold_nonres_area,
            sold_apartment_revenue_rub=sold_revenue,
            sold_nonres_revenue_rub=sold_nonres_revenue,
            total_sold_revenue_rub=round(total_sold_revenue, 2) if total_sold_revenue is not None else None,
            total_sold_area_sqm=round(total_sold_area, 3) if total_sold_area is not None else None,
            avg_apartment_sale_price_rub_per_sqm=round(safe_div(sold_revenue, sold_area), 2) if safe_div(sold_revenue, sold_area) is not None else None,
            avg_nonres_sale_price_rub_per_sqm=round(safe_div(sold_nonres_revenue, sold_nonres_area), 2) if safe_div(sold_nonres_revenue, sold_nonres_area) is not None else None,
            blended_avg_sale_price_rub_per_sqm=round(safe_div(total_sold_revenue, total_sold_area), 2) if safe_div(total_sold_revenue, total_sold_area) is not None else None,
            estimated_unsold_apartment_units=unsold_units,
            estimated_unsold_apartment_area_sqm=round(unsold_area, 3) if unsold_area is not None else None,
            sold_apartment_unit_share_pct=round(safe_div(sold_contracts, apartments_declared) * 100.0, 2) if safe_div(sold_contracts, apartments_declared) is not None else None,
            sold_apartment_area_share_pct=round(safe_div(sold_area, residential_area) * 100.0, 2) if safe_div(sold_area, residential_area) is not None else None,
            developer_net_profit_to_cost_pct=round(safe_div(developer.net_profit_rub, construction_cost) * 100.0, 2) if safe_div(developer.net_profit_rub, construction_cost) is not None else None,
            developer_payables_to_cost_pct=round(safe_div(developer.accounts_payable_rub, construction_cost) * 100.0, 2) if safe_div(developer.accounts_payable_rub, construction_cost) is not None else None,
            developer_receivables_to_cost_pct=round(safe_div(developer.accounts_receivable_rub, construction_cost) * 100.0, 2) if safe_div(developer.accounts_receivable_rub, construction_cost) is not None else None,
            land_area_per_saleable_sqm=round(safe_div(land_area, saleable_area_for_cost), 4) if safe_div(land_area, saleable_area_for_cost) is not None else None,
            raw_field_count=len(fields),
            filled_field_count=sum(bool(field.value_text) for field in fields),
            parsed_table_row_count=len(raw_table_rows),
        )


# -----------------------------------------------------------------------------
# Контроль качества и согласованности
# -----------------------------------------------------------------------------

class QualityEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        project: ProjectSummary,
        objects: Sequence[ObjectSummary],
        index: FieldIndex,
    ) -> tuple[list[QualityCheck], float]:
        checks: list[QualityCheck] = []

        self._compare_counts(
            checks,
            scope="project",
            check="object_count",
            actual=project.object_count_parsed,
            expected=project.object_count_declared,
            message="Количество объектных блоков должно совпадать с п. 9.1.1.",
        )

        for obj in objects:
            scope = f"object_{obj.object_no}"
            self._compare_counts(
                checks,
                scope=scope,
                check="apartment_count",
                actual=obj.apartment_count_parsed,
                expected=obj.apartment_count_declared,
                message="Количество строк квартир сверяется с п. 15.1.1.",
            )
            self._compare_counts(
                checks,
                scope=scope,
                check="non_residential_count",
                actual=obj.non_residential_count_parsed,
                expected=obj.non_residential_count_declared,
                message="Количество строк нежилых помещений сверяется с п. 15.1.2.",
            )

            if obj.residential_area_sqm is not None and obj.parsed_apartment_area_sqm is not None:
                self._compare_numeric(
                    checks,
                    scope=scope,
                    check="residential_area_table_vs_declared",
                    actual=obj.parsed_apartment_area_sqm,
                    expected=obj.residential_area_sqm,
                    tolerance_pct=max(self.config.quality_area_tolerance_pct, 0.2),
                    message="Сумма площадей квартир должна совпадать с п. 9.3.1.",
                )
            if obj.non_residential_area_sqm is not None and obj.parsed_non_residential_area_sqm is not None:
                self._compare_numeric(
                    checks,
                    scope=scope,
                    check="non_residential_area_table_vs_declared",
                    actual=obj.parsed_non_residential_area_sqm,
                    expected=obj.non_residential_area_sqm,
                    tolerance_pct=max(self.config.quality_area_tolerance_pct, 0.2),
                    message="Сумма площадей нежилых помещений должна совпадать с п. 9.3.2.",
                )

            if obj.residential_area_sqm is not None and obj.non_residential_area_sqm is not None and obj.saleable_area_sqm is not None:
                self._compare_numeric(
                    checks,
                    scope=scope,
                    check="saleable_area_identity",
                    actual=obj.residential_area_sqm + obj.non_residential_area_sqm,
                    expected=obj.saleable_area_sqm,
                    tolerance_pct=self.config.quality_area_tolerance_pct,
                    message="9.3.1 + 9.3.2 должно равняться 9.3.3.",
                )

            if obj.loan_amount_rub is not None and obj.loan_debt_rub is not None and obj.loan_unused_rub is not None:
                self._compare_numeric(
                    checks,
                    scope=scope,
                    check="loan_balance_identity",
                    actual=obj.loan_debt_rub + obj.loan_unused_rub,
                    expected=obj.loan_amount_rub,
                    tolerance_pct=0.05,
                    message="Задолженность + неиспользованный остаток должны давать сумму кредита.",
                )

            if obj.planned_construction_cost_rub is None:
                checks.append(
                    QualityCheck(
                        scope=scope,
                        check="construction_cost_present",
                        status="WARN",
                        message="Не найдена планируемая стоимость строительства (18.1.1).",
                    )
                )
            else:
                checks.append(
                    QualityCheck(
                        scope=scope,
                        check="construction_cost_present",
                        status="PASS",
                        actual=obj.planned_construction_cost_rub,
                        message="Планируемая стоимость строительства извлечена.",
                    )
                )

            if obj.construction_permit_date and obj.planned_transfer_date:
                duration = months_between(obj.construction_permit_date, obj.planned_transfer_date)
                status: Literal["PASS", "WARN", "FAIL", "INFO"] = "PASS" if duration is not None and duration > 0 else "FAIL"
                checks.append(
                    QualityCheck(
                        scope=scope,
                        check="timeline_order",
                        status=status,
                        actual=duration,
                        expected="> 0 месяцев",
                        message="Дата передачи должна быть позже даты разрешения на строительство.",
                    )
                )

        # Информационная проверка покрытия полей.
        field_coverage = safe_div(project.filled_field_count, project.raw_field_count)
        checks.append(
            QualityCheck(
                scope="project",
                check="raw_field_fill_rate",
                status="INFO",
                actual=round((field_coverage or 0) * 100.0, 2),
                expected="не применяется: официальная форма содержит множество необязательных полей",
                message="Доля непустых полей среди всех найденных кодов формы.",
            )
        )

        weights = {"PASS": 1.0, "INFO": 0.8, "WARN": 0.45, "FAIL": 0.0}
        scored = [weights[check.status] for check in checks if check.status != "INFO"]
        consistency_score = (sum(scored) / len(scored) * 100.0) if scored else 0.0
        coverage_score = min(100.0, (project.filled_field_count / max(project.raw_field_count, 1)) * 150.0)
        quality_score = round(0.75 * consistency_score + 0.25 * coverage_score, 2)
        return checks, quality_score

    def _compare_counts(
        self,
        checks: list[QualityCheck],
        *,
        scope: str,
        check: str,
        actual: Optional[int],
        expected: Optional[int],
        message: str,
    ) -> None:
        if expected is None:
            checks.append(QualityCheck(scope=scope, check=check, status="INFO", actual=actual, message=message + " Эталон отсутствует."))
            return
        if actual is None:
            checks.append(QualityCheck(scope=scope, check=check, status="FAIL", actual=actual, expected=expected, message=message))
            return
        delta = actual - expected
        status: Literal["PASS", "WARN", "FAIL", "INFO"] = "PASS" if abs(delta) <= self.config.quality_count_tolerance else "FAIL"
        checks.append(QualityCheck(scope=scope, check=check, status=status, actual=actual, expected=expected, delta=float(delta), message=message))

    @staticmethod
    def _compare_numeric(
        checks: list[QualityCheck],
        *,
        scope: str,
        check: str,
        actual: Optional[float],
        expected: Optional[float],
        tolerance_pct: float,
        message: str,
    ) -> None:
        if actual is None or expected is None:
            checks.append(QualityCheck(scope=scope, check=check, status="INFO", actual=actual, expected=expected, message=message + " Недостаточно данных."))
            return
        delta = pct_delta(actual, expected)
        if delta is None:
            status: Literal["PASS", "WARN", "FAIL", "INFO"] = "INFO"
        elif abs(delta) <= tolerance_pct:
            status = "PASS"
        elif abs(delta) <= max(2.0, tolerance_pct * 4):
            status = "WARN"
        else:
            status = "FAIL"
        checks.append(QualityCheck(scope=scope, check=check, status=status, actual=actual, expected=expected, delta=round(delta or 0.0, 4), message=message))


# -----------------------------------------------------------------------------
# Итоговый объект результата и экспорт
# -----------------------------------------------------------------------------

class DeclarationResult:
    def __init__(
        self,
        *,
        header: HeaderInfo,
        developer: DeveloperInfo,
        project: ProjectSummary,
        objects: Sequence[ObjectSummary],
        collections: Mapping[str, Any],
        raw_fields: Sequence[ExtractedField],
        apartments: Sequence[Mapping[str, Any]],
        non_residential: Sequence[Mapping[str, Any]],
        non_residential_parts: Sequence[Mapping[str, Any]],
        common_property: Sequence[Mapping[str, Any]],
        engineering_equipment: Sequence[Mapping[str, Any]],
        raw_table_rows: Sequence[Mapping[str, Any]],
        quality_checks: Sequence[QualityCheck],
        diagnostics: Diagnostics,
    ):
        self.header = header
        self.developer = developer
        self.project = project
        self.objects = list(objects)
        self.collections = dict(collections)
        self.raw_fields = list(raw_fields)
        self.apartments = list(apartments)
        self.non_residential = list(non_residential)
        self.non_residential_parts = list(non_residential_parts)
        self.common_property = list(common_property)
        self.engineering_equipment = list(engineering_equipment)
        self.raw_table_rows = list(raw_table_rows)
        self.quality_checks = list(quality_checks)
        self.diagnostics = diagnostics

    def to_dict(self, *, include_raw_fields: bool = True, include_units: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": PARSER_VERSION,
            "header": self.header.model_dump(mode="json"),
            "developer": self.developer.model_dump(mode="json"),
            "project": self.project.model_dump(mode="json"),
            "objects": [obj.model_dump(mode="json") for obj in self.objects],
            "collections": self.collections,
            "quality_checks": [check.model_dump(mode="json") for check in self.quality_checks],
            "diagnostics": self.diagnostics.model_dump(mode="json"),
            "feature_packs": self.task_feature_packs(),
        }
        if include_units:
            payload["apartments"] = self.apartments
            payload["non_residential"] = self.non_residential
            payload["non_residential_parts"] = self.non_residential_parts
            payload["common_property"] = self.common_property
            payload["engineering_equipment"] = self.engineering_equipment
        if include_raw_fields:
            payload["raw_fields"] = [field.model_dump(mode="json") for field in self.raw_fields]
        return payload

    def project_frame(self) -> pd.DataFrame:
        row = self.project.model_dump(mode="json")
        row.update({f"developer_{key}": value for key, value in self.developer.model_dump(mode="json").items()})
        return pd.DataFrame([row])

    def objects_frame(self) -> pd.DataFrame:
        return pd.DataFrame([obj.model_dump(mode="json") for obj in self.objects])

    def raw_fields_frame(self) -> pd.DataFrame:
        return pd.DataFrame([field.model_dump(mode="json") for field in self.raw_fields])

    def quality_frame(self) -> pd.DataFrame:
        return pd.DataFrame([check.model_dump(mode="json") for check in self.quality_checks])

    def field_catalog_frame(self) -> pd.DataFrame:
        """Каталог всех найденных кодов формы с покрытием, типами и доказательствами."""
        if not self.raw_fields:
            return pd.DataFrame(
                columns=[
                    "code", "canonical_label", "occurrences", "filled_occurrences",
                    "global_occurrences", "object_occurrences", "units", "methods",
                    "avg_confidence", "first_page", "last_page",
                ]
            )
        frame = self.raw_fields_frame().copy()
        frame["is_filled"] = frame["value_text"].fillna("").astype(str).str.len().gt(0)
        frame["scope"] = np.where(frame["object_no"].isna(), "global", "object")

        def most_common_text(series: pd.Series) -> Optional[str]:
            values = [normalize_text(value) for value in series if normalize_text(value)]
            return Counter(values).most_common(1)[0][0] if values else None

        rows: list[dict[str, Any]] = []
        for code, group in frame.groupby("code", sort=True):
            rows.append(
                {
                    "code": code,
                    "canonical_label": most_common_text(group["label"]),
                    "occurrences": int(len(group)),
                    "filled_occurrences": int(group["is_filled"].sum()),
                    "global_occurrences": int((group["scope"] == "global").sum()),
                    "object_occurrences": int((group["scope"] == "object").sum()),
                    "units": "; ".join(sorted({str(value) for value in group["unit"].dropna() if str(value)})),
                    "methods": "; ".join(sorted({str(value) for value in group["method"].dropna() if str(value)})),
                    "avg_confidence": round(float(pd.to_numeric(group["confidence"], errors="coerce").mean()), 4),
                    "first_page": int(group["page"].min()),
                    "last_page": int(group["page"].max()),
                }
            )
        return pd.DataFrame(rows)

    def official_fields_wide_frame(self) -> pd.DataFrame:
        """Одна строка на глобальный блок/объект, один столбец на официальный код.

        Повторяющиеся значения (например, несколько инженерных подключений)
        сохраняются JSON-массивом в ячейке, поэтому информация не теряется.
        """
        scopes: list[Optional[int]] = [None] + sorted(
            {field.object_no for field in self.raw_fields if field.object_no is not None}
        )
        rows: list[dict[str, Any]] = []
        for object_no in scopes:
            selected = [field for field in self.raw_fields if field.object_no == object_no]
            if not selected:
                continue
            grouped: dict[str, list[str]] = defaultdict(list)
            for field in selected:
                if field.value_text:
                    grouped[field.code].append(field.value_text)
            row: dict[str, Any] = {
                "file_name": self.project.file_name,
                "scope": "global" if object_no is None else "object",
                "object_no": object_no,
            }
            for code, values in grouped.items():
                key = f"field_{code.replace('.', '_')}"
                row[key] = values[0] if len(values) == 1 else json_dumps(values, indent=0)
            rows.append(row)
        return pd.DataFrame(rows)

    def query_fields(
        self,
        query: str,
        *,
        object_no: Optional[int] = None,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """Fuzzy-поиск по подписи, значению и коду поля без повторного парсинга PDF."""
        query_norm = normalize_text(query).lower()
        if not query_norm:
            return self.raw_fields_frame().head(0)
        candidates = [
            field
            for field in self.raw_fields
            if object_no is None or field.object_no == object_no
        ]
        try:
            from rapidfuzz.fuzz import token_set_ratio

            def score(field: ExtractedField) -> float:
                haystack = normalize_text(
                    f"{field.code} {field.label} {field.value_text}"
                ).lower()
                return float(token_set_ratio(query_norm, haystack))

        except ImportError:
            from difflib import SequenceMatcher

            def score(field: ExtractedField) -> float:
                haystack = normalize_text(
                    f"{field.code} {field.label} {field.value_text}"
                ).lower()
                return SequenceMatcher(None, query_norm, haystack).ratio() * 100.0

        ranked = sorted(
            ((score(field), field) for field in candidates),
            key=lambda item: (-item[0], item[1].page, item[1].ordinal),
        )[: max(1, top_k)]
        rows = []
        for relevance, field in ranked:
            row = field.model_dump(mode="json")
            row["relevance"] = round(relevance, 2)
            rows.append(row)
        return pd.DataFrame(rows)

    def task_feature_packs(self) -> dict[str, Any]:
        """Готовые группы признаков для основных задач проекта."""
        project = self.project.model_dump(mode="json")
        objects = [obj.model_dump(mode="json") for obj in self.objects]

        def pick(source: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
            return {name: source.get(name) for name in names}

        cost_names = [
            "max_floors", "gross_area_sqm", "saleable_area_sqm",
            "saleable_efficiency_pct", "common_property_area_sqm",
            "common_property_share_pct", "apartments_declared",
            "average_apartment_area_sqm", "non_residential_units_declared",
            "average_non_residential_unit_area_sqm", "passenger_lifts",
            "lifts_per_100_apartments", "parking_spaces_declared",
            "parking_spaces_per_100_apartments", "median_ceiling_height_m",
            "utility_connection_fees_rub", "utility_fee_per_gross_sqm",
            "planned_construction_cost_rub", "planned_cost_per_apartment_rub",
            "construction_cost_per_gross_sqm",
            "construction_cost_per_saleable_sqm",
            "parametric_cost_rub_per_saleable_sqm",
            "parametric_total_cost_rub", "cost_gap_to_parametric_pct",
        ]
        rlv_names = [
            "cadastral_number", "land_area_sqm", "floor_area_ratio",
            "land_area_per_saleable_sqm", "saleable_area_sqm",
            "avg_apartment_sale_price_rub_per_sqm",
            "blended_avg_sale_price_rub_per_sqm",
            "construction_cost_per_saleable_sqm",
            "parametric_cost_rub_per_saleable_sqm",
            "planned_construction_cost_rub", "estimated_unsold_apartment_area_sqm",
        ]
        sales_names = [
            "apartments_declared", "studio_count", "one_room_count",
            "two_room_count", "three_room_count", "four_plus_room_count",
            "sold_apartment_contracts", "sold_apartment_unit_share_pct",
            "sold_apartment_area_sqm", "sold_apartment_area_share_pct",
            "sold_apartment_revenue_rub",
            "avg_apartment_sale_price_rub_per_sqm",
            "sold_nonres_contracts", "sold_nonres_area_sqm",
            "sold_nonres_revenue_rub", "avg_nonres_sale_price_rub_per_sqm",
            "total_sold_revenue_rub", "total_sold_area_sqm",
            "blended_avg_sale_price_rub_per_sqm",
            "estimated_unsold_apartment_units",
            "estimated_unsold_apartment_area_sqm",
        ]
        finance_names = [
            "planned_construction_cost_rub", "total_loan_amount_rub",
            "total_loan_debt_rub", "total_loan_unused_rub",
            "loan_utilization_pct", "loan_to_cost_pct",
            "loan_debt_to_cost_pct", "loan_unused_to_cost_pct",
            "developer_net_profit_to_cost_pct",
            "developer_payables_to_cost_pct",
            "developer_receivables_to_cost_pct",
        ]
        object_cost_names = [
            "object_no", "name", "max_floors", "gross_area_sqm",
            "saleable_area_sqm", "saleable_efficiency_pct",
            "common_property_area_sqm", "common_property_share_pct",
            "median_ceiling_height_m", "wall_material", "floor_material",
            "energy_class", "passenger_lifts", "lifts_per_100_apartments",
            "parking_spaces_declared", "parking_spaces_per_100_apartments",
            "utility_connection_fees_rub", "utility_fee_per_gross_sqm",
            "planned_construction_cost_rub",
            "construction_cost_per_gross_sqm",
            "construction_cost_per_saleable_sqm", "construction_duration_months",
            "floor_zone", "k_fl", "class_proxy", "k_cl",
            "theoretical_cost_rub_per_sqm", "cost_gap_to_parametric_pct",
            "loan_to_cost_pct", "loan_debt_to_cost_pct",
        ]
        object_sales_names = [
            "object_no", "name", "apartment_count_declared",
            "apartment_count_parsed", "average_apartment_area_sqm",
            "median_apartment_area_sqm", "studio_count", "one_room_count",
            "two_room_count", "three_room_count", "four_plus_room_count",
            "sold_apartment_contracts", "sold_apartment_unit_share_pct",
            "sold_apartment_area_sqm", "sold_apartment_area_share_pct",
            "sold_apartment_revenue_rub", "avg_apartment_sale_price_rub_per_sqm",
            "sold_nonres_contracts", "sold_nonres_area_sqm",
            "sold_nonres_revenue_rub", "avg_nonres_sale_price_rub_per_sqm",
            "total_sold_revenue_rub", "estimated_unsold_apartment_units",
            "estimated_unsold_apartment_area_sqm",
        ]
        return {
            "cost_model": {
                "project": pick(project, cost_names),
                "objects": [pick(obj, object_cost_names) for obj in objects],
            },
            "rlv": {"project": pick(project, rlv_names)},
            "sales": {
                "project": pick(project, sales_names),
                "objects": [pick(obj, object_sales_names) for obj in objects],
            },
            "financing": {
                "project": pick(project, finance_names),
                "developer": pick(
                    self.developer.model_dump(mode="json"),
                    [
                        "reporting_date", "net_profit_rub",
                        "accounts_payable_rub", "accounts_receivable_rub",
                        "paid_up_capital_rub",
                    ],
                ),
            },
            "quality": {
                "quality_score": self.project.quality_score,
                "checks": [check.model_dump(mode="json") for check in self.quality_checks],
                "diagnostics": self.diagnostics.model_dump(mode="json"),
            },
        }

    def save(self, output_dir: str | Path, config: Optional[PipelineConfig] = None) -> Path:
        cfg = config or PipelineConfig(output_dir=Path(output_dir))
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = Path(self.project.file_name).stem
        target = root / stem
        signature_file = target / ".source_sha256"
        if target.exists() and signature_file.exists():
            existing_digest = signature_file.read_text(encoding="utf-8").strip()
            if existing_digest and existing_digest != self.project.sha256:
                target = root / f"{stem}__{self.project.sha256[:8]}"
                signature_file = target / ".source_sha256"
        target.mkdir(parents=True, exist_ok=True)
        signature_file.write_text(self.project.sha256, encoding="utf-8")

        if cfg.export_json:
            (target / "declaration_full.json").write_text(
                json_dumps(self.to_dict(include_raw_fields=True, include_units=True)),
                encoding="utf-8",
            )
            (target / "declaration_compact.json").write_text(
                json_dumps(self.to_dict(include_raw_fields=False, include_units=False)),
                encoding="utf-8",
            )
            (target / "task_feature_packs.json").write_text(
                json_dumps(self.task_feature_packs()),
                encoding="utf-8",
            )

        if cfg.export_csv:
            self.project_frame().to_csv(target / "project_summary.csv", index=False, encoding="utf-8-sig")
            self.objects_frame().to_csv(target / "objects.csv", index=False, encoding="utf-8-sig")
            self.raw_fields_frame().to_csv(target / "raw_fields.csv", index=False, encoding="utf-8-sig")
            self.quality_frame().to_csv(target / "quality_checks.csv", index=False, encoding="utf-8-sig")
            self.field_catalog_frame().to_csv(target / "field_catalog.csv", index=False, encoding="utf-8-sig")
            self.official_fields_wide_frame().to_csv(target / "official_fields_wide.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(self.apartments).to_csv(target / "apartments.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(self.non_residential).to_csv(target / "non_residential.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(self.non_residential_parts).to_csv(target / "non_residential_parts.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(self.common_property).to_csv(target / "common_property.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(self.engineering_equipment).to_csv(target / "engineering_equipment.csv", index=False, encoding="utf-8-sig")
            if self.raw_table_rows:
                pd.DataFrame(
                    [
                        {**{key: value for key, value in row.items() if key != "cells"}, "cells_json": json_dumps(row.get("cells", []), indent=0)}
                        for row in self.raw_table_rows
                    ]
                ).to_csv(target / "raw_table_rows.csv", index=False, encoding="utf-8-sig")

        if cfg.export_html_report:
            (target / "report.html").write_text(self._html_report(), encoding="utf-8")

        return target

    def _html_report(self) -> str:
        project_df = self.project_frame().T.reset_index()
        project_df.columns = ["Параметр", "Значение"]
        objects_html = self.objects_frame().to_html(index=False, border=0, classes="table")
        quality_html = self.quality_frame().to_html(index=False, border=0, classes="table")
        project_html = project_df.to_html(index=False, border=0, classes="table")
        return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Отчёт парсинга {self.project.file_name}</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:32px;color:#1f2937;background:#f8fafc}}
h1,h2{{color:#0f172a}} .kpi{{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}
.card{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px;min-width:180px;box-shadow:0 2px 8px #0000000a}}
.card b{{font-size:24px;display:block;margin-top:6px}} .table{{border-collapse:collapse;width:100%;background:white;margin-bottom:28px}}
.table th{{background:#0f766e;color:white;position:sticky;top:0}} .table th,.table td{{border:1px solid #dbe4ea;padding:7px;vertical-align:top}}
.table tr:nth-child(even){{background:#f1f5f9}} .small{{color:#475569;font-size:13px}}
</style></head><body>
<h1>Отчёт по декларации</h1>
<div class="small">Файл: {self.project.file_name} · parser {PARSER_VERSION}</div>
<div class="kpi">
<div class="card">Поля формы<b>{self.project.raw_field_count}</b></div>
<div class="card">Непустые поля<b>{self.project.filled_field_count}</b></div>
<div class="card">Объекты<b>{self.project.object_count_parsed}</b></div>
<div class="card">Квартиры<b>{self.project.apartments_parsed or 0}</b></div>
<div class="card">Quality score<b>{self.project.quality_score or 0:.1f}</b></div>
</div>
<h2>Проект</h2>{project_html}
<h2>Объекты</h2>{objects_html}
<h2>Контроль качества</h2>{quality_html}
</body></html>"""


# -----------------------------------------------------------------------------
# Кеш и основной pipeline
# -----------------------------------------------------------------------------

class DeclarationPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.extractor = LayoutDeclarationExtractor(self.config)
        self.builder = BusinessEntityBuilder(self.config)
        self.quality_engine = QualityEngine(self.config)
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, digest: str) -> Optional[Path]:
        if not self.config.cache_dir:
            return None
        return self.config.cache_dir / f"{PARSER_VERSION}_{digest}.json.gz"

    def parse_file(self, pdf_path: str | Path) -> DeclarationResult:
        path = Path(pdf_path)
        return self.parse_bytes(path.read_bytes(), path.name)

    def parse_bytes(self, pdf_bytes: bytes, file_name: str) -> DeclarationResult:
        started = time.perf_counter()
        digest = sha256_bytes(pdf_bytes)
        cache_path = self._cache_path(digest)
        if self.config.use_cache and cache_path and cache_path.exists():
            try:
                with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
                result = self._result_from_payload(payload)
                result.diagnostics.cache_hit = True
                result.diagnostics.elapsed_seconds = round(time.perf_counter() - started, 4)
                return result
            except Exception as exc:
                LOGGER.warning("Кеш %s повреждён и будет пересоздан: %s", cache_path, exc)

        extracted = self.extractor.extract(pdf_bytes, file_name)
        header: HeaderInfo = extracted["header"]
        raw_fields: list[ExtractedField] = extracted["raw_fields"]
        developer, objects, project, collections = self.builder.build(
            file_name=file_name,
            sha256=digest,
            header=header,
            fields=raw_fields,
            apartments=extracted["apartments"],
            non_residential=extracted["non_residential"],
            common_property=extracted["common_property"],
            engineering_equipment=extracted["engineering_equipment"],
            raw_table_rows=extracted["raw_table_rows"],
        )
        index = FieldIndex(raw_fields)
        checks, score = self.quality_engine.run(project, objects, index)
        project.quality_score = score
        elapsed = round(time.perf_counter() - started, 4)
        diagnostics = Diagnostics(
            file_name=file_name,
            sha256=digest,
            page_count=extracted["page_count"],
            text_pages=extracted["text_pages"],
            ocr_pages=extracted["ocr_pages"],
            scanned_page_candidates=extracted["scanned_candidates"],
            raw_field_count=len(raw_fields),
            filled_field_count=sum(bool(field.value_text) for field in raw_fields),
            raw_table_row_count=len(extracted["raw_table_rows"]),
            apartment_rows=len(extracted["apartments"]),
            non_residential_rows=len(extracted["non_residential"]),
            common_property_rows=len(extracted["common_property"]),
            engineering_equipment_rows=len(extracted["engineering_equipment"]),
            elapsed_seconds=elapsed,
            warnings=extracted["warnings"],
        )
        result = DeclarationResult(
            header=header,
            developer=developer,
            project=project,
            objects=objects,
            collections=collections,
            raw_fields=raw_fields,
            apartments=extracted["apartments"],
            non_residential=extracted["non_residential"],
            non_residential_parts=extracted["non_residential_parts"],
            common_property=extracted["common_property"],
            engineering_equipment=extracted["engineering_equipment"],
            raw_table_rows=extracted["raw_table_rows"],
            quality_checks=checks,
            diagnostics=diagnostics,
        )

        if self.config.use_cache and cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = result.to_dict(include_raw_fields=True, include_units=True)
            payload["raw_table_rows"] = result.raw_table_rows
            with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, default=str)
        return result

    @staticmethod
    def _result_from_payload(payload: Mapping[str, Any]) -> DeclarationResult:
        diagnostics = Diagnostics.model_validate(payload["diagnostics"])
        return DeclarationResult(
            header=HeaderInfo.model_validate(payload["header"]),
            developer=DeveloperInfo.model_validate(payload["developer"]),
            project=ProjectSummary.model_validate(payload["project"]),
            objects=[ObjectSummary.model_validate(item) for item in payload.get("objects", [])],
            collections=payload.get("collections", {}),
            raw_fields=[ExtractedField.model_validate(item) for item in payload.get("raw_fields", [])],
            apartments=payload.get("apartments", []),
            non_residential=payload.get("non_residential", []),
            non_residential_parts=payload.get("non_residential_parts", []),
            common_property=payload.get("common_property", []),
            engineering_equipment=payload.get("engineering_equipment", []),
            raw_table_rows=payload.get("raw_table_rows", []),
            quality_checks=[QualityCheck.model_validate(item) for item in payload.get("quality_checks", [])],
            diagnostics=diagnostics,
        )

    def run(self, input_path: str | Path, output_dir: Optional[str | Path] = None) -> list[DeclarationResult]:
        output_root = Path(output_dir) if output_dir is not None else self.config.output_dir
        output_root.mkdir(parents=True, exist_ok=True)
        pdfs = list(iter_input_pdfs(input_path))
        if not pdfs:
            raise FileNotFoundError(f"В {input_path} не найдено PDF")

        batch_started = time.perf_counter()
        results: list[DeclarationResult] = []
        failures: list[dict[str, Any]] = []

        def process(item: InputPdf) -> DeclarationResult:
            result = self.parse_bytes(item.pdf_bytes, item.file_name)
            result.save(output_root, self.config)
            return result

        if self.config.max_workers == 1 or len(pdfs) == 1:
            for index, item in enumerate(pdfs, start=1):
                LOGGER.info("[%s/%s] %s", index, len(pdfs), item.file_name)
                try:
                    results.append(process(item))
                except Exception as exc:
                    failures.append(
                        {
                            "file_name": item.file_name,
                            "source": item.source,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    LOGGER.exception("Ошибка при обработке %s", item.file_name)
        else:
            workers = min(self.config.max_workers, len(pdfs))
            LOGGER.info("Параллельная обработка: %s PDF, workers=%s", len(pdfs), workers)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="declaration") as executor:
                future_to_item = {executor.submit(process, item): item for item in pdfs}
                completed = 0
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    completed += 1
                    try:
                        results.append(future.result())
                        LOGGER.info("[%s/%s] готово: %s", completed, len(pdfs), item.file_name)
                    except Exception as exc:
                        failures.append(
                            {
                                "file_name": item.file_name,
                                "source": item.source,
                                "error": str(exc),
                                "traceback": "".join(traceback.format_exception(exc)),
                            }
                        )
                        LOGGER.exception("Ошибка при обработке %s", item.file_name)

        # Детерминированный порядок независимо от числа workers.
        results.sort(key=lambda result: result.project.file_name.lower())

        if results:
            pd.concat([result.project_frame() for result in results], ignore_index=True).to_csv(
                output_root / "dataset_projects.csv", index=False, encoding="utf-8-sig"
            )
            object_frames = [
                result.objects_frame().assign(file_name=result.project.file_name)
                for result in results
                if not result.objects_frame().empty
            ]
            if object_frames:
                pd.concat(object_frames, ignore_index=True).to_csv(
                    output_root / "dataset_objects.csv", index=False, encoding="utf-8-sig"
                )
            field_frames = [
                result.raw_fields_frame().assign(file_name=result.project.file_name)
                for result in results
                if result.raw_fields
            ]
            if field_frames:
                pd.concat(field_frames, ignore_index=True).to_csv(
                    output_root / "dataset_official_fields_long.csv", index=False, encoding="utf-8-sig"
                )
        if failures:
            (output_root / "failures.json").write_text(json_dumps(failures), encoding="utf-8")

        manifest = {
            "parser_version": PARSER_VERSION,
            "input": str(input_path),
            "output": str(output_root),
            "documents_found": len(pdfs),
            "documents_succeeded": len(results),
            "documents_failed": len(failures),
            "elapsed_seconds": round(time.perf_counter() - batch_started, 4),
            "max_workers": self.config.max_workers,
            "files": [
                {
                    "file_name": result.project.file_name,
                    "sha256": result.project.sha256,
                    "objects": result.project.object_count_parsed,
                    "quality_score": result.project.quality_score,
                    "raw_fields": result.project.raw_field_count,
                    "filled_fields": result.project.filled_field_count,
                }
                for result in results
            ],
            "failures": failures,
        }
        (output_root / "run_manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
        return results


# -----------------------------------------------------------------------------
# ML-калибровка и RLV
# -----------------------------------------------------------------------------

def build_ml_dataset(results: Sequence[DeclarationResult], *, level: Literal["object", "project"] = "object") -> pd.DataFrame:
    if level == "project":
        return pd.concat([result.project_frame() for result in results], ignore_index=True) if results else pd.DataFrame()
    frames = []
    for result in results:
        frame = result.objects_frame()
        if frame.empty:
            continue
        frame.insert(0, "file_name", result.project.file_name)
        frame.insert(1, "declaration_date", result.project.declaration_date)
        frame["developer_name"] = result.project.developer_name
        frame["land_area_sqm"] = result.project.land_area_sqm
        frame["floor_area_ratio"] = result.project.floor_area_ratio
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class CostCalibrator:
    """Безопасная оболочка над XGBoost/SHAP с проверкой размера выборки."""

    def __init__(self, min_rows: int = 30, random_state: int = 42):
        self.min_rows = min_rows
        self.random_state = random_state
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.metrics: dict[str, Any] = {}
        self.shap_importance_: Optional[pd.DataFrame] = None

    def fit(self, data: pd.DataFrame, target: str = "construction_cost_per_gross_sqm") -> "CostCalibrator":
        if target not in data.columns:
            raise KeyError(f"В датасете нет target={target}")
        numeric_candidates = [
            "max_floors",
            "gross_area_sqm",
            "saleable_area_sqm",
            "saleable_efficiency_pct",
            "apartment_count_declared",
            "non_residential_count_declared",
            "passenger_lifts",
            "median_ceiling_height_m",
            "land_area_sqm",
            "floor_area_ratio",
            "utility_connection_fees_rub",
            "construction_duration_months",
        ]
        available = [column for column in numeric_candidates if column in data.columns]
        clean = data.dropna(subset=[target]).copy()
        if len(clean) < self.min_rows:
            raise ValueError(
                f"Для ML-калибровки нужно минимум {self.min_rows} наблюдений; доступно {len(clean)}. "
                "На малой выборке используйте эмпирические медианы и мастер-коэффициенты."
            )
        X = clean[available].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        y = pd.to_numeric(clean[target], errors="coerce")
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]
        self.feature_columns = available

        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.metrics import mean_absolute_error, r2_score

        try:
            import xgboost as xgb

            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.5,
                objective="reg:squarederror",
                random_state=self.random_state,
                n_jobs=max(1, min(4, os.cpu_count() or 1)),
            )
        except ImportError:  # pragma: no cover
            from sklearn.ensemble import HistGradientBoostingRegressor

            model = HistGradientBoostingRegressor(random_state=self.random_state)

        folds = min(5, max(2, len(X) // 8))
        cv = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        mae_scores = -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=cv)
        r2_scores = cross_val_score(model, X, y, scoring="r2", cv=cv)
        model.fit(X, y)
        predictions = model.predict(X)
        self.model = model
        self.metrics = {
            "rows": len(X),
            "features": available,
            "cv_folds": folds,
            "cv_mae_mean": float(np.mean(mae_scores)),
            "cv_mae_std": float(np.std(mae_scores)),
            "cv_r2_mean": float(np.mean(r2_scores)),
            "train_mae": float(mean_absolute_error(y, predictions)),
            "train_r2": float(r2_score(y, predictions)),
        }

        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            importance = np.abs(np.asarray(shap_values)).mean(axis=0)
            self.shap_importance_ = pd.DataFrame(
                {"feature": available, "mean_abs_shap": importance}
            ).sort_values("mean_abs_shap", ascending=False)
        except Exception:
            self.shap_importance_ = None
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Сначала вызовите fit()")
        X = data[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return np.asarray(self.model.predict(X))


class RLVScenario(StrictModel):
    sell_price_rub_per_sqm: float
    target_profit_rub_per_sqm: float
    sellable_area_sqm: float
    max_floors: int
    ceiling_height_m: float
    base_cost_rub_per_sqm: float = 167_200.0
    deflator: float = 1.0
    additional_cost_rub_per_sqm: float = 0.0


def estimate_rlv(scenario: RLVScenario) -> dict[str, Any]:
    floor_label, kfl = floor_band(scenario.max_floors)
    class_label, kcl, confidence = class_proxy(scenario.ceiling_height_m)
    if kfl is None or kcl is None:
        raise ValueError("Не удалось определить K_fl/K_cl")
    construction_cost = (
        scenario.base_cost_rub_per_sqm * kfl * kcl * scenario.deflator
        + scenario.additional_cost_rub_per_sqm
    )
    residual_per_sqm = scenario.sell_price_rub_per_sqm - construction_cost - scenario.target_profit_rub_per_sqm
    return {
        "floor_zone": floor_label,
        "k_fl": kfl,
        "class_proxy": class_label,
        "class_proxy_confidence": confidence,
        "k_cl": kcl,
        "parametric_construction_cost_rub_per_sqm": round(construction_cost, 2),
        "residual_land_value_rub_per_sellable_sqm": round(residual_per_sqm, 2),
        "residual_land_value_rub": round(residual_per_sqm * scenario.sellable_area_sqm, 2),
    }


def rlv_sensitivity(
    base_scenario: RLVScenario,
    *,
    price_changes_pct: Sequence[float] = (-10, -5, 0, 5, 10),
    cost_changes_pct: Sequence[float] = (-10, -5, 0, 5, 10),
) -> pd.DataFrame:
    rows = []
    base = estimate_rlv(base_scenario)
    base_cost = float(base["parametric_construction_cost_rub_per_sqm"])
    for price_delta in price_changes_pct:
        for cost_delta in cost_changes_pct:
            sell_price = base_scenario.sell_price_rub_per_sqm * (1 + price_delta / 100.0)
            cost = base_cost * (1 + cost_delta / 100.0)
            residual = (
                sell_price - cost - base_scenario.target_profit_rub_per_sqm
            ) * base_scenario.sellable_area_sqm
            rows.append(
                {
                    "sell_price_change_pct": price_delta,
                    "construction_cost_change_pct": cost_delta,
                    "sell_price_rub_per_sqm": round(sell_price, 2),
                    "construction_cost_rub_per_sqm": round(cost, 2),
                    "rlv_rub": round(residual, 2),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Layout-aware parser проектных деклараций")
    parser.add_argument("input", help="PDF, ZIP или папка с PDF")
    parser.add_argument("--output", default="declaration_output", help="Каталог результатов")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--ocr", action="store_true", help="Включить OCR для скан-копий")
    parser.add_argument("--no-ocr", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-tables", action="store_true", help="Не разбирать большие таблицы")
    parser.add_argument("--raw-rows", action="store_true", help="Сохранять сырые строки таблиц (увеличивает объём)")
    parser.add_argument("--no-raw-rows", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1, help="Параллельные PDF (по умолчанию 1)")
    parser.add_argument("--base-cost", type=float, default=167200.0, help="C_base, руб./м²")
    parser.add_argument("--deflator", type=float, default=1.0, help="D(t)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    config = PipelineConfig(
        output_dir=Path(args.output),
        use_cache=not args.no_cache,
        enable_ocr=bool(args.ocr and not args.no_ocr),
        parse_tables=not args.no_tables,
        store_raw_table_rows=bool(args.raw_rows and not args.no_raw_rows),
        max_workers=args.workers,
        base_cost_rub_per_sqm=args.base_cost,
        deflator=args.deflator,
    )
    pipeline = DeclarationPipeline(config)
    results = pipeline.run(args.input, args.output)
    LOGGER.info("Готово. Успешно обработано: %s", len(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
