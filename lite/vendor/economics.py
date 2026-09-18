"""Derived construction economics and finish classification.

The official project declaration discloses a *planned construction cost*
(field 18.1.1), gross object area (9.2.21), saleable residential/non-residential
areas (9.3.x) and, in many declarations, detailed non-residential units.
It does not disclose one universally standardized "pure construction cost".

This module therefore keeps the disclosed investment/planned value untouched
and builds a transparent bridge to a derived construction cost:

    construction cost = disclosed investment cost
                        - disclosed non-construction costs
                        - configurable additional non-construction share

All assumptions are emitted alongside the result.  The default additional
share is zero, so the system never invents an undisclosed deduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


FINISH_TYPES: tuple[str, ...] = (
    "Без отделки",
    "Предчистовая",
    "Чистовая",
    "Смешанная",
    "Не определено",
)

PHYSICAL_FINISH_TYPES: tuple[str, ...] = ("Без отделки", "Предчистовая", "Чистовая")

_FINISH_ALIASES: dict[str, str] = {
    "б/о": "Без отделки",
    "без отделки": "Без отделки",
    "черновая": "Без отделки",
    "черновая отделка": "Без отделки",
    "предчистовая": "Предчистовая",
    "white box": "Предчистовая",
    "вайт бокс": "Предчистовая",
    "чистовая": "Чистовая",
    "чистовая отделка": "Чистовая",
    "смешанная": "Смешанная",
    "не определено": "Не определено",
}

def normalize_finish_type(value: Any) -> str:
    """Return one of :data:`FINISH_TYPES` without inventing a category."""
    text = _usable_text(value)
    if not text:
        return "Не определено"
    normalized = re.sub(r"\s+", " ", text).strip().lower().replace("ё", "е")
    if normalized in _FINISH_ALIASES:
        return _FINISH_ALIASES[normalized]
    for label in FINISH_TYPES:
        if normalized == label.lower().replace("ё", "е"):
            return label
    classified = classify_finish_texts([text])
    return str(classified.get("finish_type") or "Не определено")

# Order matters: phrases such as "под чистовую отделку" must be classified as
# pre-finish before the generic word "чистовая" is considered.
_FINISH_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "Предчистовая": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bпредчист(?:овая|овой|овую|ов)\b",
            r"\bwhite\s*box\b",
            r"\bвайт\s*бокс\b",
            r"\bпод\s+чистовую\s+отделку\b",
            r"\bподготовк\w*\s+под\s+чистовую\b",
            r"\bулучшенн\w*\s+чернов\w*\b",
        )
    ),
    "Без отделки": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bбез\s+(?:внутренней\s+)?отделк\w*\b",
            r"(?<![а-яa-z0-9])б\s*/\s*о(?![а-яa-z0-9])",
            r"\bчернов(?:ая|ой|ую|ом)\s+отделк\w*\b",
            r"\bпод\s+самоотделк\w*\b",
            r"\bсвободн\w*\s+планировк\w*\s+без\s+отделк\w*\b",
        )
    ),
    "Чистовая": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bчистов\w*\s+отделк\w*\b",
            r"\bс\s+(?:полной\s+|готовой\s+|чистовой\s+)?отделк\w*\b",
            r"\bотделк\w*\s+под\s+ключ\b",
            r"\bполная\s+отделк\w*\b",
            r"\bготовая\s+отделк\w*\b",
        )
    ),
}

_PARKING_RE = re.compile(r"машино[- ]?мест|паркинг|автостоян|гараж", re.IGNORECASE)
_STORAGE_RE = re.compile(r"кладов|келлер|storage", re.IGNORECASE)
_COMMERCIAL_RE = re.compile(
    r"коммерческ|магазин|офис|торгов|помещени[ея]\s+свободного\s+назначения|\bпсн\b|"
    r"общественно[- ]?делов|сервис|кафе|ресторан|аптек|салон",
    re.IGNORECASE,
)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _nonnegative(value: Any) -> float:
    number = _number(value)
    return max(0.0, number) if number is not None else 0.0


def _safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    n = _number(numerator)
    d = _number(denominator)
    if n is None or d is None or d <= 0:
        return None
    return n / d


def _usable_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip(" \t\r\n\"'«»")
    if not text or text.lower() in {"nan", "none", "null", "не указано", "отсутствует"}:
        return None
    return text


@dataclass(frozen=True)
class ConstructionCostConfig:
    """Transparent bridge from declaration investment cost to construction cost."""

    deduct_utility_fees: bool = True
    additional_nonconstruction_share_pct: float = 0.0
    minimum_construction_share_pct: float = 50.0

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "ConstructionCostConfig":
        raw = dict(value or {})
        try:
            share = float(raw.get("additional_nonconstruction_share_pct", 0.0))
        except (TypeError, ValueError):
            share = 0.0
        try:
            minimum = float(raw.get("minimum_construction_share_pct", 50.0))
        except (TypeError, ValueError):
            minimum = 50.0
        return cls(
            deduct_utility_fees=bool(raw.get("deduct_utility_fees", True)),
            additional_nonconstruction_share_pct=float(np.clip(share, 0.0, 45.0)),
            minimum_construction_share_pct=float(np.clip(minimum, 25.0, 100.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deduct_utility_fees": self.deduct_utility_fees,
            "additional_nonconstruction_share_pct": self.additional_nonconstruction_share_pct,
            "minimum_construction_share_pct": self.minimum_construction_share_pct,
        }


def classify_finish_texts(texts: Iterable[Any]) -> dict[str, Any]:
    """Classify finish type and preserve compact evidence.

    Only explicit phrases are accepted.  If the declaration does not contain
    them, the result is ``Не определено`` rather than a guessed class.
    """
    hits: dict[str, list[str]] = {key: [] for key in _FINISH_PATTERNS}
    for raw in texts:
        text = _usable_text(raw)
        if not text:
            continue
        normalized = re.sub(r"\s+", " ", text)
        # Remove pre-finish phrases before evaluating generic clean-finish
        # patterns so "под чистовую отделку" is not double-counted.
        for label, patterns in _FINISH_PATTERNS.items():
            candidate = normalized
            if label == "Чистовая":
                for prefinish_pattern in _FINISH_PATTERNS["Предчистовая"]:
                    candidate = prefinish_pattern.sub(" ", candidate)
            for pattern in patterns:
                match = pattern.search(candidate)
                if match:
                    # Evidence is taken from the original normalized text.  For
                    # clean-finish matching the masked candidate can shift the
                    # exact context, therefore preserve a compact full sentence
                    # when mapping the span back is ambiguous.
                    if candidate == normalized:
                        excerpt_start = max(0, match.start() - 90)
                        excerpt_end = min(len(normalized), match.end() + 130)
                        excerpt = normalized[excerpt_start:excerpt_end].strip()
                    else:
                        excerpt = normalized[:1200]
                    if excerpt not in hits[label]:
                        hits[label].append(excerpt)
                    break

    active = [label for label, values in hits.items() if values]
    if not active:
        return {
            "finish_type": "Не определено",
            "finish_confidence": 0.0,
            "finish_evidence": None,
            "finish_source": "В декларации не найдено явного указания",
        }
    if len(active) > 1:
        evidence = " | ".join(
            f"{label}: {values[0]}" for label, values in hits.items() if values
        )
        return {
            "finish_type": "Смешанная",
            "finish_confidence": 0.82,
            "finish_evidence": evidence[:1200],
            "finish_source": "Несколько явных вариантов в тексте декларации",
        }
    label = active[0]
    evidence = hits[label][0]
    # Exact explicit wording is strong evidence, but not an official structured
    # field in every version of the declaration form.
    return {
        "finish_type": label,
        "finish_confidence": 0.9,
        "finish_evidence": evidence[:1200],
        "finish_source": "Явное текстовое указание в декларации",
    }


def classify_nonresidential_purpose(value: Any) -> str:
    text = (_usable_text(value) or "").lower()
    if _PARKING_RE.search(text):
        return "parking"
    if _STORAGE_RE.search(text):
        return "storage"
    if _COMMERCIAL_RE.search(text):
        return "commercial"
    return "other"


def derive_saleable_components(
    *,
    gross_area_sqm: Any,
    residential_area_sqm: Any,
    non_residential_area_sqm: Any,
    saleable_area_sqm: Any,
    non_residential_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Build a transparent gross-to-saleable bridge.

    Two saleable denominators are preserved:

    * ``official_saleable_area_sqm`` — official 9.3.3 (all residential and
      non-residential premises);
    * ``saleable_area_for_cost_sqm`` — the business denominator requested for
      cost analytics: housing + commercial premises + parking.

    The second denominator is used only when table 15.3 provides sufficiently
    complete component evidence.  Otherwise the official 9.3.3 total is used
    as an explicit fallback, so the dashboard never silently understates area.
    """
    gross = _number(gross_area_sqm)
    housing = _number(residential_area_sqm)
    nonres_official = _number(non_residential_area_sqm)
    official_saleable = _number(saleable_area_sqm)

    if official_saleable is None and housing is not None and nonres_official is not None:
        official_saleable = housing + nonres_official
    if housing is None and official_saleable is not None and nonres_official is not None:
        housing = max(0.0, official_saleable - nonres_official)
    if nonres_official is None and official_saleable is not None and housing is not None:
        nonres_official = max(0.0, official_saleable - housing)

    buckets = {"commercial": 0.0, "storage": 0.0, "parking": 0.0, "other": 0.0}
    rows_used = 0
    for row in non_residential_rows or []:
        area = _number(row.get("area_sqm"))
        if area is None or area <= 0:
            continue
        purpose = " ".join(
            str(row.get(key) or "") for key in ("purpose", "unit_type", "name", "part_name")
        )
        buckets[classify_nonresidential_purpose(purpose)] += area
        rows_used += 1

    parsed_total_raw = sum(buckets.values())
    parsed_total = parsed_total_raw
    component_source = "Официальные поля 9.3"
    component_confidence = 0.92 if official_saleable is not None else 0.35
    component_coverage = None
    if nonres_official is not None and nonres_official >= 0 and parsed_total > 0:
        # Preserve a discrepancy; rescaling an incomplete/duplicated table would invent areas.
        component_coverage = parsed_total / nonres_official if nonres_official > 0 else None
        residual = max(0.0, nonres_official - parsed_total)
        buckets["other"] += residual
        component_source = "Официальные поля 9.3 + детализация таблицы 15.3"
        component_confidence = .92 if component_coverage is not None and .98<=component_coverage<=1.02 else .35
    elif nonres_official is not None:
        # Keep unclassified official non-residential area visible.
        buckets["other"] = max(0.0, nonres_official)
        component_coverage = 0.0

    requested_candidate = None
    if housing is not None:
        requested_candidate = max(0.0, housing) + buckets["commercial"] + buckets["parking"]

    housing_only_confirmed = (housing is not None and housing>=0 and nonres_official==0 and parsed_total==0 and official_saleable is not None and abs(housing-official_saleable)<=max(.01,official_saleable*.0001))
    components_reconcile = (housing is not None and nonres_official is not None and official_saleable is not None and abs(housing+nonres_official-official_saleable)<=max(.01,official_saleable*.02))
    use_component_denominator = housing_only_confirmed or (
        components_reconcile
        and
        requested_candidate is not None
        and rows_used > 0
        and component_coverage is not None
        and .98 <= component_coverage <= 1.02
    )
    if use_component_denominator:
        saleable_for_cost = requested_candidate
        denominator_source = "Жильё 9.3.1 + коммерческие помещения и паркинг из таблицы 15.3"
        denominator_confidence = float(np.clip(0.62 + 0.33 * component_coverage, 0.62, 0.95))
    else:
        saleable_for_cost = official_saleable
        denominator_source = (
            "Fallback: официальная сумма 9.3.3; состав КП/паркинга недостаточно детализирован"
            if official_saleable is not None
            else "Недостаточно данных для продаваемой площади"
        )
        denominator_confidence = 0.55 if official_saleable is not None else 0.0

    requested_ratio = _safe_div(gross, saleable_for_cost)
    official_ratio = _safe_div(gross, official_saleable)
    requested_efficiency = _safe_div(saleable_for_cost, gross)
    official_efficiency = _safe_div(official_saleable, gross)
    non_saleable = None
    if gross is not None and saleable_for_cost is not None:
        non_saleable = max(0.0, gross - saleable_for_cost)

    return {
        "gross_construction_area_sqm": gross,
        "official_saleable_area_sqm": official_saleable,
        "saleable_housing_area_sqm": housing,
        "saleable_commercial_area_sqm": round(buckets["commercial"], 3),
        "saleable_storage_area_sqm": round(buckets["storage"], 3),
        "saleable_parking_area_sqm": round(buckets["parking"], 3),
        "saleable_other_nonres_area_sqm": round(buckets["other"], 3),
        "saleable_nonresidential_area_sqm": nonres_official,
        "saleable_total_area_sqm": official_saleable,
        "saleable_area_for_cost_sqm": round(saleable_for_cost, 3) if saleable_for_cost is not None else None,
        "saleable_area_for_cost_source": denominator_source,
        "saleable_area_for_cost_confidence": round(denominator_confidence, 3),
        "nonresidential_component_coverage_pct": (
            round(component_coverage * 100.0, 2) if component_coverage is not None else None
        ),
        "gross_to_saleable_ratio": round(requested_ratio, 4) if requested_ratio is not None else None,
        "official_gross_to_saleable_ratio": round(official_ratio, 4) if official_ratio is not None else None,
        "saleable_efficiency_pct": round(requested_efficiency * 100.0, 2) if requested_efficiency is not None else None,
        "official_saleable_efficiency_pct": round(official_efficiency * 100.0, 2) if official_efficiency is not None else None,
        "non_saleable_area_sqm": round(non_saleable, 3) if non_saleable is not None else None,
        "saleable_components_source": component_source,
        "saleable_components_confidence": round(component_confidence, 3),
        "nonresidential_rows_used": rows_used,
        "nonresidential_table_consistent": component_coverage is not None and .98<=component_coverage<=1.02,
        "business_area_confirmed": use_component_denominator,
    }

def derive_construction_cost(
    *,
    investment_cost_rub: Any,
    gross_area_sqm: Any,
    saleable_area_sqm: Any,
    official_saleable_area_sqm: Any = None,
    utility_connection_fees_rub: Any = None,
    social_infrastructure_cost_rub: Any = None,
    territory_development_payments_rub: Any = None,
    land_cost_rub: Any = None,
    other_disclosed_nonconstruction_cost_rub: Any = None,
    config: Optional[ConstructionCostConfig] = None,
) -> dict[str, Any]:
    """Derive pure construction cost without overwriting source evidence."""
    cfg = config or ConstructionCostConfig()
    investment = _number(investment_cost_rub)
    gross = _number(gross_area_sqm)
    saleable = _number(saleable_area_sqm)
    official_saleable = _number(official_saleable_area_sqm)
    utility = _nonnegative(utility_connection_fees_rub) if cfg.deduct_utility_fees else 0.0
    components = {
        "Подключение к сетям": utility,
        "Социальная инфраструктура": _nonnegative(social_infrastructure_cost_rub),
        "Платежи развития территории": _nonnegative(territory_development_payments_rub),
        "Земля": _nonnegative(land_cost_rub),
        "Иные раскрытые внестроительные затраты": _nonnegative(other_disclosed_nonconstruction_cost_rub),
    }
    disclosed = sum(components.values())
    if investment is None or investment <= 0:
        return {
            "investment_cost_rub": investment,
            "investment_cost_per_gross_sqm": None,
            "investment_cost_per_saleable_sqm": None,
            "investment_cost_per_official_saleable_sqm": None,
            "disclosed_nonconstruction_cost_rub": disclosed or None,
            "additional_nonconstruction_cost_rub": None,
            "construction_cost_rub": None,
            "construction_cost_per_gross_sqm": None,
            "construction_cost_per_saleable_sqm": None,
            "construction_cost_per_official_saleable_sqm": None,
            "construction_share_of_investment_pct": None,
            "construction_cost_bridge_method": "Нет инвестиционной стоимости 18.1.1",
            "construction_cost_confidence": 0.0,
            "construction_cost_confidence_label": "Недостаточно данных",
            "construction_cost_components_json": json.dumps(components, ensure_ascii=False),
        }

    disclosed = min(disclosed, investment)
    base_after_disclosed = max(0.0, investment - disclosed)
    additional = base_after_disclosed * cfg.additional_nonconstruction_share_pct / 100.0
    derived = max(0.0, base_after_disclosed - additional)
    minimum = investment * cfg.minimum_construction_share_pct / 100.0
    if derived < minimum:
        derived = minimum
        additional = max(0.0, investment - disclosed - derived)

    method_parts = ["Инвестиционная стоимость 18.1.1"]
    if disclosed > 0:
        method_parts.append("минус раскрытые внестроительные затраты")
    if cfg.additional_nonconstruction_share_pct > 0:
        method_parts.append(f"минус {cfg.additional_nonconstruction_share_pct:g}% прочих инвестиционных затрат")
    if disclosed <= 0 and cfg.additional_nonconstruction_share_pct <= 0:
        method_parts.append("без дополнительных вычетов")
    confidence = 0.42
    if disclosed > 0:
        confidence = 0.58
    if cfg.additional_nonconstruction_share_pct > 0:
        confidence = 0.64
    label = "Средняя" if confidence >= 0.55 else "Ограниченная"

    return {
        "investment_cost_rub": round(investment, 2),
        "investment_cost_per_gross_sqm": round(investment / gross, 2) if gross and gross > 0 else None,
        "investment_cost_per_saleable_sqm": round(investment / saleable, 2) if saleable and saleable > 0 else None,
        "investment_cost_per_official_saleable_sqm": round(investment / official_saleable, 2) if official_saleable and official_saleable > 0 else None,
        "disclosed_nonconstruction_cost_rub": round(disclosed, 2),
        "additional_nonconstruction_cost_rub": round(additional, 2),
        "construction_cost_rub": round(derived, 2),
        "construction_cost_per_gross_sqm": round(derived / gross, 2) if gross and gross > 0 else None,
        "construction_cost_per_saleable_sqm": round(derived / saleable, 2) if saleable and saleable > 0 else None,
        "construction_cost_per_official_saleable_sqm": round(derived / official_saleable, 2) if official_saleable and official_saleable > 0 else None,
        "construction_share_of_investment_pct": round(derived / investment * 100.0, 2),
        "investment_to_construction_delta_rub": round(investment - derived, 2),
        "construction_cost_bridge_method": " ".join(method_parts),
        "construction_cost_confidence": confidence,
        "construction_cost_confidence_label": label,
        "construction_cost_components_json": json.dumps(components, ensure_ascii=False),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=4096)
def artifact_details(artifact_path: str, object_no: Optional[int]) -> dict[str, Any]:
    """Read optional detailed evidence for old registries without reparsing PDFs."""
    root = Path(str(artifact_path or ""))
    if not root.exists():
        return {"non_residential": [], "finish": classify_finish_texts([])}

    nonres = _read_csv(root / "non_residential.csv")
    if not nonres.empty and object_no is not None and "object_no" in nonres.columns:
        numbers = pd.to_numeric(nonres["object_no"], errors="coerce")
        nonres = nonres[numbers.eq(int(object_no))]
    nonres_rows = nonres.replace({np.nan: None}).to_dict("records") if not nonres.empty else []

    texts: list[str] = []
    raw = _read_csv(root / "raw_fields.csv")
    if not raw.empty:
        if object_no is not None and "object_no" in raw.columns:
            numbers = pd.to_numeric(raw["object_no"], errors="coerce")
            # Include global fields and exact object fields, but exclude other objects.
            raw = raw[numbers.isna() | numbers.eq(int(object_no))]
        for _, row in raw.iterrows():
            label = _usable_text(row.get("label"))
            value = _usable_text(row.get("value_text"))
            if label or value:
                texts.append(f"{label or ''}: {value or ''}")

    # declaration_full.json is a fallback when CSV exports were disabled.
    if not texts:
        full_path = root / "declaration_full.json"
        if full_path.exists():
            try:
                payload = json.loads(full_path.read_text(encoding="utf-8"))
                fields = payload.get("raw_fields", []) or []
                for field in fields:
                    field_no = field.get("object_no")
                    if object_no is not None and field_no not in (None, object_no):
                        continue
                    texts.append(f"{field.get('label') or ''}: {field.get('value_text') or ''}")
                if not nonres_rows:
                    rows = payload.get("non_residential", []) or []
                    nonres_rows = [
                        row for row in rows
                        if object_no is None or row.get("object_no") == object_no
                    ]
            except Exception:
                pass

    return {
        "non_residential": nonres_rows,
        "finish": classify_finish_texts(texts),
    }


def enrich_object_record(
    record: Mapping[str, Any],
    *,
    config: Optional[ConstructionCostConfig] = None,
    artifact_path: Optional[str] = None,
    non_residential_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    finish_texts: Optional[Iterable[Any]] = None,
    finish_override: Optional[str] = None,
) -> dict[str, Any]:
    """Return an enriched copy and keep all legacy source columns."""
    result = dict(record)
    details: dict[str, Any] = {}
    path = artifact_path or result.get("artifact_path")
    object_no_raw = result.get("object_no")
    try:
        object_no = int(object_no_raw) if object_no_raw is not None else None
    except (TypeError, ValueError):
        object_no = None
    if path and (non_residential_rows is None or finish_texts is None):
        details = artifact_details(str(path), object_no)

    rows = list(non_residential_rows) if non_residential_rows is not None else details.get("non_residential", [])
    area = derive_saleable_components(
        gross_area_sqm=result.get("gross_area_sqm"),
        residential_area_sqm=result.get("residential_area_sqm"),
        non_residential_area_sqm=result.get("non_residential_area_sqm"),
        saleable_area_sqm=result.get("saleable_area_sqm"),
        non_residential_rows=rows,
    )
    result.update(area)
    # Preserve canonical existing names, but anchor them to official / enriched totals.
    if area.get("gross_construction_area_sqm") is not None:
        result["gross_area_sqm"] = area["gross_construction_area_sqm"]
    if area.get("saleable_total_area_sqm") is not None:
        # Keep the official 9.3.3 total under the legacy/canonical column.
        result["saleable_area_sqm"] = area["saleable_total_area_sqm"]

    cost = derive_construction_cost(
        investment_cost_rub=result.get("planned_construction_cost_rub") or result.get("investment_cost_rub"),
        gross_area_sqm=result.get("gross_area_sqm"),
        saleable_area_sqm=area.get("saleable_area_for_cost_sqm") or result.get("saleable_area_sqm"),
        official_saleable_area_sqm=result.get("saleable_area_sqm"),
        utility_connection_fees_rub=result.get("utility_connection_fees_rub"),
        social_infrastructure_cost_rub=result.get("social_infrastructure_cost_rub"),
        territory_development_payments_rub=result.get("territory_development_payments_rub"),
        land_cost_rub=result.get("land_cost_rub"),
        other_disclosed_nonconstruction_cost_rub=result.get("other_disclosed_nonconstruction_cost_rub"),
        config=config,
    )
    # Preserve legacy calculated values under explicit source names before replacing
    # the ambiguous "construction_cost" fields with the derived construction cost.
    if result.get("construction_cost_per_gross_sqm") is not None:
        result.setdefault("legacy_declared_cost_per_gross_sqm", result.get("construction_cost_per_gross_sqm"))
    if result.get("construction_cost_per_saleable_sqm") is not None:
        result.setdefault("legacy_declared_cost_per_saleable_sqm", result.get("construction_cost_per_saleable_sqm"))
    result.update(cost)

    normalized_override = normalize_finish_type(finish_override) if finish_override is not None else "Не определено"
    if normalized_override in FINISH_TYPES and normalized_override != "Не определено":
        finish = {
            "finish_type": normalized_override,
            "finish_confidence": 1.0,
            "finish_evidence": "Ручное подтверждение аналитика",
            "finish_source": "Ручное подтверждение",
        }
    elif finish_texts is not None:
        finish = classify_finish_texts(finish_texts)
    else:
        finish = details.get("finish") or classify_finish_texts([])
    result.update(finish)
    return result


def enrich_objects_frame(
    frame: Optional[pd.DataFrame],
    *,
    config: Optional[ConstructionCostConfig] = None,
    finish_overrides: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=frame.columns if frame is not None else None)
    overrides = dict(finish_overrides or {})
    records = []
    for row in frame.to_dict("records"):
        uid = str(row.get("object_uid") or "")
        override_value = overrides.get(uid)
        if isinstance(override_value, Mapping):
            override = override_value.get("finish_type")
        else:
            override = override_value
        records.append(enrich_object_record(row, config=config, finish_override=override))
    return pd.DataFrame(records)


def aggregate_finish(values: Iterable[Any]) -> dict[str, Any]:
    labels = [str(value) for value in values if value and str(value) != "Не определено"]
    unique = list(dict.fromkeys(labels))
    if not unique:
        return {"finish_type": "Не определено", "finish_confidence": 0.0}
    if len(unique) == 1:
        return {"finish_type": unique[0], "finish_confidence": 0.9}
    return {"finish_type": "Смешанная", "finish_confidence": 0.82}


def enrich_projects_from_objects(
    projects: Optional[pd.DataFrame],
    objects: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if projects is None or projects.empty:
        return pd.DataFrame(columns=projects.columns if projects is not None else None)
    result = projects.copy()
    if objects is None or objects.empty or "sha256" not in result.columns or "sha256" not in objects.columns:
        return result
    rows = []
    for sha, group in objects.groupby("sha256", dropna=False):
        investment = pd.to_numeric(group.get("investment_cost_rub"), errors="coerce").sum(min_count=1)
        construction = pd.to_numeric(group.get("construction_cost_rub"), errors="coerce").sum(min_count=1)
        gross = pd.to_numeric(group.get("gross_area_sqm"), errors="coerce").sum(min_count=1)
        official_saleable = pd.to_numeric(group.get("saleable_area_sqm"), errors="coerce").sum(min_count=1)
        cost_saleable = pd.to_numeric(
            group.get("saleable_area_for_cost_sqm", group.get("saleable_area_sqm")), errors="coerce"
        ).sum(min_count=1)
        finish = aggregate_finish(group.get("finish_type", pd.Series(dtype=object)).tolist())
        rows.append({
            "sha256": sha,
            "investment_cost_rub": investment,
            "construction_cost_rub": construction,
            "investment_to_construction_delta_rub": investment - construction if pd.notna(investment) and pd.notna(construction) else np.nan,
            "investment_cost_per_gross_sqm": investment / gross if pd.notna(investment) and pd.notna(gross) and gross > 0 else np.nan,
            "investment_cost_per_saleable_sqm": investment / cost_saleable if pd.notna(investment) and pd.notna(cost_saleable) and cost_saleable > 0 else np.nan,
            "investment_cost_per_official_saleable_sqm": investment / official_saleable if pd.notna(investment) and pd.notna(official_saleable) and official_saleable > 0 else np.nan,
            "construction_cost_per_gross_sqm": construction / gross if pd.notna(construction) and pd.notna(gross) and gross > 0 else np.nan,
            "construction_cost_per_saleable_sqm": construction / cost_saleable if pd.notna(construction) and pd.notna(cost_saleable) and cost_saleable > 0 else np.nan,
            "construction_cost_per_official_saleable_sqm": construction / official_saleable if pd.notna(construction) and pd.notna(official_saleable) and official_saleable > 0 else np.nan,
            "saleable_area_for_cost_sqm": cost_saleable,
            "official_saleable_area_sqm": official_saleable,
            "gross_to_saleable_ratio": gross / cost_saleable if pd.notna(gross) and pd.notna(cost_saleable) and cost_saleable > 0 else np.nan,
            "official_gross_to_saleable_ratio": gross / official_saleable if pd.notna(gross) and pd.notna(official_saleable) and official_saleable > 0 else np.nan,
            "saleable_efficiency_pct": cost_saleable / gross * 100.0 if pd.notna(gross) and gross > 0 and pd.notna(cost_saleable) else np.nan,
            "official_saleable_efficiency_pct": official_saleable / gross * 100.0 if pd.notna(gross) and gross > 0 and pd.notna(official_saleable) else np.nan,
            "finish_type": finish["finish_type"],
            "finish_confidence": finish["finish_confidence"],
        })
    return result.merge(pd.DataFrame(rows), on="sha256", how="left", suffixes=("", "__derived")).pipe(_coalesce_derived_columns)

def _coalesce_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in list(result.columns):
        if not column.endswith("__derived"):
            continue
        base = column[:-9]
        if base in result.columns:
            result[base] = result[column].where(result[column].notna(), result[base])
            result = result.drop(columns=[column])
        else:
            result = result.rename(columns={column: base})
    return result
