from io import BytesIO
import json
import pandas as pd

COLUMNS = {
    "project": "Проект / ЖК", "name": "Объект / корпус", "object_no": "№ объекта",
    "developer": "Застройщик", "inn": "ИНН", "region": "Регион", "address": "Адрес",
    "declaration_number": "№ декларации", "declaration_date": "Дата декларации",
    "floors_max": "Этажей", "gross_area": "ОСП, м²", "saleable_area": "Продаваемая площадь, м²",
    "planned_cost": "Плановая стоимость, ₽", "cost_per_gross": "Стоимость / ОСП, ₽/м²",
    "cost_per_saleable": "Стоимость / продаваемая, ₽/м²", "saleable_share": "Доля продаваемой площади",
    "version": "Версия", "eligible": "Включён в сводные показатели", "has_ocr": "Использован OCR",
    "filename": "Файл", "status": "Полнота данных", "issues": "Примечания",
}


def frame(rows):
    return pd.DataFrame([{COLUMNS[k]: "; ".join(r.get(k, [])) if k == "issues" else r.get(k) for k in COLUMNS} for r in rows], columns=list(COLUMNS.values()))


def csv_bytes(rows):
    df = frame(rows)
    # Prevent spreadsheet formulas when opening exported user-supplied text.
    for col in df.columns:
        df[col] = df[col].map(lambda v: "'" + v if isinstance(v,str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def excel_bytes(rows, docs):
    buffer = BytesIO()
    selected = {r["sha"] for r in rows}
    document_rows, evidence = [], []
    for d in docs:
        if d["sha"] not in selected or not d["result"]:
            continue
        result = json.loads(d["result"])
        document_rows.append({"Файл": d["filename"], "SHA-256": d["sha"], "Страниц": result["pages"], "Страниц OCR": result["ocr_pages"],
                              "Дата декларации": result["header"].get("date_iso"), "Дата загрузки (служебная)": d["uploaded_at"],
                              "Парсер": result["parser_version"], "Примечания": "; ".join(result["warnings"])})
        for f in result["fields"]:
            evidence.append({"Файл":d["filename"], "Объект":f["object_no"], "Страница":f["page"], "Поле":f["code"],
                             "Название":f["label"], "Исходное значение":f["value"], "Метод":f["method"]})
    with pd.ExcelWriter(buffer, engine="xlsxwriter", engine_kwargs={"options":{"strings_to_formulas":False, "strings_to_urls":False}}) as writer:
        sheets = {"Объекты":frame(rows), "Документы":pd.DataFrame(document_rows), "Источники значений":pd.DataFrame(evidence)}
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes(1, 0)
            if len(df):
                ws.autofilter(0,0,len(df),len(df.columns)-1)
            number = writer.book.add_format({"num_format":"#,##0.00"})
            for i, col in enumerate(df.columns):
                ws.set_column(i,i, min(48,max(18,len(col)+2)), number if pd.api.types.is_numeric_dtype(df[col]) else None)
    return buffer.getvalue()
