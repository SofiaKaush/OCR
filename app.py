from __future__ import annotations
import hashlib
import json
from pathlib import Path

import fitz
import pandas as pd
import streamlit as st

from lite import __version__
from lite.export import frame, csv_bytes, excel_bytes
from lite.storage import data_dir, documents, enqueue, unpack_uploads, rows, summary, retry
from lite.worker import ensure_worker

st.set_page_config(page_title="Декларации · Стоимость и объекты", page_icon="◈", layout="wide")
st.markdown("""<style>
:root {--ink:#202b29;--muted:#707b75;--accent:#226a50;--line:#e1e5dd;}
.stApp {background:#f5f5ef;color:var(--ink)}
.block-container {max-width:1500px;padding:2.5rem 3.5rem 3rem}
header[data-testid="stHeader"] {background:transparent}
h1,h2,h3 {color:var(--ink);letter-spacing:-.035em!important}
h1 {font-size:2.9rem!important;font-weight:650!important;line-height:1.1!important}
h2 {font-size:1.55rem!important} h3 {font-size:1.15rem!important}
[data-testid="stSidebar"] {background:#eaece4}
[data-testid="stMetric"] {background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 22px;min-height:128px}
[data-testid="stMetricLabel"] {color:var(--muted);font-size:13px}
[data-testid="stMetricValue"] {font-size:1.85rem;letter-spacing:-.04em}
[data-testid="stFileUploader"] {background:#fff;border-radius:14px;padding:6px;border:1px solid var(--line)}
[data-testid="stFileUploaderDropzone"] {background:#fafbf7;border:none;border-radius:12px}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden}
.stButton button,.stDownloadButton button {border-radius:9px;font-weight:550}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:25px;border-bottom:1px solid var(--line)}
[data-testid="stTabs"] [data-baseweb="tab"] {height:48px}
.kicker {font-size:11px;font-weight:650;letter-spacing:.18em;color:#226a50;margin-bottom:14px}
.intro {color:var(--muted);font-size:16px;max-width:780px;margin-bottom:22px;line-height:1.6}
.pill {display:inline-block;background:#e3ecdf;color:#326143;border-radius:20px;padding:5px 12px;font-size:12px;margin:8px 5px 20px 0}
.rule {height:1px;background:var(--line);margin:22px 0}
.empty {padding:42px 32px;background:#fff;border:1px solid var(--line);border-radius:16px}
.empty b {font-size:22px;font-weight:550;display:block;margin-bottom:10px}
.empty p {color:var(--muted);margin:0;line-height:1.7;max-width:750px}
.foot {font-size:12px;color:#879088;border-top:1px solid var(--line);padding-top:18px;margin-top:35px}
@media(max-width:800px){.block-container{padding:1.5rem 1rem}h1{font-size:2rem!important}[data-testid="stMetricValue"]{font-size:1.5rem}}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="kicker">DECLARATION LITE / 01</div>', unsafe_allow_html=True)
st.title("Стоимость начинается с данных.")
st.markdown('<div class="intro">Загрузите проектные декларации — получите объекты, площади и плановую стоимость строительства в одной таблице.</div>', unsafe_allow_html=True)
st.markdown('<span class="pill">PDF + ZIP</span><span class="pill">OCR для сканов</span><span class="pill">Данные остаются у вас</span>', unsafe_allow_html=True)

with st.container(border=True):
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    with left:
        files = st.file_uploader("Загрузить декларации", type=["pdf", "zip"], accept_multiple_files=True,
                                 max_upload_size=100, help="До 100 PDF в пакете. До 100 МБ на файл и 500 МБ в распакованном виде.")
    with right:
        force = st.checkbox("OCR всех страниц", help="Включайте для PDF с повреждённым текстовым слоем. Обычно сканы определяются автоматически.")
        submit = st.button("Распознать декларации →", type="primary", width="stretch", disabled=not files)
    st.caption("Сканы распознаются автоматически. После запуска можно закрыть вкладку — обработка продолжится, пока работает приложение.")
    if submit:
        try:
            batch = unpack_uploads([(f.name, f.getvalue()) for f in files])
            added, skipped = enqueue(batch, force_ocr=force)
            st.success(f"Добавлено в обработку: {added}. Уже загружено ранее: {skipped}.")
            ensure_worker()
        except (ValueError, OSError) as exc:
            st.error(str(exc))

docs = documents()
ensure_worker()
revision = tuple((d["sha"],d["status"], bool(d["result"])) for d in docs)

@st.fragment(run_every="3s")
def progress_panel():
    fresh = documents()
    active = [d for d in fresh if d["status"] in ("queued", "processing")]
    new_revision = tuple((d["sha"],d["status"], bool(d["result"])) for d in fresh)
    if new_revision != revision:
        st.rerun()
    for d in active[:5]:
        if d["status"] == "processing":
            st.progress(d["page"] / max(d["pages"],1), text=f"{d['filename']} · страница {d['page']} из {d['pages'] or '…'}")
        else:
            st.caption(f"В очереди · {d['filename']}")
    if len(active) > 5:
        st.caption(f"Ещё в обработке: {len(active)-5}")

progress_panel()

tab_data, tab_files = st.tabs(["Объекты и стоимость", "Загруженные документы"])
with tab_data:
    a,b,c = st.columns([2,1,1])
    query = a.text_input("Поиск", placeholder="ЖК, корпус, адрес или застройщик")
    all_rows = rows(docs, latest=False)
    regions = sorted({r["region"] for r in all_rows if r.get("region")})
    region = b.selectbox("Регион", ["Все регионы"] + regions)
    view = c.selectbox("Версии деклараций", ["Актуальные", "Вся история"])
    items = rows(docs, latest=view == "Актуальные")
    if query:
        items = [r for r in items if query.casefold() in " ".join(str(r.get(k) or "") for k in ["project","name","address","developer","declaration_number"]).casefold()]
    if region != "Все регионы":
        items = [r for r in items if r.get("region") == region]
    totals = summary(items)
    def number(v, decimals=0):
        return "—" if v is None else f"{v:,.{decimals}f}".replace(","," ")
    kpis = st.columns(4)
    kpis[0].metric("Объекты в сводке", totals["objects"])
    kpis[1].metric("Плановая стоимость", f"{number(totals['cost']/1e9,2)} млрд ₽" if totals["cost"] is not None else "—")
    kpis[2].metric("Стоимость / ОСП", f"{number(totals['gross'][0])} ₽/м²" if totals["gross"][0] is not None else "—")
    kpis[3].metric("Стоимость / продаваемая", f"{number(totals['saleable'][0])} ₽/м²" if totals["saleable"][0] is not None else "—")
    st.caption(f"Сводка по актуальным датированным версиям. Стоимость: {totals['cost_coverage']} объектов; стоимость / ОСП: {totals['gross'][1]}; стоимость / продаваемая: {totals['saleable'][1]}. Средние взвешены по площади.")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    if not items:
        if not all_rows:
            st.markdown('<div class="empty"><b>Ваш первый объект — в одной загрузке.</b><p>Добавьте декларацию в PDF или архив с несколькими документами. Здесь появятся корпуса, площади, стоимость строительства и стоимость квадратного метра. Каждый показатель можно открыть в исходном документе.</p></div>', unsafe_allow_html=True)
        else:
            st.info("По выбранным фильтрам нет объектов.")
    else:
        top_left, top_right = st.columns([3,2], vertical_alignment="center")
        top_left.subheader(f"Реестр объектов · {len(items)}")
        with top_right:
            x,y = st.columns(2)
            # Expensive export is built only on an explicit click.
            if x.button("Подготовить Excel", width="stretch"):
                st.session_state["xlsx"] = excel_bytes(items, docs)
                st.session_state["export_key"] = hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            y.download_button("Скачать CSV", csv_bytes(items), file_name="declaration_objects.csv", mime="text/csv", width="stretch")
        export_key = hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if st.session_state.get("export_key") == export_key and st.session_state.get("xlsx"):
            st.download_button("Скачать подготовленный Excel", st.session_state["xlsx"], file_name="declaration_objects.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        table = frame(items)
        visible = ["Проект / ЖК","Объект / корпус","Дата декларации","ОСП, м²","Продаваемая площадь, м²","Плановая стоимость, ₽","Стоимость / ОСП, ₽/м²","Стоимость / продаваемая, ₽/м²","Версия"]
        st.dataframe(table[visible], hide_index=True, width="stretch",
                     column_config={c:st.column_config.NumberColumn(format="localized",width="medium") for c in visible if "₽" in c or "м²" in c})
        st.caption("Стоимость взята из поля 18.1.1. Это плановая стоимость по декларации; фактические затраты на строительство документ не подтверждает.")
        st.subheader("Карточка объекта")
        index = st.selectbox("Открыть объект", list(range(len(items))), format_func=lambda i: f"{items[i]['project'] or 'Проект'} · {items[i]['name']} · {items[i]['declaration_date'] or 'без даты'}", label_visibility="collapsed")
        row = items[index]
        details, evidence = st.columns([1,1.4], gap="large")
        with details:
            st.markdown(f"**{row['name']}**")
            st.write(row.get("address") or "Адрес не распознан")
            st.caption(row.get("developer") or "Застройщик не распознан")
            st.dataframe(pd.DataFrame({"Показатель":["Декларация", "Дата", "Этажей", "ОСП", "Продаваемая площадь", "Доля продаваемой площади", "Плановая стоимость", "Метод"],
                         "Значение":[str(row['declaration_number'] or '—'),str(row['declaration_date'] or '—'),number(row['floors_max']),number(row['gross_area'])+' м²',number(row['saleable_area'])+' м²',number(row['saleable_share']*100,1)+' %' if row['saleable_share'] is not None else '—',number(row['planned_cost'])+' ₽',"PDF + OCR" if row['has_ocr'] else "Текст PDF"]}), hide_index=True,width="stretch")
            for note in row["issues"] + row["document_warnings"]:
                st.caption("• " + note)
            if not row["eligible"]:
                st.info(f"В сводку не включён: {row['version'].lower()}.")
            original = data_dir() / "pdf" / f"{row['sha']}.pdf"
            if original.exists():
                st.download_button("Исходная декларация PDF", original.read_bytes(), file_name=row["filename"], mime="application/pdf")
        with evidence:
            fields = list(row["evidence"].values())
            if fields:
                chosen = st.selectbox("Источник показателя", fields, format_func=lambda f: f"{f['code']} · {f['label'][:75]} · стр. {f['page']}")
                st.code(chosen["value"], language=None, wrap_lines=True)
                st.caption(f"{chosen['method']} · страница {chosen['page']}. Рамкой выделен фрагмент, из которого взято значение.")
                if original.exists():
                    with fitz.open(original) as pdf:
                        page = pdf[chosen["page"]-1]
                        box = fitz.Rect(chosen["bbox"])
                        if not box.is_empty:
                            page.draw_rect(box, color=(.1,.48,.32), width=1.4, overlay=True)
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.3,1.3), alpha=False)
                        st.image(pix.tobytes("png"), width="stretch")
            else:
                st.info("Для этого объекта значения не распознаны.")
    with st.expander("Как рассчитаны показатели"):
        st.markdown("""**ОСП** — общая площадь здания из поля **9.2.21**. **Продаваемая площадь** — общая площадь жилых и нежилых помещений из поля **9.3.3**. Это разные основания для расчёта.

**Стоимость / ОСП** = плановая стоимость из 18.1.1 ÷ ОСП. **Стоимость / продаваемая** = та же стоимость ÷ продаваемая площадь. **Доля продаваемой площади** = продаваемая площадь ÷ ОСП.

Сводная удельная стоимость = сумма стоимостей ÷ сумма соответствующих площадей **только объектов с обоими значениями**. Отсутствующие числа не заменяются нулём. Старая редакция не увеличивает итоги. Конфликтующие версии одной даты и документы без даты видны в таблице, но не входят в сводку.

Значения отражают содержание декларации на её дату. Корректировки на класс, этажность, инфляцию и нераскрытые расходы не применяются.""")

with tab_files:
    st.subheader("Документы и распознавание")
    if not docs:
        st.info("Загруженные файлы появятся здесь.")
    status_labels = {"queued":"В очереди","processing":"Распознаётся","done":"Обработан","failed":"Не обработан"}
    for d in docs:
        result = json.loads(d["result"]) if d["result"] else None
        with st.expander(f"{d['filename']} · {status_labels[d['status']]}"):
            if result:
                st.write(f"Объектов: {len(result['objects'])} · Страниц: {result['pages']} · OCR: {result['ocr_pages']} · Время: {result['elapsed']:.0f} с")
                st.caption(f"Дата декларации: {result['header'].get('declaration_date') or 'не распознана'}")
                for note in result["warnings"]:
                    st.warning(note)
                st.download_button("Скачать данные JSON", d["result"], file_name=Path(d["filename"]).stem + ".json", mime="application/json", key="json"+d["sha"])
            if d["error"]:
                st.error(d["error"])
            if d["status"] in ("done", "failed"):
                ocr_again = st.checkbox("Применить OCR ко всем страницам", key="force"+d["sha"])
                if st.button("Распознать заново", key="retry"+d["sha"]):
                    retry(d["sha"], force_ocr=ocr_again)
                    ensure_worker()
                    st.rerun()
            st.caption(f"Загружен: {d['uploaded_at'][:19].replace('T',' ')} UTC · SHA-256: {d['sha'][:16]}…")

st.markdown(f'<div class="foot">Declaration Lite {__version__} · PDF → данные об объекте → стоимость · Локальная обработка без API-ключей</div>', unsafe_allow_html=True)
