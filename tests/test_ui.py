import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
from lite.parser import parse_pdf
from lite.storage import enqueue, documents, update
from .fixtures import declaration


def test_ui_empty_populated_export_and_filter(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR',str(tmp_path))
    app=Path(__file__).resolve().parents[1]/'app.py'
    a=AppTest.from_file(str(app)).run(timeout=30)
    assert not a.exception
    assert a.metric[0].value=='0'
    pdf=declaration()
    enqueue([('demo.pdf',pdf)])
    d=documents()[0]
    update(d['sha'],status='done',result=json.dumps(parse_pdf(pdf,'demo.pdf')))
    a.run(timeout=30)
    assert not a.exception
    assert a.metric[0].value=='2'
    assert len(a.dataframe)==2
    next(b for b in a.button if b.label=='Подготовить Excel').click().run(timeout=30)
    assert not a.exception
    assert a.session_state['xlsx'].startswith(b'PK')
    a.text_input[0].set_value('НЕСУЩЕСТВУЮЩИЙ ОБЪЕКТ').run(timeout=30)
    assert a.metric[0].value=='0'
    assert not a.exception
