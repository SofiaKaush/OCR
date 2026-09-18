import hashlib
import json
from io import BytesIO
from zipfile import ZipFile
import pytest
import fitz
from lite.parser import parse_pdf, amount
from lite.storage import enqueue, documents, rows, summary, unpack_uploads, update
from lite.export import excel_bytes, csv_bytes
from .fixtures import declaration

@pytest.mark.parametrize('value,label,want',[
 ('1 847 065 565,00 руб.','',1847065565),('123,5 млн руб.','',123500000),
 ('50 000,00','Стоимость, тыс. руб.',50000000),('50 000,00','',None),
 ('от 100 до 200 млн руб.','',None),('0 руб.','',0),('-1 руб.','',None),
])
def test_money(value,label,want):
    assert amount(value,label)==want

@pytest.fixture(scope='module')
def result():
    return parse_pdf(declaration(),'sample.pdf')

def test_native_two_objects(result):
    assert result['header']['date_iso']=='2026-09-01'
    assert len(result['objects'])==2
    a,b=result['objects']
    assert (a['gross_area'],a['saleable_area'],a['planned_cost'])==(10000,7000,1e9)
    assert (b['gross_area'],b['saleable_area'],b['planned_cost'])==(20000,14000,3e9)
    assert b['cost_per_gross']==150000
    assert not a['issues'] and not b['issues']
    assert a['evidence']['18.1.1']['page']==b['evidence']['18.1.1']['page']
    assert a['evidence']['18.1.1']['object_no']==1
    assert b['evidence']['18.1.1']['object_no']==2

@pytest.mark.ocr
def test_real_russian_ocr():
    result=parse_pdf(declaration(scan=True),'scan.pdf')
    assert result['ocr_pages']==result['pages']
    assert result['header']['date_iso']=='2026-09-01'
    assert len(result['objects'])==2
    assert result['objects'][0]['planned_cost']==1e9
    assert result['objects'][1]['planned_cost']==3e9
    assert result['objects'][0]['gross_area']==10000
    assert result['objects'][1]['saleable_area']==14000
    assert result['objects'][1]['has_ocr']


def test_blank_and_protected():
    doc=fitz.open();doc.new_page()
    blank=doc.tobytes()
    r=parse_pdf(blank,'blank.pdf')
    assert r['objects']==[]
    locked=doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256,owner_pw='owner',user_pw='password')
    with pytest.raises(ValueError,match='паролем'):
        parse_pdf(locked,'locked.pdf')
    doc.close()


def test_zip_and_dedup(tmp_path):
    data=declaration(); buf=BytesIO()
    with ZipFile(buf,'w') as z:z.writestr('folder/sample.pdf',data)
    files=unpack_uploads([('files.zip',buf.getvalue())])
    assert files[0][0]=='sample.pdf'
    assert enqueue(files,root=tmp_path)==(1,0)
    assert enqueue(files,root=tmp_path)==(0,1)
    assert len(documents(tmp_path))==1
    buf=BytesIO()
    with ZipFile(buf,'w') as z:z.writestr('../evil.pdf',data)
    with pytest.raises(ValueError,match='пути'):
        unpack_uploads([('evil.zip',buf.getvalue())])


def doc(result,sha='a',date=None):
    r=json.loads(json.dumps(result))
    if date is not None:r['header']['date_iso']=date
    return dict(sha=sha,filename='sample.pdf',status='done',result=json.dumps(r),uploaded_at='2026-09-18')

def test_latest_date_not_upload_and_weighted(result):
    old,new=doc(result,'old','2026-01-01'),doc(result,'new','2026-09-01')
    items=rows([old,new])
    assert len(items)==2 and all(r['sha']=='new' for r in items)
    assert len(rows([old,new],latest=False))==4
    s=summary(items)
    assert s['cost']==4e9
    assert s['gross'][0]==pytest.approx(4e9/30000)
    items[0]['cost_per_gross']=None
    assert summary(items)['gross']==(150000,1)

def test_same_date_conflict_and_undated(result):
    a,b=doc(result,'a'),doc(result,'b')
    r=json.loads(b['result']);r['objects'][0]['planned_cost']+=1;b['result']=json.dumps(r)
    assert summary(rows([a,b]))['objects']==0
    assert all(x['version']=='Конфликт одной даты' for x in rows([a,b]))
    r['header']['date_iso']=None;b['result']=json.dumps(r)
    assert summary(rows([b]))['objects']==0
    assert len(rows([a,b]))==4

def test_identical_revision_not_doubled(result):
    assert summary(rows([doc(result,'a'),doc(result,'b')]))['objects']==2

def test_excel_and_csv(result):
    docs=[doc(result)];items=rows(docs)
    items[0]['project']='=HYPERLINK("https://bad.example")'
    xlsx=excel_bytes(items,docs)
    with ZipFile(BytesIO(xlsx)) as z:
        assert b'<f>' not in z.read('xl/worksheets/sheet1.xml')
        assert 'xl/worksheets/sheet3.xml' in z.namelist()
    assert b"'=HYPERLINK" in csv_bytes(items)
