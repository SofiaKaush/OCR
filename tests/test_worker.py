from lite.storage import enqueue, documents, update
from lite.worker import work
from .fixtures import declaration


def test_bad_pdf_does_not_block_queue_and_restart(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    enqueue([('broken.pdf', b'%PDF-corrupt'),('valid.pdf',declaration(second=False))])
    work()
    found={d['filename']:d for d in documents()}
    assert found['broken.pdf']['status']=='failed'
    assert found['valid.pdf']['status']=='done'
    sha=found['valid.pdf']['sha']
    update(sha, status='processing', result=None)
    work()
    found={d['filename']:d for d in documents()}
    assert found['valid.pdf']['status']=='done'
