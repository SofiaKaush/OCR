"""Recreate the two explicitly synthetic demo PDFs (development dependencies)."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from tests.fixtures import declaration
(root/'examples').mkdir(exist_ok=True)
(root/'examples'/'01_demo_text.pdf').write_bytes(declaration())
(root/'examples'/'02_demo_scan.pdf').write_bytes(declaration(scan=True))
print('Created examples/01_demo_text.pdf and examples/02_demo_scan.pdf')
