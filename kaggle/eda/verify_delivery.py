"""Verifica los artefactos descargados, sin volver a ejecutar el análisis."""
import csv
import hashlib
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs'
run = json.loads((OUT/'run_summary.json').read_text(encoding='utf-8'))
assert run['status']=='complete', run
assert run['stage']=='done', run
nb = nbformat.read(OUT/'m5_eda_executed.ipynb',as_version=4)
nbformat.validate(nb)
source = nbformat.read(ROOT/'input'/'m5_eda.ipynb',as_version=4)
assert [c.source for c in source.cells]==[c.source for c in nb.cells], 'La fuente ejecutada no coincide con la local'
code = [c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
errors = [o for c in code for o in c.outputs if o.output_type=='error']
assert not errors, errors
inline_images = sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
png = list((OUT/'figures').glob('*.png'))
assert len(png)==9 and inline_images==9, (len(png),inline_images)
csv_files = list((OUT/'tables').glob('*.csv'))
assert len(csv_files)>=25, len(csv_files)
assert all(p.stat().st_size>0 for p in csv_files), 'CSV vacío o descarga incompleta'
with (OUT/'tables'/'quality_checks.csv').open(encoding='utf-8',newline='') as f:
    checks = list(csv.DictReader(f))
assert checks and all(r['passed']=='True' for r in checks)
verified_raw = []
with (OUT/'tables'/'official_file_manifest.csv').open(encoding='utf-8',newline='') as f:
    for row in csv.DictReader(f):
        p = ROOT/'data'/row['file']
        assert p.stat().st_size==int(row['bytes'])
        with p.open('rb') as stream:
            actual = hashlib.file_digest(stream,'sha256').hexdigest()
        assert actual==row['sha256'], row['file']
        verified_raw.append(row['file'])
result = {'status':'passed','executed_code_cells':len(code),'notebook_error_outputs':len(errors),
    'inline_png_figures':inline_images,'downloaded_png_figures':len(png),'analytical_csv_files':len(csv_files),
    'quality_checks_passed':len(checks),'official_csv_hashes_verified':verified_raw}
(ROOT/'delivery_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
