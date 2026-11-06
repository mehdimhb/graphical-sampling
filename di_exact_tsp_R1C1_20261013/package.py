from pathlib import Path
import json,hashlib,zipfile
P=Path(__file__).resolve().parent
assert json.loads((P/'verification.json').read_text())['status']=='passed'
files=[p for p in sorted(P.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
manifest={str(p.relative_to(P)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
(P/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
archive=Path('/tmp/DI_EXACT_TSP_03_All_Outputs.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[P/'artifact_manifest.json']:z.write(p,'DI_EXACT_TSP/'+str(p.relative_to(P)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
info=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files)+1)
Path('/tmp/di_exact_tsp_archive.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
