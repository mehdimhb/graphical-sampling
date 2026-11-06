"""Create a new Desktop copy and ZIP; reject existing destinations."""
from pathlib import Path
import json,hashlib,shutil,zipfile
P=Path(__file__).resolve().parent;D=Path('/home/bardia/Desktop')/P.name;Z=D.with_suffix('.zip')
assert not D.exists() and not Z.exists(),'Destination exists; preserve it and choose a new delivery name.'
assert json.loads((P/'verification.json').read_text())['status']=='passed'
assert json.loads((P/'replication_verification.json').read_text())['status']=='passed'
files=[p for p in sorted(P.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
for p in files:
    target=D/p.relative_to(P);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    assert p.read_bytes()==target.read_bytes()
# Record delivery only after its entire reviewable scientific content has been copied.
plan=P/'plan.md';text=plan.read_text().replace('6. [ ]','6. [x]')
text+='\n- Task6 complete: new Desktop directory copied and byte-verified; ZIP and artifact manifest generated in the delivery action. Existing destinations were rejected rather than overwritten.\n'
plan.write_text(text);shutil.copy2(plan,D/'plan.md')
manifest={str(p.relative_to(P)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
(P/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(P/'artifact_manifest.json',D/'artifact_manifest.json')
for name,meta in manifest.items():
    copied=D/name;assert copied.stat().st_size==meta['bytes'];assert hashlib.sha256(copied.read_bytes()).hexdigest()==meta['sha256']
with zipfile.ZipFile(Z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(D.rglob('*')):
        if p.is_file():archive.write(p,D.name+'/'+str(p.relative_to(D)))
with zipfile.ZipFile(Z) as archive:assert archive.testzip() is None
sha=hashlib.sha256(Z.read_bytes()).hexdigest();Z.with_suffix('.zip.sha256').write_text(sha+'  '+Z.name+'\n')
print(json.dumps(dict(directory=str(D),archive=str(Z),files_verified=len(manifest),archive_bytes=Z.stat().st_size,sha256=sha),indent=2))
