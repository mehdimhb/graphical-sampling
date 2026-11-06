"""Package self-contained results, code and provenance; test archive integrity."""
from pathlib import Path
import zipfile,hashlib,json
HOME=Path(__file__).resolve().parent
ARCHIVE=HOME.parent/'HARD_CLUSTER_IP_BALANCE_20261014.zip'

def files():return sorted(p for p in HOME.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name not in ('MANIFEST_SHA256.json','delivery.json'))

def package():
    manifest={str(p.relative_to(HOME)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
    (HOME/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files()+[HOME/'MANIFEST_SHA256.json']:z.write(p,HOME.name+'/'+str(p.relative_to(HOME)))
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(HOME.name+'/'+name)).hexdigest()==digest
    result=dict(archive=ARCHIVE.name,bytes=ARCHIVE.stat().st_size,MiB=ARCHIVE.stat().st_size/2**20,sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),manifest_files=len(manifest),archive_integrity_passed=True)
    (HOME/'delivery.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':package()
