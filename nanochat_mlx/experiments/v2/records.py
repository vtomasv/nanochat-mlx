"""Canonical manifests, frozen source snapshots and tamper-evident append ledger."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[3]


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path,value,exclusive=True):
    serialized=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:f.write(serialized)


def source_files():
    paths=[]
    for directory in ('nanochat_mlx','scripts','tests','research/navarro','configs/navarro/v2'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.json','.md')]
    paths += [ROOT/p for p in ('pyproject.toml','uv.lock','ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md')]
    return sorted(paths)


def snapshot(destination):
    files={str(p.relative_to(ROOT)):digest(p) for p in source_files()}
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:archive.write(ROOT/name,name)
    return {'files':files,'tree_sha256':hashlib.sha256(canonical(files)).hexdigest(),
        'archive_sha256':digest(destination),
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}


def verify_sources(files):
    bad=[name for name,h in files.items() if not (ROOT/name).is_file() or digest(ROOT/name)!=h]
    if bad:raise ValueError('Frozen source mismatch: '+', '.join(bad))


def ledger_read(path):
    path=Path(path);records=[];previous='0'*64
    if not path.exists():return records
    for line in path.read_text().splitlines():
        record=json.loads(line);signature=record.pop('record_sha256')
        if record['previous_sha256']!=previous or hashlib.sha256(canonical(record)).hexdigest()!=signature:
            raise ValueError('Ledger chain/integrity failure')
        record['record_sha256']=signature;records.append(record);previous=signature
    return records


def ledger_append(path,event):
    previous=ledger_read(path)
    value={'sequence':len(previous),'at':now(),'previous_sha256':previous[-1]['record_sha256'] if previous else '0'*64,**event}
    value['record_sha256']=hashlib.sha256(canonical(value)).hexdigest()
    with Path(path).open('a') as f:f.write(canonical(value).decode()+'\n');f.flush()
    return value
