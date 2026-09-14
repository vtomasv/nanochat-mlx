"""Verify persisted artifacts without mutating scientific measurements."""
from pathlib import Path
import json
from .config import sha256,dump

def verify_results(root):
    root=Path(root);result=[]
    for directory in sorted(root.glob('*/*')):
        if not directory.is_dir() or not (directory/'manifest.json').exists():continue
        errors=[]
        index=directory/'artifacts.sha256'
        if not index.exists():errors.append('missing artifacts.sha256')
        else:
            for line in index.read_text().splitlines():
                digest,name=line.split('  ',1);p=directory/name
                if not p.exists() or sha256(p)!=digest:errors.append(f'checksum mismatch: {name}')
        summary=json.loads((directory/'summary.json').read_text())
        if summary['decision']=='SUCCESS' and (not summary['correctness_pass'] or summary['confidence_intervals'] is None):errors.append('unsupported SUCCESS')
        result.append({'run':str(directory),'pass':not errors,'errors':errors,'status':summary['status']})
    dump(root/'audit.json',result)
    return result
