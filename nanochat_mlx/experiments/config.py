"""Closed, hashable protocol configuration."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDITED_SHA = 'b54b9fc139f455a9a5e60dc9a688ca9dbdb22944'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def load_config(path):
    c = json.loads(Path(path).read_text())
    p = c['profile']
    expected = {'smoke': (4,128,2,256,'L',[17]), 'primary': (8,1024,4,8192,'SSSL',[17,29,43]), 'confirm': (12,1024,4,8192,'SSSL',[17,29,43])}[p]
    actual = tuple(c[k] for k in ('depth','sequence_len','device_batch_size','total_batch_size','window_pattern','seeds'))
    if actual != expected or c['dtype'] != 'float32' or c['num_iterations'] != 1000:
        raise ValueError('Closed protocol profile changed; register a new protocol before running')
    if c['aspect_ratio']!=64 or c['head_dim']!=128 or c['pilot_steps']!=100 or c['pilot_seed']!=101:
        raise ValueError('Closed model or pilot profile changed')
    if c['rank_sample_bits'] != 256 or c['rows_per_block'] != 128:
        raise ValueError('Pilot calibration must be registered before changing representation')
    c['accumulation'] = c['total_batch_size'] // (c['device_batch_size'] * c['sequence_len'])
    c['config_sha256'] = sha256(path)
    c['data_dir'] = str(Path(c['data_dir']).expanduser().resolve())
    c['artifact_dir'] = str((ROOT / c['artifact_dir']).resolve())
    if not Path(c['artifact_dir']).is_relative_to(ROOT/'experiments/navarro/artifacts'):
        raise ValueError('Experimental artifacts must stay in the isolated artifacts root')
    return c
