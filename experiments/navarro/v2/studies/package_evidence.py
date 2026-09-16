"""Create and verify a local evidence capsule; exclude datasets/checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from nanochat_mlx.experiments.v2.records import ROOT, digest, now


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.with_suffix('.sha256').exists():
        raise FileExistsError('Capsule destination must be new')
    paths = set()
    for name in ('nanochat_mlx', 'scripts', 'tests', 'configs/navarro/v2',
                 'research/navarro', 'experiments/navarro/v2'):
        paths.update(p for p in (ROOT / name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts
                     and p.suffix in {'.py', '.cpp', '.h', '.md', '.json', '.jsonl', '.csv',
                                      '.tsv', '.xml', '.log', '.txt', '.zip', '.svg', '.pdf', '.png'})
    paths.update(ROOT / p for p in ('pyproject.toml', 'uv.lock', 'LICENSE', 'README.md',
                 'ESPECIFICACIONES_NAVARRO_NANOCHAT_MLX.md', 'docs/navarro/autoresearch-v2.md'))
    # A capsule created within the evidence tree must not include itself.
    if output in paths:
        raise ValueError('Output is already an input')
    files = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    metadata = ROOT / 'experiments/navarro/artifacts/v2-data-20260915/manifest.json'
    if metadata.exists():
        files[str(metadata.relative_to(ROOT))] = digest(metadata)
        paths.add(metadata)
    record = {'created_at': now(), 'source_sha256': digest(Path(__file__)), 'files': files,
              'scope': 'Code and evidence, no corpus/tokenizer/checkpoint payload. Not a self-contained training reproduction.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            archive.write(p, str(p.relative_to(ROOT)))
        archive.writestr('EVIDENCE-MANIFEST.json', json.dumps(record, indent=2, sort_keys=True) + '\n')
    with zipfile.ZipFile(output) as archive:
        if len(archive.namelist()) != len(files) + 1:
            raise ValueError('Unexpected archive member count')
        for name, expected in files.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Archive roundtrip mismatch: ' + name)
    checksum = digest(output)
    with output.with_suffix('.sha256').open('x') as f:
        f.write(f'{checksum}  {output.name}\n')
    print(json.dumps({'archive': str(output), 'files_verified': len(files),
                      'bytes': output.stat().st_size, 'sha256': checksum}))


if __name__ == '__main__':
    main()
