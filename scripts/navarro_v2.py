"""Controlled autoresearch for the Navarro v2 campaign on real MLX/Metal."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=['prepare','preflight','init','reference','gate','campaign','report','status','submit','seal','confirm','worker','screen'])
    parser.add_argument('--config',default='configs/navarro/v2/campaign.json')
    parser.add_argument('--campaign',default='experiments/navarro/v2/20260915')
    parser.add_argument('--data',default='experiments/navarro/artifacts/v2-data-20260915')
    parser.add_argument('--trial')
    parser.add_argument('--scenario')
    parser.add_argument('--hypothesis')
    parser.add_argument('--candidate',default='research/navarro/candidate.py')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        from nanochat_mlx.experiments.v2.data import prepare
        result=prepare(json.loads(Path(args.config).read_text()),args.data)
        print(json.dumps({k:v for k,v in result.items() if k not in ('files','source_files')},indent=2));return
    if args.command=='preflight':
        from nanochat_mlx.experiments.measure import environment
        print(json.dumps(environment(),indent=2));return
    if args.command=='worker':
        from nanochat_mlx.experiments.v2.worker import run
        run(Path(args.campaign),args.trial);return
    from nanochat_mlx.experiments.v2.campaign import dispatch
    dispatch(args)


if __name__=='__main__':main()
