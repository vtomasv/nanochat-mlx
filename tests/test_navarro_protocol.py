import json
import numpy as np
import pytest
from nanochat_mlx.experiments.config import load_config
from nanochat_mlx.experiments.metrics import paired_ratios,noninferiority
from nanochat_mlx.experiments.replay import verify_files

def test_closed_profiles_and_hashes(tmp_path):
    c=load_config('configs/navarro/primary.json');assert c['accumulation']==2
    changed=json.loads(open('configs/navarro/primary.json').read());changed['num_iterations']=50
    p=tmp_path/'bad.json';p.write_text(json.dumps(changed))
    with pytest.raises(ValueError):load_config(p)
    with pytest.raises(ValueError):verify_files({str(p):'incorrect'})

def test_process_estimators():
    result=paired_ratios([2,4,6,8,10],[1,2,3,4,5]);assert result['geomean']==2
    np.testing.assert_allclose(result['ci95'],[2,2])
    assert noninferiority([3,4,5],[3,4,5])['degenerate_zero']
    assert not noninferiority([3.1,4.1,5.1],[3,4,5])['pass']
    with pytest.raises(ValueError):noninferiority([1],[1])
