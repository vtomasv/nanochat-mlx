"""Protocol estimators; repetition unit is a process, never a token."""
import numpy as np
from scipy.stats import t

def errors(actual, reference, atol=1e-5, rtol=1e-4):
    a, b = np.asarray(actual), np.asarray(reference)
    diff = a.astype(np.float64)-b.astype(np.float64)
    denom = float(np.linalg.norm(b.astype(np.float64)))
    return {'max_abs': float(np.max(np.abs(diff), initial=0)),
            'relative_l2': float(np.linalg.norm(diff)/max(denom,1e-30)),
            'pass': bool(np.allclose(a,b,atol=atol,rtol=rtol, equal_nan=False))}

def paired_ratios(experimental, control):
    a,b = np.asarray(experimental,float),np.asarray(control,float)
    if a.shape != b.shape or a.ndim != 1 or len(a)<2 or np.any(a<=0) or np.any(b<=0):
        raise ValueError('Need matched positive process measurements')
    logs = np.log(a/b)
    ix = np.random.default_rng(20260913).integers(0,len(a),(10000,len(a)))
    boot = np.exp(logs[ix].mean(axis=1))
    return {'points': (a/b).tolist(), 'geomean': float(np.exp(logs.mean())), 'ci95': np.quantile(boot,[.025,.975]).tolist()}

def noninferiority(experimental, control):
    d = np.asarray(experimental,float)-np.asarray(control,float)
    if d.shape != (3,): raise ValueError('Three paired seed trajectories required')
    upper = float(d.mean()+t.ppf(.95,2)*d.std(ddof=1)/np.sqrt(3))
    return {'differences': d.tolist(), 'upper95': upper, 'pass': upper<.01, 'degenerate_zero': bool(np.all(d==0))}
