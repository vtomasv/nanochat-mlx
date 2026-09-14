"""Synchronized timing and independent RSS/Metal accounting."""
import importlib.metadata
import os
import platform
import resource
import re
import subprocess
import threading
import time

def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return {'error': str(e)}

def environment():
    d = {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
         'repo_sha': command('git','rev-parse','HEAD'), 'power': command('pmset','-g','batt'),
         'thermal': command('pmset','-g','therm'), 'swap': command('sysctl','vm.swapusage'),
         'compiler': command('clang++','--version'), 'packages': {}}
    for p in ('mlx','numpy','scipy','pytest','pyarrow','tiktoken'):
        try: d['packages'][p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: d['packages'][p] = None
    try:
        import mlx.core as mx
        a = mx.ones((8,8)); mx.eval(a@a); mx.synchronize()
        d.update(metal_available=mx.metal.is_available(), device=mx.device_info())
        d['memory_limit_bytes'] = min(80*1024**3, d['device']['max_recommended_working_set_size'])
    except Exception as e:
        d.update(metal_available=False, device=None, metal_error=repr(e), memory_limit_bytes=None)
    return d

def setup_device():
    import mlx.core as mx
    limit = min(80*1024**3, mx.device_info()['max_recommended_working_set_size'])
    mx.set_memory_limit(limit)
    mx.set_cache_limit(1024**3)
    return limit

def memory():
    import mlx.core as mx
    result = {'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)}
    for field, name in [('mlx_active_peak_bytes','get_peak_memory'),('mlx_active_bytes','get_active_memory'),('mlx_cache_bytes','get_cache_memory')]:
        try: result[field] = getattr(mx,name)()
        except Exception as e: result[field] = None; result[field+'_error'] = repr(e)
    return result

class RSSSampler:
    """Sample parent and descendants concurrently; never add separate lifetime peaks."""
    def __init__(self):
        self.peak = 0
        self.error = None
        self.swap_violation = False
        self.swap_samples = []
        self.stop = threading.Event()
    def __enter__(self):
        def sample():
            last_swap=0.;initial_swap=None;sustained=0
            while not self.stop.is_set():
                try:
                    rows = subprocess.check_output(['ps','-axo','pid=,ppid=,rss='], text=True)
                    rows = [tuple(map(int,r.split())) for r in rows.splitlines()]
                    ids = {os.getpid()}
                    for _ in range(10):
                        expanded = ids | {p for p,parent,_ in rows if parent in ids}
                        if expanded == ids: break
                        ids = expanded
                    self.peak = max(self.peak, sum(rss*1024 for p,_,rss in rows if p in ids))
                    if platform.system()=='Darwin' and time.monotonic()-last_swap>=1:
                        swap=subprocess.check_output(['sysctl','vm.swapusage'],text=True)
                        found=re.search(r'used = ([0-9.]+)([MG])',swap)
                        if found:
                            used=float(found[1])*(1024**2 if found[2]=='M' else 1024**3)
                            if initial_swap is None:initial_swap=used
                            sustained=sustained+1 if used-initial_swap>1024**3 else 0
                            self.swap_violation=sustained>=3
                            self.swap_samples.append({'monotonic_seconds':time.monotonic(),'used_bytes':int(used)})
                        last_swap=time.monotonic()
                except Exception as e: self.error = repr(e)
                self.stop.wait(.05)
        self.thread = threading.Thread(target=sample, daemon=True); self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join()
