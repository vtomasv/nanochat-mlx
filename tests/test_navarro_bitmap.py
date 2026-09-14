import numpy as np
import pytest
from nanochat_mlx.experiments.bitmap import PackedPositive

@pytest.mark.parametrize('n',[0,1,31,32,33,255,256,257,1025])
@pytest.mark.parametrize('kind',['zero','one','alternating'])
def test_rank_roundtrip(n,kind):
    p=np.zeros(n,np.float32) if kind=='zero' else np.ones(n,np.float32)
    if kind=='alternating': p[::2]=0
    h=PackedPositive.pack(p,force=True)
    assert h.unpack().tobytes()==p.tobytes()
    for i in range(n+1): assert h.rank(i)==np.count_nonzero(p[:i])
    for i in range(n): assert h.access(i)==p[i]

def test_domain_physical():
    p=np.array([0,np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).max,2],np.float32)
    assert PackedPositive.pack(p[::2],force=True).unpack().tobytes()==p[::2].tobytes()
    for bad in ([-1.],[float('inf')],[float('nan')]):
        with pytest.raises(ValueError): PackedPositive.pack(np.array(bad,np.float32))
    h=PackedPositive.pack(np.tile(np.array([0,2],np.float32),4096))
    assert h.values.flags.owndata and h.values.nbytes==4096*4
    assert h.resident_bytes()<.75*h.raw_bytes
