"""Independent fixtures for the read-only structural census used by spec v2."""
import json
import struct

import numpy as np
import pytest

from scripts.navarro_structural_census import census


def fixture_file(path, arrays):
    header = {}
    payload = bytearray()
    for name, array in arrays.items():
        raw = array.tobytes()
        header[name] = {
            'dtype': 'F32' if array.dtype == np.dtype('<f4') else 'I32',
            'shape': list(array.shape),
            'data_offsets': [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)


def test_all_blocks_and_read_only(tmp_path):
    path = tmp_path / 'fixture.safetensors'
    fixture_file(path, {'weight': np.arange(1, 521, dtype='<f4').reshape(130, 4)})
    original = path.read_bytes()
    rows, skipped = census(path, 128)
    assert skipped == []
    assert [(r['row_start'], r['rows']) for r in rows] == [(0, 128), (128, 2)]
    assert sum(r['cells'] for r in rows) == 520
    assert all(r['rejected_by_dictionary_bound'] for r in rows)
    assert path.read_bytes() == original


def test_zeros_remain_undecided_and_scalars_are_explicitly_skipped(tmp_path):
    path = tmp_path / 'fixture.safetensors'
    fixture_file(path, {
        'zeros': np.zeros((128, 4), dtype='<f4'),
        'counter': np.array(1000, dtype='<i4'),
        'vector': np.ones(8, dtype='<f4'),
    })
    rows, skipped = census(path, 128)
    assert len(rows) == 1
    assert rows[0]['nonzero_bit_patterns'] == 0
    assert rows[0]['positive_zero_cells'] == 512
    assert rows[0]['dictionary_header_lower_bound_bytes'] == 64
    assert not rows[0]['rejected_by_dictionary_bound']
    assert {r['tensor'] for r in skipped} == {'counter', 'vector'}


def test_ieee_identity_is_counted_by_bits(tmp_path):
    path = tmp_path / 'fixture.safetensors'
    bits = np.array([0, 0x80000000, 0x7fc00001, 0x7fc00002,
                     0x7f800000, 0x3f800000, 0x3f800000, 1], dtype='<u4')
    fixture_file(path, {'special': bits.view('<f4').reshape(2, 4)})
    rows, _ = census(path, 128)
    assert rows[0]['special_cells'] == 4
    assert rows[0]['nonzero_bit_patterns'] == 6
    assert rows[0]['positive_zero_cells'] == 1


def test_dictionary_bound_not_a_claim_about_grammar(tmp_path):
    path = tmp_path / 'fixture.safetensors'
    fixture_file(path, {'repeated': np.tile(np.array([1, 2, 3, 4], dtype='<f4'), (128, 1))})
    rows, _ = census(path, 128)
    assert rows[0]['dictionary_header_lower_bound_bytes'] == 80
    assert not rows[0]['rejected_by_dictionary_bound']
    assert 'grammar_bytes' not in rows[0]


def test_truncated_tensor_is_rejected(tmp_path):
    path = tmp_path / 'fixture.safetensors'
    fixture_file(path, {'weight': np.ones((4, 4), dtype='<f4')})
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match='Invalid tensor extent'):
        census(path, 128)
