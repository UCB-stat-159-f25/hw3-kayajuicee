import os
import json
import numpy as np
from ligotools import readligo as rl

DATA = "data"
FNJSON = os.path.join(DATA, "BBH_events_v3.json")

def _event():
    with open(FNJSON, "r") as f:
        evs = json.load(f)
    # the homework uses GW150914 by default
    return evs["GW150914"]

def test_loaddata_shapes_and_monotonic_time():
    ev = _event()
    fn_H1 = os.path.join(DATA, ev["fn_H1"])
    fn_L1 = os.path.join(DATA, ev["fn_L1"])

    strain_H1, time_H1, chan_H1 = rl.loaddata(fn_H1, "H1")
    strain_L1, time_L1, chan_L1 = rl.loaddata(fn_L1, "L1")

    # non-empty and same length
    assert len(strain_H1) > 0 and len(time_H1) == len(strain_H1)
    assert len(strain_L1) > 0 and len(time_L1) == len(strain_L1)

    # time strictly increasing
    assert np.all(np.diff(time_H1) > 0)
    assert np.all(np.diff(time_L1) > 0)

def test_dq_channel_to_seglist_nonempty_for_DATA_flag():
    ev = _event()
    fn_H1 = os.path.join(DATA, ev["fn_H1"])
    strain_H1, time_H1, chan_H1 = rl.loaddata(fn_H1, "H1")

    # should have a DATA bit field
    assert "DATA" in chan_H1
    segs = rl.dq_channel_to_seglist(chan_H1["DATA"])
    # at least one segment of usable data
    assert isinstance(segs, list)
    seg_lengths = [(s.stop - s.start) if isinstance(s, slice) else len(s) for s in segs]
    assert any(L > 0 for L in seg_lengths)
