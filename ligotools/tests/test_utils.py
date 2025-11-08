import os
import numpy as np
from scipy.interpolate import interp1d
from ligotools.utils import whiten, write_wavfile, reqshift

def test_whiten_identity_on_flat_psd():
    # if PSD is flat (==1), whitening ~ identity in RMS scale (within tolerance)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4096)
    freqs = np.linspace(0, 2048, len(x)//2 + 1)
    psd = np.ones_like(freqs)
    interp_psd = interp1d(freqs, psd, bounds_error=False, fill_value=1.0)
    y = whiten(x, interp_psd, dt=1/4096)
    # normalized variance stays ~1 within tolerance
    assert np.isclose(np.var(y), 1.0, rtol=0.2)

def test_reqshift_preserves_length_and_energy_scale():
    fs = 4096
    t = np.arange(0, 1, 1/fs)
    x = np.sin(2*np.pi*100*t)
    y = reqshift(x, fshift=200, sample_rate=fs)
    assert len(y) == len(x)
    # energy shouldn’t explode
    ratio = np.linalg.norm(y) / np.linalg.norm(x)
    assert 0.5 < ratio < 2.0

def test_write_wavfile_creates_file(tmp_path):
    fs = 4096
    x = np.sin(2*np.pi*200*np.arange(0, 0.1, 1/fs))
    fn = tmp_path / "tone.wav"
    write_wavfile(str(fn), fs, x)
    assert fn.exists()
    assert fn.stat().st_size > 0
