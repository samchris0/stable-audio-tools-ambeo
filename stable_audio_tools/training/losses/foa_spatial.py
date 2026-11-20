"""
Spatial covariance targets for first-order ambisonics (FOA), ACN channel order.

Assumes (B, 4, T) audio with channels [W, Y, Z, X] (ACN) and SN3D normalization.
For N3D, divide the X/Y/Z channels by sqrt(3) first. For FuMa, convert to ACN/SN3D first.

Features per mel band and time step (computed from the 4x4 spatial covariance):
  level : log energy of W                          (1)
  rel   : log energy of Y, Z, X relative to W      (3)
  coh   : normalized cross-correlation, 6 pairs    (6)   [+6 imaginary parts if include_imag]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ACN channel indices
W, Y, Z, X = 0, 1, 2, 3
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_band_matrix(sample_rate, n_fft, n_bands, f_min, f_max):
    """Triangular mel bands over STFT bins, rows normalized to sum to 1 -> (n_bands, n_freq)."""
    n_freq = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freq)
    mels = torch.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_bands + 2).tolist()
    hz = [_mel_to_hz(m) for m in mels]
    fb = torch.zeros(n_bands, n_freq)
    for b in range(n_bands):
        lo, ctr, hi = hz[b], hz[b + 1], hz[b + 2]
        up = (freqs - lo) / (ctr - lo)
        down = (hi - freqs) / (hi - ctr)
        fb[b] = torch.clamp(torch.minimum(up, down), min=0.0)
        if fb[b].sum() == 0:  # band narrower than one STFT bin: use the nearest bin
            fb[b, torch.argmin((freqs - ctr).abs())] = 1.0
    return fb / fb.sum(dim=1, keepdim=True)


class FOASpatialCovariance(nn.Module):
    def __init__(self, sample_rate, n_bands=24, n_fft=None, hop_length=None,
                 f_min=50.0, f_max=None, pool_frames=32, include_imag=False, eps=1e-10):
        super().__init__()
        if n_fft is None:  # ~43 ms window at any sample rate
            n_fft = 2 ** round(math.log2(0.043 * sample_rate))
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.pool_frames = pool_frames
        self.include_imag = include_imag
        self.eps = eps
        nyq = sample_rate / 2
        f_max = min(f_max or nyq, nyq)
        # persistent=False keeps these out of checkpoints, so existing state dicts still load
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("fbank", mel_band_matrix(sample_rate, n_fft, n_bands, f_min, f_max),
                             persistent=False)
        self.n_bands = n_bands
        self.n_feat = 1 + 3 + len(PAIRS) * (2 if include_imag else 1)
        self.out_channels = self.n_feat * n_bands  # for a Conv1d regressor head

    def _stft(self, x):
        B, C, T = x.shape
        assert C == 4, f"expected 4 FOA channels, got {C}"
        S = torch.stft(x.reshape(B * C, T).float(), self.n_fft, self.hop_length,
                       window=self.window, return_complex=True)
        return S.view(B, C, S.shape[-2], S.shape[-1])

    def _pool(self, t, n_frames):
        if n_frames is not None:  # align to a latent sequence length
            return F.adaptive_avg_pool1d(t, n_frames)
        k = min(self.pool_frames, t.shape[-1])
        return F.avg_pool1d(t, k, stride=k, ceil_mode=True)

    def covariance(self, x, n_frames=None):
        """Band-pooled covariance. Returns real and imaginary parts, each (B, 4, 4, bands, T')."""
        S = self._stft(x)
        re, im = {}, {}
        for i in range(4):
            for j in range(i, 4):
                prod = S[:, i] * S[:, j].conj()  # (B, freq, frames)
                re[i, j] = self._pool(torch.einsum("kf,bft->bkt", self.fbank, prod.real), n_frames)
                im[i, j] = self._pool(torch.einsum("kf,bft->bkt", self.fbank, prod.imag), n_frames)
        for i in range(4):  # Hermitian symmetry: C[j,i] = conj(C[i,j])
            for j in range(i):
                re[i, j] = re[j, i]
                im[i, j] = -im[j, i]
        Cr = torch.stack([torch.stack([re[i, j] for j in range(4)], 1) for i in range(4)], 1)
        Ci = torch.stack([torch.stack([im[i, j] for j in range(4)], 1) for i in range(4)], 1)
        return Cr, Ci

    def features(self, x, n_frames=None):
        Cr, Ci = self.covariance(x, n_frames)
        E = torch.stack([Cr[:, i, i] for i in range(4)], 1)  # (B, 4, bands, T')
        logE = torch.log(E + self.eps)
        out = {
            "level": logE[:, 0],                       # (B, bands, T')
            "rel": logE[:, 1:] - logE[:, :1],          # (B, 3, bands, T')
        }
        denom = lambda i, j: torch.sqrt(E[:, i] * E[:, j] + self.eps)
        out["coh"] = torch.stack([Cr[:, i, j] / denom(i, j) for i, j in PAIRS], 1)
        if self.include_imag:
            out["coh_im"] = torch.stack([Ci[:, i, j] / denom(i, j) for i, j in PAIRS], 1)
        return out

    def forward(self, x, n_frames=None):
        """Flat feature tensor (B, out_channels, T') -- the regression target for latents."""
        f = self.features(x, n_frames)
        parts = [f["level"].unsqueeze(1), f["rel"], f["coh"]]
        if self.include_imag:
            parts.append(f["coh_im"])
        B, T = f["level"].shape[0], f["level"].shape[-1]
        return torch.cat(parts, 1).reshape(B, -1, T)


class SpatialCovarianceLoss(nn.Module):
    """Output-side loss: L1 between spatial features of decoded and original audio.
    Call convention is loss(input, target), like auraloss."""

    def __init__(self, sample_rate, level_weight=0.1, rel_weight=1.0, coh_weight=1.0,
                 **extractor_kwargs):
        super().__init__()
        self.extractor = FOASpatialCovariance(sample_rate, **extractor_kwargs)
        self.weights = {"level": level_weight, "rel": rel_weight,
                        "coh": coh_weight, "coh_im": coh_weight}

    def forward(self, input, target):
        fi = self.extractor.features(input)
        with torch.no_grad():
            ft = self.extractor.features(target)
        return sum(self.weights[k] * F.l1_loss(fi[k], ft[k]) for k in fi)


@torch.no_grad()
def foa_report(extractor, x, n_frames=None):
    """Diagnostics for the decode-and-compare test (assumes SN3D).
    Returns per-band diffuseness in [0,1] (0 = single plane wave, 1 = fully diffuse) and the
    azimuth/elevation (radians) of the mean intensity vector, which points toward the apparent source."""
    Cr, _ = extractor.covariance(x, n_frames)
    Ix, Iy, Iz = Cr[:, W, X], Cr[:, W, Y], Cr[:, W, Z]
    energy = 0.5 * sum(Cr[:, i, i] for i in range(4))
    norm = torch.sqrt(Ix ** 2 + Iy ** 2 + Iz ** 2)
    return {
        "diffuseness": (1 - norm / (energy + extractor.eps)).clamp(0, 1),
        "azimuth": torch.atan2(Iy, Ix),
        "elevation": torch.atan2(Iz, torch.hypot(Ix, Iy)),
    }

class FOAVirtualMicLoss(torch.nn.Module):
    """MRSTFT on first-order virtual microphones decoded from ACN/SN3D FOA."""
    def __init__(self, mrstft, directions=None, pattern=0.5):
        super().__init__()
        self.mrstft = mrstft
        if directions is None:  # +/-X, +/-Y, +/-Z (x, y, z)
            directions = torch.tensor([
                [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
            ], dtype=torch.float32)
        d = directions / directions.norm(dim=1, keepdim=True)
        # columns in ACN order: W, Y, Z, X
        D = torch.stack([
            torch.full((d.size(0),), pattern),
            (1 - pattern) * d[:, 1],
            (1 - pattern) * d[:, 2],
            (1 - pattern) * d[:, 0],
        ], dim=1)  # (n_mics, 4)
        self.register_buffer("D", D, persistent=False)

    def forward(self, pred, target):  # (B, 4, T)
        p = torch.einsum("nc,bct->bnt", self.D, pred)
        t = torch.einsum("nc,bct->bnt", self.D, target)
        return self.mrstft(p, t)

if __name__ == "__main__":
    torch.manual_seed(0)
    for sr in (48000, 44100, 16000):
        ext = FOASpatialCovariance(sr)
        T = sr * 4
        s = torch.randn(1, T)
        az = math.radians(60)  # plane wave, ACN order [W, Y, Z, X]
        plane = torch.stack([s, s * math.sin(az), torch.zeros_like(s), s * math.cos(az)], 1)
        diffuse = torch.cat([torch.randn(1, 1, T), torch.randn(1, 3, T) / math.sqrt(3)], 1)

        r = foa_report(ext, plane)
        d = foa_report(ext, diffuse)
        loss_fn = SpatialCovarianceLoss(sr)
        x = diffuse.clone().requires_grad_(True)
        l_same = loss_fn(diffuse, diffuse).item()
        l_diff = loss_fn(x, plane)
        l_diff.backward()
        print(f"sr={sr} n_fft={ext.n_fft} target_ch={ext(plane).shape[1]} "
              f"| plane: diff={r['diffuseness'].mean():.3f} az={math.degrees(r['azimuth'].mean()):.1f} deg "
              f"| diffuse: diff={d['diffuseness'].mean():.3f} "
              f"| loss same={l_same:.4f} vs plane={l_diff.item():.3f} grad_ok={x.grad.abs().sum().item() > 0}")
