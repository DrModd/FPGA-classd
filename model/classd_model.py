"""
Open-loop model of the FPGA class-D modulator (DigiD D1, experimental branch).

Chain:
  PCM 192 kHz -> x8 interpolation -> [feedforward on bus voltage]
  -> error-feedback noise shaper (NTF with optimised zeros, multibit quantiser)
  -> BD (3-level) double-edge PWM, edges on the OSER tick grid
  -> per-edge timing errors: dead time (current-dependent, ZVS-aware) + jitter
  -> exact LC + load + Zobel simulation (for inductor currents at the edges)
  -> differential switch-node voltage, sampled with a triangular kernel
  -> decimation to 192 kHz -> coherent FFT -> THD, THD+N.

The LC filter is linear, so the in-band spectrum of the switch-node voltage
equals the spectrum at the speaker up to the filter's linear response.
Distortion therefore comes only from the modulator, UPWM, dead time,
jitter and bus ripple, which is what this model is for.
"""

from dataclasses import dataclass, field, replace
import numpy as np
from scipy import signal
from scipy.linalg import expm


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

@dataclass
class Params:
    # clocks
    fs_in: float = 192_000.0          # ASRC output rate
    up: int = 8                       # interpolation factor -> shaper/update rate
    f_oser: float = 786_432_000.0     # edge grid (49.152 MHz x16)

    # test signal
    f_sig: float = 1000.0             # snapped to an FFT bin (coherent)
    f_sig2: float = 0.0               # >0: twin tone (each at level - 6 dB)
    level_dbfs: float = -1.0          # relative to full modulation
    n_fft: int = 2 ** 15              # FFT length at fs_in
    pad: int = 4096                   # settling samples at fs_in, both ends

    # noise shaper
    ntf_order: int = 5
    ntf_obg: float = 3.0              # out-of-band NTF gain (multibit -> safe)
    f_band: float = 20_000.0
    quantize: bool = True             # False -> ideal edges (pure UPWM)
    # BD split of the differential level M between the legs:
    #  "even"   - M in steps of 2 ticks, nA + nB == nh always (constant CM)
    #  "parity" - M in steps of 1 tick, nA + nB alternates nh / nh+1.
    #             The pulse centre then moves by 1/2 tick with the parity of
    #             the quantiser noise -> signal-proportional noise floor.
    bd_split: str = "even"
    # ISG3208: input pulses < 10 ns are filtered, kept ones are stretched to
    # 12 ns. Every leg pulse and gap is kept >= min_pulse (incl. dead time
    # margin): |M| <= nh - n_min ticks, input pre-clipped with some headroom
    # so the shaper itself does not clip in normal operation.
    min_pulse: float = 15e-9
    clip_headroom: int = 6            # ticks between input clip and hard limit
    # UPWM linearisation, applied before the shaper:
    #  "none"  - plain uniform sampling
    #  "cubic" - m = x - D2(x^3)/24  (see pwm_linearise)
    pwm_corr: str = "cubic"

    # power stage
    vbus: float = 50.0
    bus_ripple: float = 0.0           # relative amplitude
    f_ripple: float = 100.0
    feedforward: bool = False         # scale modulation by measured bus

    deadtime: float = 0.0             # s
    coss: float = 805e-12             # F, per FET, time-related Coss(TR)
                                      # ISG3208: 805 pF @ 0-50 V (Qoss 40 nC)
    jitter_rms: float = 0.0           # s, white, per edge

    # output filter (per BTL half) + load
    L: float = 4.7e-6
    r_dcr: float = 0.01               # inductor DCR (also damps the CM mode)
    c_gnd: float = 470e-9             # each node to ground
    c_diff: float = 470e-9            # between nodes
    r_load: float = 8.0               # differential, np.inf = open
    zobel_r: float = 4.7
    zobel_c: float = 470e-9           # 0 -> no Zobel

    # analysis grid (multiple of fs_in, divisible by 8)
    grid_ratio: int = 128

    @property
    def f_upd(self):
        return self.fs_in * self.up

    @property
    def f_pwm(self):
        return self.f_upd / 2

    @property
    def nh(self):                      # ticks per half period
        return int(round(self.f_oser / self.f_upd))

    @property
    def tick(self):
        return 1.0 / (self.f_upd * self.nh)


# ----------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------

def kaiser_lp(fs, f_pass, f_stop, atten_db):
    beta = 0.1102 * (atten_db - 8.7)
    df = (f_stop - f_pass) / fs
    n = int(np.ceil((atten_db - 7.95) / (14.36 * df))) | 1
    return signal.firwin(n, (f_pass + f_stop) / 2, window=("kaiser", beta), fs=fs)


def interpolate(x, p):
    h = kaiser_lp(p.f_upd, 24_000.0, p.fs_in - 24_000.0, 150.0) * p.up
    return signal.upfirdn(h, x, up=p.up)


# ----------------------------------------------------------------------------
# NTF design: optimised zeros + Butterworth-high-pass poles, tuned for OBG
# ----------------------------------------------------------------------------

_OPT_ZEROS = {  # normalised to the band edge (Schreier)
    1: [0.0], 2: [0.577], 3: [0.0, 0.775], 4: [0.340, 0.861],
    5: [0.0, 0.539, 0.906], 6: [0.240, 0.711, 0.946],
}


def design_ntf(p):
    L = p.ntf_order
    zeros = []
    for r in _OPT_ZEROS[L]:
        w = 2 * np.pi * r * p.f_band / p.f_upd
        zeros += [1.0 + 0j] if r == 0 else [np.exp(1j * w), np.exp(-1j * w)]
    zeros = np.array(zeros)

    w_grid = np.linspace(0, np.pi, 8192)
    zg = np.exp(1j * w_grid)

    def ntf_for(fc):
        _, poles, _ = signal.butter(L, fc, "high", output="zpk", fs=p.f_upd)
        num = np.prod(zg[:, None] - zeros[None, :], axis=1)
        den = np.prod(zg[:, None] - poles[None, :], axis=1)
        return poles, np.max(np.abs(num / den))

    lo, hi = 1e3, 0.45 * p.f_upd
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        _, g = ntf_for(mid)
        lo, hi = (mid, hi) if g < p.ntf_obg else (lo, mid)
    poles, _ = ntf_for(lo)
    b = np.real(np.poly(zeros))
    a = np.real(np.poly(poles))
    return b, a


def noise_shaper(x_units, p, lim=None, e_max=2.0):
    """Error feedback: y = Q(x + H e), H = NTF - 1. Returns integer levels.
    On clipping the fed-back error is clamped to +-e_max LSB so the loop
    filter state stays bounded (no limit cycles after overload)."""
    b, a = design_ntf(p)
    h_b = b - a                       # numerator of H, h_b[0] == 0
    n = len(a) - 1
    z = [0.0] * (n + 1)
    if lim is None:
        lim = p.nh
    out = np.empty(len(x_units), dtype=np.int64)
    hb = h_b.tolist()
    aa = a.tolist()
    for k, xv in enumerate(x_units.tolist()):
        w = z[0]
        v = xv + w
        q = int(round(v))
        if q > lim:
            q = lim
        elif q < -lim:
            q = -lim
        e = q - v
        if e > e_max:
            e = e_max
        elif e < -e_max:
            e = -e_max
        # transposed DF-II update for H = h_b / a, driven by e, output w
        for i in range(n):
            z[i] = hb[i + 1] * e - aa[i + 1] * w + z[i + 1]
        out[k] = q
    return out


# ----------------------------------------------------------------------------
# BD-PWM edge generation on the tick grid
# ----------------------------------------------------------------------------

def leg_edges(n_high, p):
    """n_high[k]: ticks high in half period k. Even halves: high at the end,
    odd halves: high at the start (centre-aligned double-edge PWM).
    Returns edge times (ticks, int) and new levels (0/1)."""
    nh = p.nh
    k = np.arange(len(n_high))
    even = (k % 2) == 0
    d1 = np.where(even, nh - n_high, n_high)
    l1 = np.where(even, 0, 1)
    starts = np.empty(2 * len(k), dtype=np.int64)
    levels = np.empty(2 * len(k), dtype=np.int8)
    durs = np.empty(2 * len(k), dtype=np.int64)
    starts[0::2] = k * nh
    starts[1::2] = k * nh + d1
    levels[0::2] = l1
    levels[1::2] = 1 - l1
    durs[0::2] = d1
    durs[1::2] = nh - d1
    m = durs > 0
    starts, levels = starts[m], levels[m]
    prev = np.concatenate(([0], levels[:-1]))
    ch = levels != prev
    return starts[ch], levels[ch]


def split_bd(m_units, nh):
    """Differential level M (ticks per half) -> per-leg high ticks (BD)."""
    m = np.clip(m_units, -nh, nh)
    na = (nh + m + 1) // 2
    nb = na - m
    return np.clip(na, 0, nh), np.clip(nb, 0, nh)


# ----------------------------------------------------------------------------
# Output filter state space: x = [iL1, iL2, v1, v2, v_zobel]
# ----------------------------------------------------------------------------

def lc_state_space(p):
    g_load = 0.0 if not np.isfinite(p.r_load) else 1.0 / p.r_load
    use_z = p.zobel_c > 0
    Cm = np.array([[p.c_gnd + p.c_diff, -p.c_diff],
                   [-p.c_diff, p.c_gnd + p.c_diff]])
    Ci = np.linalg.inv(Cm)
    A = np.zeros((5, 5))
    B = np.zeros((5, 2))
    A[0, 0] = -p.r_dcr / p.L
    A[1, 1] = -p.r_dcr / p.L
    A[0, 2] = -1 / p.L
    A[1, 3] = -1 / p.L
    B[0, 0] = 1 / p.L
    B[1, 1] = 1 / p.L
    # node currents: i_node = [i1, i2] + G * [v1, v2] + zobel
    gz = 1.0 / p.zobel_r if use_z else 0.0
    gtot = g_load + gz
    J = np.zeros((2, 5))
    J[0, 0] = 1
    J[1, 1] = 1
    J[0, 2], J[0, 3] = -gtot, gtot
    J[1, 2], J[1, 3] = gtot, -gtot
    J[0, 4], J[1, 4] = gz, -gz
    A[2:4, :] = Ci @ J
    if use_z:
        A[4, 2], A[4, 3], A[4, 4] = gz / p.zobel_c, -gz / p.zobel_c, -gz / p.zobel_c
    else:
        A[4, 4] = -1.0
    return A, B


class Discretiser:
    def __init__(self, A, B, tick):
        self.A, self.B, self.tick = A, B, tick
        self.cache = {}
        n, m = B.shape
        self.M = np.zeros((n + m, n + m))
        self.M[:n, :n] = A
        self.M[:n, n:] = B

    def __call__(self, nticks):
        r = self.cache.get(nticks)
        if r is None:
            E = expm(self.M * nticks * self.tick)
            n = self.A.shape[0]
            r = (E[:n, :n].copy(), E[:n, n:].copy())
            self.cache[nticks] = r
        return r


def simulate_currents(ta, la, tb, lb, vbus_fn, p):
    """Inductor current (out of the switch node) at every edge of both legs."""
    A, B = lc_state_space(p)
    disc = Discretiser(A, B, p.tick)
    t = np.concatenate((ta, tb))
    leg = np.concatenate((np.zeros(len(ta), np.int8), np.ones(len(tb), np.int8)))
    lvl = np.concatenate((la, lb))
    order = np.argsort(t, kind="stable")
    x = np.array([0.0, 0.0, p.vbus / 2, p.vbus / 2, 0.0])  # start at CM operating point
    u = [0, 0]
    t_prev = 0
    cur = np.empty(len(t))
    vb_cache = vbus_fn(t[order] * p.tick)
    for j, idx in enumerate(order.tolist()):
        tt = int(t[idx])
        dt = tt - t_prev
        if dt:
            Phi, Gam = disc(dt)
            vb = vb_cache[j]
            x = Phi @ x + Gam @ np.array([u[0] * vb, u[1] * vb])
            t_prev = tt
        g = leg[idx]
        cur[idx] = x[g]
        u[g] = lvl[idx]
    return cur[:len(ta)], cur[len(ta):]


# ----------------------------------------------------------------------------
# Edge timing errors
# ----------------------------------------------------------------------------

def edge_shift(levels, i_out, vb, p, rng):
    """Area-equivalent edge delay [s]. i_out > 0 flows out of the node.
    Rising edges are helped by i_out < 0, falling edges by i_out > 0."""
    shift = np.zeros(len(levels))
    if p.deadtime > 0:
        rising = levels == 1
        aiding = np.where(rising, i_out < 0, i_out > 0)
        with np.errstate(divide="ignore"):
            t_tr = 2 * p.coss * vb / np.abs(i_out)
        td = p.deadtime
        frac = np.minimum(td / t_tr, 1.0)
        aid_shift = np.where(t_tr <= td, t_tr / 2, td * (1 - frac / 2))
        shift = np.where(aiding, aid_shift, td)
    if p.jitter_rms > 0:
        shift = shift + rng.normal(0, p.jitter_rms, len(levels))
    return shift


# ----------------------------------------------------------------------------
# Triangular-kernel sampling of a step waveform
# ----------------------------------------------------------------------------

def sample_steps(tau, delta, T, n_out):
    """y[k] = sum_j delta_j * F(kT - tau_j), F = CDF of a triangle of half
    width T. Exact second-order B-spline anti-aliasing of a step signal."""
    pos = tau / T
    m = np.floor(pos).astype(np.int64)
    phi = pos - m
    y = np.zeros(n_out + 2)
    ok = (m >= 0) & (m + 1 < n_out + 2)
    m, phi, d = m[ok], phi[ok], delta[ok]
    np.add.at(y, m + 1, d)
    y = np.cumsum(y)
    np.add.at(y, m, d * (1 - phi) ** 2 / 2)
    np.add.at(y, m + 1, -d * phi ** 2 / 2)
    return y[:n_out]


def decimate_to_fs(y, p, grid_ratio):
    """grid (fs_in * grid_ratio) -> fs_in, two Kaiser stages."""
    f_grid = p.fs_in * grid_ratio
    r1 = grid_ratio // 8
    f1 = f_grid / r1
    h1 = kaiser_lp(f_grid, 100e3, f1 - 100e3, 160.0)
    y = signal.upfirdn(h1, y, down=r1)
    h2 = kaiser_lp(f1, 22e3, 100e3, 160.0)
    return signal.upfirdn(h2, y, down=8)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------

def analyse(y, p, kf):
    """kf: fundamental bin, or (k1, k2) for a twin tone. For a twin tone
    'thd' holds the in-band IMD (2f1-f2, 2f2-f1, f2-f1, 3rd/2nd order
    products) and 'thdn' everything except the two tones, both relative
    to the total rms of the tones."""
    n = p.n_fft
    Y = np.fft.rfft(y) / n * 2
    f = np.arange(len(Y)) * p.fs_in / n
    mag = np.abs(Y)
    band = (f >= 20) & (f <= p.f_band)
    tones = list(kf) if isinstance(kf, tuple) else [kf]
    sig_rms = np.sqrt(np.sum(mag[tones] ** 2) / 2)
    band[tones] = False
    noise_dist = np.sqrt(np.sum(mag[band] ** 2) / 2)
    if len(tones) == 1:
        k1 = tones[0]
        prods = [h * k1 for h in range(2, 10)]
    else:
        k1, k2 = tones
        prods = [2 * k1 - k2, 2 * k2 - k1, k2 - k1]
    harm = np.array([mag[k] for k in prods if 0 < k < len(mag) and f[k] <= p.f_band])
    thd = np.sqrt(np.sum(harm ** 2) / 2) / sig_rms if len(harm) else 0.0
    return dict(f=f, mag=mag, fund=mag[tones[0]], thd=thd,
                thdn=noise_dist / sig_rms,
                harm_db=20 * np.log10(harm / (sig_rms * np.sqrt(2)) + 1e-30))


def run(p: Params, seed=1):
    rng = np.random.default_rng(seed)
    n_in = p.n_fft + 2 * p.pad
    kf = int(round(p.f_sig * p.n_fft / p.fs_in))
    amp = 10 ** (p.level_dbfs / 20)
    n = np.arange(n_in)
    if p.f_sig2 > 0:
        kf2 = int(round(p.f_sig2 * p.n_fft / p.fs_in))
        x = amp / 2 * (np.sin(2 * np.pi * kf * n / p.n_fft)
                       + np.sin(2 * np.pi * kf2 * n / p.n_fft))
        kf = (kf, kf2)
    else:
        x = amp * np.sin(2 * np.pi * kf * n / p.n_fft)

    xi = interpolate(x, p)[: n_in * p.up]
    t_upd = np.arange(len(xi)) / p.f_upd

    def vbus_fn(t):
        return p.vbus * (1 + p.bus_ripple * np.sin(2 * np.pi * p.f_ripple * t))

    m = xi
    if p.feedforward:
        m = m * p.vbus / vbus_fn(t_upd + 0.5 / p.f_upd)
    n_min = int(np.ceil(p.min_pulse / p.tick - 1e-9))
    m_lim = p.nh - n_min                      # max |M| in ticks
    m_in = (m_lim - p.clip_headroom) / p.nh   # input clip (normalised)
    m = np.clip(m, -m_in, m_in)
    m = pwm_linearise(m, p.pwm_corr)
    m_units = m * p.nh

    if p.quantize:
        if p.bd_split == "even":
            q = 2 * noise_shaper(m_units / 2, replace(p, f_oser=p.f_oser / 2),
                                 lim=m_lim // 2)
        else:
            q = noise_shaper(m_units, p, lim=m_lim)
        na, nb = split_bd(q, p.nh)
        ta, la = leg_edges(na, p)
        tb, lb = leg_edges(nb, p)
        ta_f, tb_f = ta.astype(float), tb.astype(float)
    else:
        # ideal (unquantised) edges, centre-aligned, still uniformly sampled
        na = (1 + np.clip(m_units / p.nh, -1, 1)) / 2 * p.nh
        nb = p.nh - na
        ta, la, ta_f = _ideal_edges(na, p)
        tb, lb, tb_f = _ideal_edges(nb, p)

    vb_a = vbus_fn(ta_f * p.tick)
    vb_b = vbus_fn(tb_f * p.tick)
    if p.deadtime > 0:
        ia, ib = simulate_currents(ta, la, tb, lb, vbus_fn, p)
    else:
        ia, ib = np.zeros(len(ta)), np.zeros(len(tb))
    ta_f = ta_f + edge_shift(la, ia, vb_a, p, rng) / p.tick
    tb_f = tb_f + edge_shift(lb, ib, vb_b, p, rng) / p.tick

    grid_ratio = p.grid_ratio
    T = p.f_oser / (p.fs_in * grid_ratio)          # grid period in ticks
    n_grid = n_in * grid_ratio
    tau = np.concatenate((ta_f, tb_f))
    delta = np.concatenate((np.where(la == 1, 1.0, -1.0),
                            np.where(lb == 1, -1.0, 1.0)))
    yd = sample_steps(tau, delta, T, n_grid)
    tg = np.arange(n_grid) / (p.fs_in * grid_ratio)
    yd = yd * vbus_fn(tg) / p.vbus

    yo = decimate_to_fs(yd, p, grid_ratio)
    start = len(yo) - p.n_fft - p.pad // 2
    seg = yo[start: start + p.n_fft]
    res = analyse(seg, p, kf)
    res["params"] = p
    res["min_pulse_ticks"] = min(_min_pulse(ta), _min_pulse(tb))
    res["n_min"] = n_min
    res["switch_rate"] = (len(ta) + len(tb)) / (n_in / p.fs_in)
    # commanded (int ticks) and actual (float ticks) edges, new levels
    res["edges"] = dict(ta=ta, tb=tb, ta_f=ta_f, tb_f=tb_f, la=la, lb=lb)
    return res


def _min_pulse(t_edges):
    """Shortest high or low interval of a leg (ticks)."""
    d = np.diff(np.asarray(t_edges))
    return int(d.min()) if len(d) else 10 ** 9


def pwm_linearise(x, mode):
    """Pre-correction of the uniformly sampled BD PWM.

    With the constant-CM split every half period k carries one differential
    pulse of width w_k (in half periods, w = m) centred at a FIXED position.
    Its low-frequency spectrum is
        2/w * sin(w*w/2) * e^{-j w c} = (w - w^3 w^2/24 + ...) e^{-j w c},
    i.e. the pulse train equals the sample sequence w_k plus (1/24) d^2/dt^2
    of w_k^3 (time unit = half period). So the baseband sees
        w_k + D2(w^3)_k / 24,   D2 = second difference,
    and the inverse is m = x - D2(x^3)/24 (error ~ w^5 term, below -140 dB).
    FPGA cost: a cube, a second difference, a shift-add (1/24)."""
    if mode == "none":
        return x
    if mode == "cubic":
        c = x ** 3
        d2 = np.concatenate((c[:1], c[:-1])) - 2 * c + np.concatenate((c[1:], c[-1:]))
        return x - d2 / 24
    raise ValueError(mode)


def _ideal_edges(n_high, p):
    nh = p.nh
    k = np.arange(len(n_high))
    even = (k % 2) == 0
    t_edge = np.where(even, k * nh + (nh - n_high), k * nh + n_high)
    lvl = np.where(even, 1, 0).astype(np.int8)
    return np.round(t_edge).astype(np.int64), lvl, t_edge.astype(float)


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-30))
