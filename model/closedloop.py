"""
Closed-loop model of the FPGA class-D amplifier.

Loop timing (one sample = one PWM half period, 1.536 MHz):
  half k:     PWM edges from q[k]; power stage and LC run in continuous time
  end of k:   measurements of half k are available
                y_pre[k]  - area of the differential switch-node voltage
                            (what a sinc1-decimated sigma-delta ADC sees)
                y_post[k] - filtered output, boxcar over half k (order 1) or
                            triangle over halves k-1, k (order 2, less aliasing)
  half k+1+D: correction computed from measurements of half k is applied

Loops:
  "none" - open loop
  "pre"  - local loop on the switch-node area. Error = target area (reference
           plus outer-loop correction, without its own correction) - measured
           area. Corrects power-stage errors: dead time, bus voltage, edge
           timing. Plant = pure delay.
  "post" - global loop after the LC filter. 2-DOF: error = reference passed
           through the nominal (8 Ohm) plant model - measured output, so the
           loop corrects deviations only and does not fight the intended
           response.
  "dual" - both.

Linear design (controllers, margins) uses the exact discrete plant: LC + load
+ Zobel, PWM pulse = area impulse at the centre of the half, measurement
kernel. The time-domain simulation is the full nonlinear one: BD PWM on the
tick grid, ZVS-aware dead time from the instantaneous inductor current, bus
ripple, ADC noise, quantised shaper inside the loop.

Output analysis: the filtered output is sampled with a triangular kernel at
4 points per half (6.144 MHz), so the residual carrier at 1.536 MHz cannot
alias into the audio band (a 1.536 MHz kernel would fold the carrier
sidebands straight onto the signal harmonics).
"""

from dataclasses import dataclass, field, replace
import numpy as np
from scipy import signal
from scipy.linalg import expm

from classd_model import (Params, interpolate, pwm_linearise, design_ntf,
                          lc_state_space, kaiser_lp, analyse, db)

SUB = 4                               # analysis sub-intervals per half


# ----------------------------------------------------------------------------
# Loop parameters
# ----------------------------------------------------------------------------

@dataclass
class LoopParams:
    mode: str = "dual"                # none | pre | post | dual
    delay: int = 1                    # extra halves of latency (D)
    post_order: int = 2               # post measurement kernel: 1 boxcar, 2 triangle
    adc_noise: float = 0.0            # rms per measurement, normalised (1 = Vbus)
    # C(s) = K / s * ((s + wz) / s)^(n_int - 1) [* LC compensation], bilinear
    pre: dict = field(default_factory=lambda: dict(n_int=3, fc=67e3, fz_ratio=0.2))
    post: dict = field(default_factory=lambda: dict(n_int=2, fc=17e3, fz_ratio=0.2))
    r_nominal: float = 8.0            # load of the reference model (post loop)
    # pre-loop measurement kernel: "sinc1" area of half k, "sinc2" triangle
    # over halves k-1, k (2nd-order nulls, +0.5 half of delay)
    pre_kernel: str = "sinc1"
    adc_err: object = None            # per-half measurement error to inject


# ----------------------------------------------------------------------------
# Discrete plant: area impulse at the half centre -> measured output
# ----------------------------------------------------------------------------

def _half_maps(p: Params):
    """Per half: z_{k+1} = Phi z + Gam m;  a = Ca z + Da m (integral of the
    normalised output over the half), b = Cb z + Db m (double integral)."""
    A, B = lc_state_space(p)
    n = A.shape[0]
    th = 1.0 / p.f_upd
    out = np.zeros(n)
    out[2], out[3] = 1.0 / p.vbus, -1.0 / p.vbus
    Aa = np.zeros((n + 2, n + 2))
    Aa[:n, :n] = A
    Aa[n, :n] = out
    Aa[n + 1, n] = 1.0
    Ph = expm(Aa * th / 2)
    b_imp = np.zeros(n + 2)
    # differential area impulse of 1 (normalised): leg A +1/2, leg B -1/2
    b_imp[:n] = B @ np.array([0.5 * th * p.vbus, -0.5 * th * p.vbus])
    P2 = Ph @ Ph
    g = Ph @ b_imp
    return (P2[:n, :n], g[:n], P2[n, :n], g[n], P2[n + 1, :n], g[n + 1], th)


def plant_discrete(p: Params, order=1):
    """State space (Phi, Gam, Cz, Dz) from m[k] to the post measurement."""
    Phi, Gam, Ca, Da, Cb, Db, th = _half_maps(p)
    n = Phi.shape[0]
    if order == 1:
        return Phi, Gam, Ca / th, Da / th
    # order 2: y_k = (w_k + b_k) / th^2, w_{k+1} = th a_k - b_k
    P = np.zeros((n + 1, n + 1))
    P[:n, :n] = Phi
    P[n, :n] = th * Ca - Cb
    G = np.concatenate((Gam, [th * Da - Db]))
    C = np.concatenate((Cb, [1.0])) / th ** 2
    D = Db / th ** 2
    return P, G, C, D


def plant_freq(Phi, Gam, Cz, Dz, w):
    n = Phi.shape[0]
    I = np.eye(n)
    return np.array([Cz @ np.linalg.solve(np.exp(1j * ww) * I - Phi, Gam) + Dz
                     for ww in w])


def plant_dc(Phi, Gam, Cz, Dz):
    return float(Cz @ np.linalg.solve(np.eye(len(Phi)) - Phi, Gam) + Dz)


# ----------------------------------------------------------------------------
# Controllers and loop analysis
# ----------------------------------------------------------------------------

def controller_tf(cfg, f_upd, gain_dc=1.0):
    """Analog prototype K/s * ((s+wz)/s)^(n-1), optional LC compensation
    (complex zero pair at f_lc with damping zeta_z, two real poles at f_p),
    bilinear transform at f_upd. K sets |C(j wc)| * gain_dc = 1."""
    n = cfg.get("n_int", 1)
    wc = 2 * np.pi * cfg["fc"]
    wz = wc * cfg.get("fz_ratio", 0.25)
    zeros = [-wz] * (n - 1)
    poles = [0.0] * n
    if cfg.get("f_lc"):
        w0 = 2 * np.pi * cfg["f_lc"]
        zt = cfg.get("zeta_z", 0.5)
        zeros += list(np.roots([1, 2 * zt * w0, w0 ** 2]))
        wp = 2 * np.pi * cfg.get("f_p", 4 * cfg["f_lc"])
        poles += [-wp, -wp]
    s = 1j * wc
    mag = abs(np.prod([s - z for z in zeros]) / np.prod([s - q for q in poles]))
    k = 1.0 / mag / gain_dc
    zd, pd, kd = signal.bilinear_zpk(np.array(zeros), np.array(poles), k, f_upd)
    b, a = signal.zpk2tf(zd, pd, kd)
    return np.real(b), np.real(a)


def ctrl_freq(b, a, w):
    z1 = np.exp(-1j * w)
    return np.polyval(b[::-1], z1) / np.polyval(a[::-1], z1)


def closed_loop_stable(Phi, Gam, Cz, Dz, b, a, delay):
    """Spectral radius of plant + controller + delay line (e = -y)."""
    Ac, Bc, Cc, Dc = signal.tf2ss(b, a)
    n, nc, nd = Phi.shape[0], Ac.shape[0], 1 + delay
    N = n + nc + nd
    M = np.zeros((N, N))
    im = n + nc + nd - 1              # m_k = oldest entry of the delay line
    M[:n, :n] = Phi
    M[:n, im] += Gam
    M[n:n + nc, :n] = -np.outer(Bc[:, 0], Cz)
    M[n:n + nc, n:n + nc] = Ac
    M[n:n + nc, im] += -Bc[:, 0] * Dz
    r0 = n + nc
    M[r0, :n] = -Dc[0, 0] * Cz
    M[r0, n:n + nc] = Cc[0]
    M[r0, im] += -Dc[0, 0] * Dz
    for i in range(1, nd):
        M[r0 + i, r0 + i - 1] = 1.0
    return np.max(np.abs(np.linalg.eigvals(M)))


# ----------------------------------------------------------------------------
# Time-domain building blocks
# ----------------------------------------------------------------------------

class Disc:
    """exp(A t) and its input integral, cached on a 1/64-tick grid."""

    def __init__(self, A, B, tick, sub=64):
        n, m = B.shape
        self.n = n
        self.M = np.zeros((n + m, n + m))
        self.M[:n, :n] = A
        self.M[:n, n:] = B
        self.tick, self.sub = tick, sub
        self.cache = {}

    def step(self, x, u, dt_ticks):
        key = int(round(dt_ticks * self.sub))
        if key <= 0:
            return x
        r = self.cache.get(key)
        if r is None:
            E = expm(self.M * key / self.sub * self.tick)
            r = (E[:self.n, :self.n].copy(), E[:self.n, self.n:].copy())
            self.cache[key] = r
        return r[0] @ x + r[1] @ u


class Shaper:
    """Error-feedback shaper of the open-loop model, one sample at a time."""

    def __init__(self, p, lim):
        b, a = design_ntf(p)
        self.hb = (b - a).tolist()
        self.a = a.tolist()
        self.n = len(a) - 1
        self.z = [0.0] * (self.n + 1)
        self.lim = lim

    def __call__(self, x):
        z, hb, a = self.z, self.hb, self.a
        w = z[0]
        v = x + w
        q = max(-self.lim, min(self.lim, int(round(v))))
        e = max(-2.0, min(2.0, q - v))
        for i in range(self.n):
            z[i] = hb[i + 1] * e - a[i + 1] * w + z[i + 1]
        return q


class IIR:
    """Transposed direct form II, one sample at a time."""

    def __init__(self, b, a):
        self.b = np.asarray(b, float) / a[0]
        self.a = np.asarray(a, float) / a[0]
        n = max(len(self.a), len(self.b))
        self.b = np.pad(self.b, (0, n - len(self.b)))
        self.a = np.pad(self.a, (0, n - len(self.a)))
        self.z = np.zeros(n)

    def __call__(self, x):
        b, a, z = self.b, self.a, self.z
        y = b[0] * x + z[0]
        n = len(z)
        for i in range(1, n):
            z[i - 1] = b[i] * x - a[i] * y + (z[i] if i < n - 1 else 0.0)
        return y


def design_loops(p: Params, lp: LoopParams):
    out = {}
    if lp.mode in ("pre", "dual"):
        out["pre"] = controller_tf(lp.pre, p.f_upd)
    if lp.mode in ("post", "dual"):
        ss = plant_discrete(replace(p, r_load=lp.r_nominal), lp.post_order)
        out["post"] = controller_tf(lp.post, p.f_upd, gain_dc=abs(plant_dc(*ss)))
        num, den = signal.ss2tf(ss[0], ss[1][:, None], ss[2][None, :],
                                np.array([[ss[3]]]))
        out["ref_model"] = (np.real(num[0]), np.real(den))
    return out


def test_signal(p):
    n_in = p.n_fft + 2 * p.pad
    kf = int(round(p.f_sig * p.n_fft / p.fs_in))
    amp = 10 ** (p.level_dbfs / 20)
    n = np.arange(n_in)
    if p.f_sig2 > 0:
        kf2 = int(round(p.f_sig2 * p.n_fft / p.fs_in))
        x = amp / 2 * (np.sin(2 * np.pi * kf * n / p.n_fft)
                       + np.sin(2 * np.pi * kf2 * n / p.n_fft))
        return x, (kf, kf2)
    return amp * np.sin(2 * np.pi * kf * n / p.n_fft), kf


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------

def simulate(p: Params, lp: LoopParams, seed=1):
    rng = np.random.default_rng(seed)
    x, kf = test_signal(p)
    n_in = len(x)
    r = interpolate(x, p)[: n_in * p.up]

    nh = p.nh
    n_min = int(np.ceil(p.min_pulse / p.tick - 1e-9))
    m_lim = nh - n_min
    m_in = (m_lim - p.clip_headroom) / nh
    r = pwm_linearise(np.clip(r, -m_in, m_in), p.pwm_corr)

    loops = design_loops(p, lp)
    c_pre = IIR(*loops["pre"]) if "pre" in loops else None
    c_post = IIR(*loops["post"]) if "post" in loops else None
    rm = signal.lfilter(*loops["ref_model"], r) if "ref_model" in loops else None
    shaper = Shaper(replace(p, f_oser=p.f_oser / 2), m_lim // 2)

    A, B = lc_state_space(p)
    ns = A.shape[0]
    # plant + local integrals of the normalised output (reset every sub-interval)
    Aa = np.zeros((ns + 2, ns + 2))
    Aa[:ns, :ns] = A
    Aa[ns, 2], Aa[ns, 3] = 1.0 / p.vbus, -1.0 / p.vbus
    Aa[ns + 1, ns] = 1.0
    Ba = np.zeros((ns + 2, 2))
    Ba[:ns] = B
    disc = Disc(Aa, Ba, p.tick)
    th = 1.0 / p.f_upd
    ts = th / SUB
    ns_tick = nh // SUB

    xs = np.zeros(ns + 2)
    xs[2] = xs[3] = p.vbus / 2
    uA = uB = 0
    K = len(r)
    sub_a = np.zeros(K * SUB)
    sub_b = np.zeros(K * SUB)
    corr_pre = np.zeros(K + 2 + lp.delay)
    corr_post = np.zeros(K + 2 + lp.delay)
    w_prev = 0.0
    m_prev = t_prev = 0.0
    td, coss = p.deadtime, p.coss
    two_pi_fr = 2 * np.pi * p.f_ripple

    for k in range(K):
        vb = p.vbus * (1 + p.bus_ripple * np.sin(two_pi_fr * k * th))
        target = r[k] + corr_post[k]
        m = target + corr_pre[k]
        q = 2 * shaper(m * nh / 2)
        na = (nh + q) // 2
        nb = na - q
        if k % 2 == 0:                          # rising edges, high at the end
            ev = [(nh - na, 0, 1), (nh - nb, 1, 1)]
        else:                                   # falling edges, high at start
            ev = [(na, 0, 0), (nb, 1, 0)]
        ev.sort()
        t_cur, area, mom, j_sub, ei = 0.0, 0.0, 0.0, 0, 0
        a_half, b_half = 0.0, 0.0
        while j_sub < SUB:
            u = np.array([uA * vb, uB * vb])
            t_b = (j_sub + 1) * ns_tick
            if ei < len(ev) and ev[ei][0] < t_b:
                tc, leg, lvl = ev[ei]
                xc = disc.step(xs, u, tc - t_cur)
                i_out = xc[leg]
                sh = 0.0
                if td > 0:
                    aiding = (i_out < 0) if lvl == 1 else (i_out > 0)
                    if aiding:
                        t_tr = 2 * coss * vb / max(abs(i_out), 1e-9)
                        sh = t_tr / 2 if t_tr <= td else td * (1 - (td / t_tr) / 2)
                    else:
                        sh = td
                if p.jitter_rms > 0:
                    sh += rng.normal(0, p.jitter_rms)
                te = min(tc + sh / p.tick, t_b - 1e-6)
                xs = disc.step(xs, u, te - t_cur)
                area += (uA - uB) * (te - t_cur)
                mom += (uA - uB) * (te * te - t_cur * t_cur) / 2
                t_cur = te
                if leg == 0:
                    uA = lvl
                else:
                    uB = lvl
                ei += 1
                continue
            xs = disc.step(xs, u, t_b - t_cur)
            area += (uA - uB) * (t_b - t_cur)
            mom += (uA - uB) * (t_b * t_b - t_cur * t_cur) / 2
            t_cur = t_b
            ia, ib = xs[ns], xs[ns + 1]
            idx = k * SUB + j_sub
            sub_a[idx], sub_b[idx] = ia, ib
            b_half += ib + a_half * ts
            a_half += ia
            xs[ns] = xs[ns + 1] = 0.0
            j_sub += 1

        a_k = area / nh * vb / p.vbus
        m_k = mom / nh ** 2 * vb / p.vbus
        if lp.pre_kernel == "sinc2":
            y_pre = m_prev + a_k - m_k
            t_meas = 0.5 * (t_prev + target)
        else:
            y_pre = a_k
            t_meas = target
        m_prev, t_prev = m_k, target
        if lp.adc_err is not None and k < len(lp.adc_err):
            y_pre += lp.adc_err[k]
        if lp.post_order == 1:
            y_post = a_half / th
        else:
            y_post = (w_prev + b_half) / th ** 2
        w_prev = th * a_half - b_half
        if lp.adc_noise > 0:
            y_pre += rng.normal(0, lp.adc_noise)
            y_post += rng.normal(0, lp.adc_noise)
        if c_pre is not None:
            corr_pre[k + 1 + lp.delay] = c_pre(t_meas - y_pre)
        if c_post is not None:
            corr_post[k + 1 + lp.delay] = c_post(rm[k] - y_post)

    # triangular kernel at SUB points per half: y_j = (ts a_{j-1} - b_{j-1} + b_j)/ts^2
    yo = (ts * sub_a[:-1] - sub_b[:-1] + sub_b[1:]) / ts ** 2
    f_o = p.f_upd * SUB
    h1 = kaiser_lp(f_o, 100e3, p.f_upd - 100e3, 160.0)
    y1 = signal.upfirdn(h1, yo, down=SUB)
    h2 = kaiser_lp(p.f_upd, 22e3, 100e3, 160.0)
    y192 = signal.upfirdn(h2, y1, down=p.up)
    start = len(y192) - p.n_fft - p.pad // 2
    res = analyse(y192[start: start + p.n_fft], p, kf)
    res["params"], res["loop"] = p, lp
    return res
