"""
Behavioural model of the discrete sigma-delta ADC for the "pre" feedback loop
(measurement of the bridge output before the LC filter).

The ADC input is the real differential switch-node waveform of the open-loop
model (BD PWM, dead time, 4 Ohm), normalised to Vbus. Two input options:

  direct  : u = 0.8 * v_sw                       (ADC full scale = 1.25 * Vbus)
  replica : u = G * (v_sw - v_ref)               v_ref = the commanded PWM, made
            by the FPGA as a clean logic-level copy (delay matched) and
            subtracted at the integrator input, so the ADC only sees the
            error of the power stage (PEDEC-like) and can run G times finer.

Modulators (continuous time, 1-bit, NRZ DAC from a flip-flop, excess loop
delay ELD, white clock jitter on the DAC edges, comparator input noise):

  P1   passive RC integrator (the first schematic): leak tau = R||C
  A1   active integrator (op-amp), 1st order
  A2   2nd order CIFB, both integrators active, ELD compensated (k0)
  AP2  2nd order, active first + passive second integrator

Decimation to one value per PWM half period (1.536 MHz):
  sinc1  box over the half       (no extra delay, what the closed loop used)
  sinc2  triangle over two halves (2nd-order nulls, +0.5 half of delay)

Result: in-band (20 Hz - 20 kHz) noise of the measured sequence minus the exact
one, referred to the amplifier full scale -> "measurement floor, dB", which in
the closed loop appears at the output 1:1 wherever the loop gain is high.
"""

import os
import sys
import numpy as np
from dataclasses import dataclass, replace
from scipy import signal
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(__file__))
from classd_model import Params, run, db  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


@dataclass
class AdcParams:
    kind: str = "A2"              # P1 | A1 | A2 | AP2
    fs: float = 49.152e6
    eld: float = 4e-9             # comparator + flip-flop delay
    jitter: float = 0.0           # rms, white, per DAC edge
    comp_noise: float = 1e-3      # comparator input noise, state units
    tau1: float = 1.9e-6          # P1: R||C of the passive integrator
    tau2: float = 1.0e-6          # AP2: passive second integrator
    obg: float = 1.5              # 2nd order NTF out-of-band gain
    mode: str = "replica"         # direct | replica
    gain: float = 16.0            # replica gain G (direct uses 0.8)
    replica_skew: float = 0.5e-9  # residual delay mismatch of the replica
    hyst: float = 0.0             # comparator hysteresis, state units
    coef: object = None           # (a1, a2, k0) override, e.g. from real parts


# ----------------------------------------------------------------------------
# Loop filter coefficients: match the sampled pulse response of the CT loop
# (NRZ pulse on [eld, T + eld]) to the target DT loop filter 1/NTF - 1.
# ----------------------------------------------------------------------------

def target_l(obg, n):
    lo, hi = 1e-4, 0.45
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        _, pz, _ = signal.butter(2, mid, "high", output="zpk", fs=1.0)
        b = np.poly([1.0, 1.0])
        a = np.real(np.poly(pz))
        w, h = signal.freqz(b, a, worN=4096)
        lo, hi = (mid, hi) if np.max(abs(h)) < obg else (lo, mid)
    _, pz, _ = signal.butter(2, lo, "high", output="zpk", fs=1.0)
    b = np.poly([1.0, 1.0])
    a = np.real(np.poly(pz))
    imp = np.zeros(n + 1)
    imp[0] = 1.0
    return signal.lfilter(a - b, b, imp)[1:]          # l[1..n]


def ct_matrices(ap: AdcParams):
    T = 1.0 / ap.fs
    if ap.kind in ("P1", "A1"):
        leak = 1.0 / ap.tau1 if ap.kind == "P1" else 0.0
        A = np.array([[-leak]])
        return A, T
    leak2 = 1.0 / ap.tau2 if ap.kind == "AP2" else 0.0
    A = np.array([[0.0, 0.0], [1.0 / T, -leak2]])
    return A, T


def seg(A, Bv, t):
    """exp(A t) and integral of exp(A s) Bv ds over [0, t]."""
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = Bv
    E = expm(M * t)
    return E[:n, :n], E[:n, n]


def design(ap: AdcParams):
    """Returns per-clock maps and coefficients for the 2nd-order loops."""
    A, T = ct_matrices(ap)
    if ap.kind in ("P1", "A1"):
        return dict(a=np.array([1.0]), k0=0.0)
    if ap.coef is not None:
        return dict(a=np.array(ap.coef[:2]), k0=ap.coef[2])
    tau = ap.eld
    nmatch = 8
    tgt = -target_l(ap.obg, nmatch)
    basis = []
    for bv in (np.array([-1.0 / T, 0.0]), np.array([0.0, -1.0 / T])):
        E1, G1 = seg(A, bv, tau)
        E2, G2 = seg(A, bv, T - tau)
        Ef, _ = seg(A, bv, T)
        # pulse on [tau, T + tau]: first clock: off then on; second: on then off
        x = G2.copy()                      # after clock 0 (on from tau to T)
        resp = [x[1]]
        x = E2 @ (E1 @ x + G1)             # clock 1: on until tau, then off
        resp.append(x[1])
        for _ in range(nmatch - 2):
            x = Ef @ x
            resp.append(x[1])
        basis.append(resp)
    k0_basis = [-1.0] + [0.0] * (nmatch - 1)   # direct path at sample 1
    Mx = np.array(basis + [k0_basis]).T
    coef, *_ = np.linalg.lstsq(Mx, tgt, rcond=None)
    return dict(a=coef[:2], k0=coef[2])


# ----------------------------------------------------------------------------
# Input waveform: area and first moment of u per ADC clock
# ----------------------------------------------------------------------------

def clock_integrals(tau, delta, level0, n_clk):
    """u(t) = level0 + sum delta_j H(t - tau_j), t in clock units.
    Returns area A_n and moment M_n = int u (t - n) dt over clock n."""
    m = np.floor(tau).astype(np.int64)
    ok = (m >= 0) & (m < n_clk)
    m, ph, d = m[ok], tau[ok] - m[ok], delta[ok]
    jump = np.zeros(n_clk + 1)
    np.add.at(jump, m + 1, d)
    lvl = level0 + np.cumsum(jump)[:n_clk]           # level at clock start
    A = lvl.copy()
    M = lvl / 2
    np.add.at(A, m, d * (1 - ph))
    np.add.at(M, m, d * (1 - ph ** 2) / 2)
    return A, M


def switch_node(p: Params, ap: AdcParams, seed=1):
    """Edges of the real and of the replica waveform in ADC clock units."""
    r = run(p, seed=seed)
    e = r["edges"]
    tck = ap.fs / p.f_oser                            # ADC clocks per tick
    n_half = int(np.ceil(max(e["ta"].max(), e["tb"].max()) / p.nh))
    N = int(round(p.nh * tck))
    sgn_a = np.where(e["la"] == 1, 1.0, -1.0)
    sgn_b = np.where(e["lb"] == 1, -1.0, 1.0)
    tau_sw = np.concatenate((e["ta_f"], e["tb_f"])) * tck
    tau_rf = np.concatenate((e["ta"], e["tb"])).astype(float) * tck \
        - ap.replica_skew * ap.fs                     # replica leads by the skew
    dlt = np.concatenate((sgn_a, sgn_b))
    return tau_sw, tau_rf, dlt, n_half * N, N


def input_integrals(sn, ap, shift=0.0):
    tau_sw, tau_rf, dlt, n_clk, N = sn
    A_sw, M_sw = clock_integrals(tau_sw + shift, dlt, 0.0, n_clk)
    if ap.mode == "direct":
        return 0.8 * A_sw, 0.8 * M_sw, 0.8
    A_rf, M_rf = clock_integrals(tau_rf + shift, dlt, 0.0, n_clk)
    g = ap.gain
    return g * (A_sw - A_rf), g * (M_sw - M_rf), g


# ----------------------------------------------------------------------------
# Modulator simulation
# ----------------------------------------------------------------------------

def modulate(Au, Mu, ap: AdcParams, seed=2):
    rng = np.random.default_rng(seed)
    A, T = ct_matrices(ap)
    dz = design(ap)
    tau = ap.eld
    n = len(Au)
    d = np.empty(n)
    jit = rng.normal(0, ap.jitter, n) / T if ap.jitter > 0 else np.zeros(n)
    cn = rng.normal(0, ap.comp_noise, n)
    if ap.kind in ("P1", "A1"):
        lk = -A[0, 0]
        E1, G1 = seg(A, np.array([-1.0 / T]), tau)
        E2, G2 = seg(A, np.array([-1.0 / T]), T - tau)
        e1, g1, e2, g2 = E1[0, 0], G1[0], E2[0, 0], G2[0]
        x, dp = 0.0, 1.0
        for k in range(n):
            # decision at the start of clock k from the state at its start
            dk = 1.0 if x + cn[k] >= 0 else -1.0
            x = e2 * (e1 * x + g1 * dp) + g2 * dk + Au[k] \
                - (dp - dk) * jit[k]
            d[k] = dk
            dp = dk
        return d
    a1, a2 = dz["a"]
    k0 = dz["k0"]
    c1 = a1                                       # unity signal gain
    E1, G1 = seg(A, np.array([-a1 / T, -a2 / T]), tau)
    E2, G2 = seg(A, np.array([-a1 / T, -a2 / T]), T - tau)
    P = E2 @ E1
    Q1 = E2 @ G1
    e00, e01, e10, e11 = P[0, 0], P[0, 1], P[1, 0], P[1, 1]
    q0, q1 = Q1
    g0, g1 = G2
    x0 = x1 = 0.0
    dp = 1.0
    for k in range(n):
        y = x1 - k0 * dp + cn[k]
        dk = 1.0 if y >= -0.5 * ap.hyst * dp else -1.0
        n0 = e00 * x0 + e01 * x1 + q0 * dp + g0 * dk
        n1 = e10 * x0 + e11 * x1 + q1 * dp + g1 * dk
        # input (area / first moment), leak over one clock neglected
        n0 += c1 * Au[k]
        n1 += c1 * (Au[k] - Mu[k])
        # jitter: old DAC value held j longer
        jj = (dp - dk) * jit[k]
        n0 -= a1 * jj
        n1 -= a2 * jj
        x0, x1 = n0, n1
        d[k] = dk
        dp = dk
    return d


# ----------------------------------------------------------------------------
# Decimation and floor
# ----------------------------------------------------------------------------

def decimate(seq, N, kernel):
    """Per-half value of a per-clock sequence (mean level units)."""
    K = len(seq) // N
    b = seq[: K * N].reshape(K, N).mean(axis=1)
    if kernel == "sinc1":
        return b
    if kernel == "sinc2":
        w = np.convolve(np.ones(N), np.ones(N)) / N ** 2   # length 2N-1
        y = np.convolve(seq, w)[2 * N - 2::N]               # ends at half k
        return y[:K]
    raise ValueError(kernel)


def floor_db(err, f_s=1.536e6, band=(20.0, 20e3), nfft=2 ** 15):
    err = err[-nfft:]
    w = signal.windows.blackmanharris(nfft)
    E = np.fft.rfft(err * w)
    psd = np.abs(E) ** 2 / np.sum(w ** 2)                  # power per bin * nfft
    f = np.fft.rfftfreq(nfft, 1 / f_s)
    m = (f >= band[0]) & (f <= band[1])
    p_band = 2 * np.sum(psd[m]) / nfft
    return 10 * np.log10(p_band / 0.5)                     # vs full-scale sine


def _floor(d, Aref, N, kern, g):
    meas = decimate(d, N, kern)[8:-8]
    exact = decimate(Aref, N, kern)[8:-8]
    alpha = np.dot(meas, exact) / np.dot(exact, exact)   # remove pure gain error
    return floor_db((meas - alpha * exact) / g), alpha


def error_sequence(p: Params, ap: AdcParams, kern):
    """Measurement error per half period (amplifier full-scale units), with the
    ADC delay and gain fitted out, aligned so that index k is the value the
    loop receives at the end of half k. For injection into closedloop.py."""
    sn = switch_node(p, ap)
    N = sn[4]
    Au, Mu, g = input_integrals(sn, ap)
    d = modulate(Au, Mu, ap)
    best = (1e9, 0.0)
    grid = np.arange(0.0, 4.01, 0.25)
    for it in range(3):
        for sh in grid:
            f, _ = _floor(d, input_integrals(sn, ap, sh)[0], N, kern, g)
            if f < best[0]:
                best = (f, sh)
        step = grid[1] - grid[0]
        grid = best[1] + np.linspace(-step, step, 9)
    A_ref = input_integrals(sn, ap, best[1])[0]
    meas = decimate(d, N, kern)
    exact = decimate(A_ref, N, kern)
    alpha = np.dot(meas[8:-8], exact[8:-8]) / np.dot(exact[8:-8], exact[8:-8])
    err = (meas - alpha * exact) / g
    err[:8] = 0.0
    err[-8:] = 0.0
    if kern == "sinc2":                    # window ends at the end of half k+1
        err = np.concatenate(([0.0], err[:-1]))
    return err, best[0], best[1]


def measure(p: Params, ap: AdcParams, kernels=("sinc1", "sinc2"), cache={}):
    """In-band floor per kernel. The ADC's signal transfer is a delay of a
    few clocks; it is part of the loop latency, not noise, so the reference
    is the exact sequence delayed by the best-fit (fractional) delay."""
    key = (repr(p), ap.fs, ap.replica_skew)
    if key not in cache:
        cache[key] = switch_node(p, ap)
    sn = cache[key]
    N = sn[4]
    Au, Mu, g = input_integrals(sn, ap)
    d = modulate(Au, Mu, ap)
    out = {}
    for kern in kernels:
        best = (1e9, 0.0, 1.0)
        grid = np.arange(0.0, 4.01, 0.25)
        for it in range(3):
            for sh in grid:
                A_ref, _, _ = input_integrals(sn, ap, sh)
                f, al = _floor(d, A_ref, N, kern, g)
                if f < best[0]:
                    best = (f, sh, al)
            step = grid[1] - grid[0]
            grid = best[1] + np.linspace(-step, step, 9)
        out[kern], out[kern + "_delay"], out[kern + "_gain"] = best
    return out


SCEN = [
    ("P1  пассивный RC 1-го пор., 98 МГц (первая схема)",
     dict(kind="P1", fs=98.304e6, mode="direct", comp_noise=0.03)),
    ("A1  активный интегратор 1-го пор., 98 МГц", dict(kind="A1", fs=98.304e6, mode="direct")),
    ("A1  активный интегратор 1-го пор., 49 МГц", dict(kind="A1", mode="direct")),
    ("A2  2-й порядок, 49 МГц", dict(kind="A2", mode="direct")),
    ("AP2 2-й порядок, 2-й интегратор пассивный, 49 МГц", dict(kind="AP2", mode="direct")),
    ("A2  + вычитание реплики ×4", dict(kind="A2", gain=4.0)),
    ("A2  + вычитание реплики ×8", dict(kind="A2", gain=8.0)),
    ("A2  + вычитание реплики ×16", dict(kind="A2", gain=16.0)),
    ("A2  джиттер тактов АЦП 1 пс", dict(kind="A2", mode="direct", jitter=1e-12)),
    ("A2  джиттер тактов АЦП 5 пс", dict(kind="A2", mode="direct", jitter=5e-12)),
    ("A2  джиттер тактов АЦП 20 пс", dict(kind="A2", mode="direct", jitter=20e-12)),
    ("A2  + реплика ×16, джиттер 5 пс", dict(kind="A2", gain=16.0, jitter=5e-12)),
    ("AP2 + реплика ×8, джиттер 5 пс (рекомендуемый)", dict(kind="AP2", gain=8.0, jitter=5e-12)),
    ("AP2 + реплика ×8, джиттер 5 пс, ELD 8 нс", dict(kind="AP2", gain=8.0, jitter=5e-12, eld=8e-9)),
]


def main():
    p = Params(deadtime=2e-9, r_load=4.0, n_fft=4096 + 256, pad=512)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for lab, kw in SCEN:
        ap = AdcParams(**kw)
        r = measure(p, ap)
        rows.append((lab, r["sinc1"], r["sinc2"], r["sinc2_delay"]))
        print(f"{lab:52s} sinc1 {r['sinc1']:7.1f} dB   sinc2 {r['sinc2']:7.1f} dB"
              f"  (delay {r['sinc2_delay']:.2f} clk)", flush=True)
    with open(os.path.join(OUT, "fb_adc.md"), "w") as fh:
        fh.write("# АЦП обратной связи (pre-петля): шум измерения в полосе\n\n"
                 "Вход — реальное напряжение моста (BD-ШИМ, dead time 2 нс, 4 Ом, 1 кГц −1 дБFS). "
                 "Шум = (измеренная − точная последовательность) за полупериод, 20 Гц–20 кГц, "
                 "относительно полной шкалы усилителя. В замкнутой петле он попадает на выход 1:1. "
                 "ELD 4 нс, шум компаратора учтён. sinc1 — сумма битов за полупериод (без "
                 "доп. задержки), sinc2 — треугольник на два полупериода (+0,5 полупериода).\n\n")
        fh.write("| АЦП | sinc1, дБ | sinc2, дБ |\n|---|---:|---:|\n")
        for lab, s1, s2, _ in rows:
            fh.write(f"| {lab} | {s1:.1f} | {s2:.1f} |\n")


if __name__ == "__main__":
    main()
