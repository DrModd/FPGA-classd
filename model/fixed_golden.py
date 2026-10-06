"""
Bit-exact fixed-point reference of the RTL chain (one channel):

  24-bit PCM @192 kHz
   -> hb_stage x3 (halfband x2 interpolators)      -> 1.536 MHz, Q.24
   -> pwm_corr  (clip, m = x - D2(x^3)/24)          -> 1.536 MHz, Q.24
   -> ns_shaper (5th-order error feedback)          -> q in units of 2 ticks
   -> bd_pwm    (nA = 256 + q, nB = 256 - q per half period)

Every arithmetic step uses Python integers with the same widths, rounding
(add half, arithmetic shift right) and saturation as the Verilog, so the
streams written to tb/vectors/ must match the RTL exactly.

    python model/fixed_golden.py           # coefficients -> rtl/, vectors -> tb/,
                                           # and a THD+N check of the fixed chain
"""

import os
import sys
import numpy as np
from dataclasses import replace
from scipy import signal

sys.path.insert(0, os.path.dirname(__file__))
from classd_model import (Params, design_ntf, leg_edges, sample_steps,  # noqa: E402
                          decimate_to_fs, analyse, db)

ROOT = os.path.join(os.path.dirname(__file__), "..")

# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------
W = 28            # sample width, value = int / 2^24 (range +-8)
F = 24            # fractional bits of samples
CW = 25           # halfband coefficient width (Q.24, |g| < 1)
SW = 44           # shaper state width, F fractional bits
SCF = 28          # shaper coefficient fractional bits
SCW = 34          # shaper coefficient width
NH = 512          # ticks per half period
N_MIN = 12        # 15 ns at 1.27 ns per tick
HEADROOM = 6
LIM = (NH - N_MIN) // 2                  # shaper limit, units of 2 ticks = 250
XMAX = ((NH - N_MIN - HEADROOM) << F) // NH   # input clip, Q.24
INV24 = round((1 << F) / 24)
E_MAX = 2 << F                           # shaper error clamp (2 LSB)

HB_TAPS = (31, 27, 27)                   # per stage, 4m-1 (images <= -144 dB)
HB_BETA = 15.6


def rnd_shift(x, s):
    """round half up, arithmetic shift (Verilog: (x + (1<<(s-1))) >>> s)"""
    return (x + (1 << (s - 1))) >> s


def sat(x, w):
    lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------
# Halfband design: Kaiser-windowed sinc at fs/4 (exact halfband zeros)
# ---------------------------------------------------------------------------

def halfband_coefs(nt):
    n = np.arange(nt) - (nt - 1) / 2
    h = 0.5 * np.sinc(n / 2) * np.kaiser(nt, HB_BETA)
    h[0::2] /= 2 * np.sum(h[0::2])      # polyphase branch sums to 1/2 (centre stays 1/2)
    m = (nt + 1) // 4
    # even-index taps h[0], h[2], ... ; the polyphase output is 2*sum
    g = 2 * h[0::2][:m]                  # symmetric: first m of 2m taps
    return [int(round(v * (1 << F))) for v in g], h


def hb_stage_fixed(x, gq):
    """Interpolate x2. Output order per input k: y_even (MAC), y_odd (delay)."""
    m = len(gq)
    L = 2 * m
    d = [0] * L
    out = []
    for xv in x:
        d = [xv] + d[:-1]
        acc = 0
        for i in range(m):
            acc += gq[i] * (d[i] + d[L - 1 - i])
        out.append(sat(rnd_shift(acc, F), W))
        out.append(d[m - 1])
    return out


def interp8_fixed(x24):
    x = [v << 1 for v in x24]            # s.23 -> Q.24
    for nt in HB_TAPS:
        gq, _ = halfband_coefs(nt)
        x = hb_stage_fixed(x, gq)
    return x


# ---------------------------------------------------------------------------
# PWM correction
# ---------------------------------------------------------------------------

def cube_fixed(x):
    c1 = rnd_shift(x * x, F)
    return rnd_shift(c1 * x, F)


def pwm_corr_fixed(x):
    """m[k] = clip(x[k]) - D2(clip(x)^3)/24, produced when x[k+1] arrives."""
    out = []
    xc = [0, 0]              # x[k-1], x[k] (clipped)
    cc = [0, 0]              # cubes
    for i, xv in enumerate(x):
        xn = max(-XMAX, min(XMAX, xv))
        cn = cube_fixed(xn)
        if i >= 1:
            d2 = cc[0] - 2 * cc[1] + cn
            out.append(xc[1] - rnd_shift(d2 * INV24, F))
        xc = [xc[1], xn]
        cc = [cc[1], cn]
    return out


# ---------------------------------------------------------------------------
# Noise shaper
# ---------------------------------------------------------------------------

def shaper_coefs():
    p = replace(Params(), f_oser=Params().f_oser / 2)
    b, a = design_ntf(p)
    hb = b - a
    hbq = [int(round(v * (1 << SCF))) for v in hb[1:]]
    aq = [int(round(v * (1 << SCF))) for v in a[1:]]
    return hbq, aq


def shaper_fixed(m):
    """m: Q.24 modulation (1.0 = full scale). Internal: v = m * 256 in units of
    2 ticks with F fractional bits -> q = round(v), clamped error feedback."""
    hbq, aq = shaper_coefs()
    n = len(aq)
    z = [0] * (n + 1)
    out = []
    for mv in m:
        w = z[0]
        v = (mv << 8) + w                # *256, still F fractional bits
        q = rnd_shift(v, F)
        q = max(-LIM, min(LIM, q))
        e = (q << F) - v
        e = max(-E_MAX, min(E_MAX, e))
        for i in range(n):
            z[i] = sat(rnd_shift(hbq[i] * e, SCF) - rnd_shift(aq[i] * w, SCF) + z[i + 1], SW)
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Verilog include files with the coefficients
# ---------------------------------------------------------------------------

def write_rtl_coefs():
    lines = ["// generated by model/fixed_golden.py - do not edit",
             "// halfband stage coefficients g[i] = 2*h[2i], Q.24, i < m"]
    lines.append("function signed [24:0] hb_coef;\n  input integer stage;\n  input integer i;\n  begin\n    hb_coef = 25'sd0;\n    case (stage)")
    for s, nt in enumerate(HB_TAPS, start=1):
        gq, _ = halfband_coefs(nt)
        lines.append(f"      {s}: case (i)")
        for i, v in enumerate(gq):
            sgn = "-" if v < 0 else ""
            lines.append(f"        {i}: hb_coef = {sgn}25'sd{abs(v)};")
        lines.append("        default: hb_coef = 25'sd0;\n      endcase")
    lines.append("      default: hb_coef = 25'sd0;\n    endcase\n  end\nendfunction")
    with open(os.path.join(ROOT, "rtl", "hb_coefs.vh"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    hbq, aq = shaper_coefs()
    lines = ["// generated by model/fixed_golden.py - do not edit",
             f"// noise shaper H = NTF - 1 (transposed DF-II), Q.{SCF}",
             f"function signed [{SCW-1}:0] ns_hb;\n  input integer i;\n  begin\n    case (i)"]
    for i, v in enumerate(hbq):
        sgn = "-" if v < 0 else ""
        lines.append(f"      {i}: ns_hb = {sgn}{SCW}'sd{abs(v)};")
    lines.append(f"      default: ns_hb = {SCW}'sd0;\n    endcase\n  end\nendfunction")
    lines.append(f"function signed [{SCW-1}:0] ns_a;\n  input integer i;\n  begin\n    case (i)")
    for i, v in enumerate(aq):
        sgn = "-" if v < 0 else ""
        lines.append(f"      {i}: ns_a = {sgn}{SCW}'sd{abs(v)};")
    lines.append(f"      default: ns_a = {SCW}'sd0;\n    endcase\n  end\nendfunction")
    with open(os.path.join(ROOT, "rtl", "ns_coefs.vh"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Vectors and checks
# ---------------------------------------------------------------------------

def hexw(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"


def make_input(n, f=1000.0, level_dbfs=-1.0, fs=192000.0, seed=0):
    t = np.arange(n) / fs
    x = 10 ** (level_dbfs / 20) * np.sin(2 * np.pi * f * t)
    rng = np.random.default_rng(seed)
    x += rng.uniform(-1, 1, n) * 2 ** -23       # 1 LSB TPDF-ish dither
    return [int(v) for v in np.clip(np.round(x * (1 << 23)), -(1 << 23), (1 << 23) - 1)]


def write_vectors(n_in=512):
    vdir = os.path.join(ROOT, "tb", "vectors")
    os.makedirs(vdir, exist_ok=True)
    # a sine plus a short full-scale overload burst exercises clip and limits
    x = make_input(n_in, f=7000.0, level_dbfs=-0.3)
    for i in range(300, 340):
        x[i] = (1 << 23) - 1 if (i // 4) % 2 else -(1 << 23)
    xi = interp8_fixed(x)
    m = pwm_corr_fixed(xi)
    q = shaper_fixed(m)
    for name, seq, w in [("in", x, 24), ("interp", xi, W), ("corr", m, W), ("q", q, 9)]:
        with open(os.path.join(vdir, f"{name}.hex"), "w") as fh:
            fh.write("\n".join(hexw(v, w) for v in seq) + "\n")
    with open(os.path.join(vdir, "counts.vh"), "w") as fh:
        fh.write(f"localparam N_IN = {len(x)};\nlocalparam N_INTERP = {len(xi)};\n"
                 f"localparam N_CORR = {len(m)};\nlocalparam N_Q = {len(q)};\n")
    return x, xi, m, q


def performance(f_sig=1000.0, level=-1.0):
    """THD+N of the fixed chain with ideal edges (same analysis as the float model)."""
    p = Params()
    n_in = p.n_fft + 2 * p.pad
    kf = int(round(f_sig * p.n_fft / p.fs_in))
    t = np.arange(n_in)
    x = 10 ** (level / 20) * np.sin(2 * np.pi * kf * t / p.n_fft)
    rng = np.random.default_rng(0)
    x24 = [int(v) for v in np.clip(np.round(x * (1 << 23) + rng.uniform(-1, 1, n_in)),
                                   -(1 << 23), (1 << 23) - 1)]
    q = np.array(shaper_fixed(pwm_corr_fixed(interp8_fixed(x24))), dtype=np.int64)
    M = 2 * q
    na = (NH + M) // 2
    nb = na - M
    ta, la = leg_edges(na, p)
    tb, lb = leg_edges(nb, p)
    T = p.f_oser / (p.fs_in * p.grid_ratio)
    tau = np.concatenate((ta.astype(float), tb.astype(float)))
    delta = np.concatenate((np.where(la == 1, 1.0, -1.0), np.where(lb == 1, -1.0, 1.0)))
    y = sample_steps(tau, delta, T, len(q) // p.up * p.grid_ratio)
    yo = decimate_to_fs(y, p, p.grid_ratio)
    start = len(yo) - p.n_fft - p.pad // 2
    r = analyse(yo[start:start + p.n_fft], p, kf)
    return r


def halfband_response():
    """Composite response of the three quantised stages (DC gain 1): passband
    ripple and the worst attenuation of the images of a <= 24 kHz input."""
    h_tot = np.array([1.0])
    for s, nt in enumerate(HB_TAPS):
        gq, h = halfband_coefs(nt)
        m = len(gq)
        g = np.array(gq + gq[::-1], float) / (1 << F)
        hq = np.zeros(nt)
        hq[0::2] = g / 2
        hq[(nt - 1) // 2] = 0.5
        up = 2 ** (len(HB_TAPS) - 1 - s)   # stage 1 runs at 384 kHz
        hu = np.zeros(len(hq) * up - (up - 1))
        hu[::up] = hq
        h_tot = np.convolve(h_tot, hu)
    w, H = signal.freqz(h_tot, worN=1 << 17, fs=1.536e6)
    pb = np.abs(H[w <= 20e3])
    # images of a <=24 kHz input sit at k*192 kHz +- 24 kHz
    img = np.zeros(len(w), bool)
    for k in range(1, 5):
        img |= np.abs(w - k * 192e3) <= 24e3
    sb = np.abs(H[img])
    return 20 * np.log10(pb.max()), 20 * np.log10(pb.min()), 20 * np.log10(sb.max())


def main():
    os.makedirs(os.path.join(ROOT, "rtl"), exist_ok=True)
    write_rtl_coefs()
    x, xi, m, q = write_vectors()
    print(f"vectors: {len(x)} in, {len(xi)} interp, {len(m)} corr, {len(q)} q; "
          f"q range {min(q)}..{max(q)} (limit +-{LIM})")
    pmax, pmin, sb = halfband_response()
    print(f"interpolator: passband {pmin:+.4f}..{pmax:+.4f} dB, stopband {sb:.1f} dB")
    for f in (1000.0, 6600.0):
        r = performance(f)
        print(f"fixed chain {f/1e3:.1f} kHz: THD+N {db(r['thdn']):.1f} dB, THD {db(r['thd']):.1f} dB")


if __name__ == "__main__":
    main()
