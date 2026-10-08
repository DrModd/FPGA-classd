"""
Coefficient tables for the polyphase interpolator of jitter_fifo.

Read position = rd_int + frac, frac = j/M + mu/M (M = 2^MPH_SH stored
phases, mu = the remaining FW - MPH_SH bits). Output:
    y = sum_n x[rd_int - NT/2 + 1 + n] * c_n,
    c_n = C[j][n] + round((C[j+1][n] - C[j][n]) * mu / 2^MU)
C in Q.28, CW = 30 bits signed (the ROM is two 18-bit BSRAM blocks wide anyway). The ROM holds M + 1 phases (j = 0..M) of NT
coefficients, address j*NT + n; the RTL reads C[j] and C[j+1] through the
two ports of a dual-port block RAM.

Designs:
  ls  - weighted least-squares fractional delay, NT taps (default 64),
        fitted on 0..20 kHz at 44.1 kHz (32 taps: 0..16 kHz) with more weight at low/mid
        frequencies (the same table is used at every sample rate; at
        higher rates the audio band is a smaller part of it)
  cr  - Catmull-Rom (4 taps), the original cubic interpolator

    python jitter_fifo/model/gen_coefs.py   -> rtl/fir_ls64_m128.hex (default),
                                               rtl/fir_ls64_m64.hex, rtl/fir_ls32_m128.hex,
                                               rtl/fir_cr4.hex,
                                               coefs.md
"""

import os
import numpy as np

HERE = os.path.dirname(__file__)
RTL = os.path.join(HERE, "..", "rtl")
FW, CW, CSH = 16, 30, 28


def wls_fd(N, d, fs=44100.0, fmax=20500.0, wfun=None, ngrid=800):
    w = np.linspace(0, 2 * np.pi * fmax / fs, ngrid)
    f = w * fs / (2 * np.pi)
    W = np.ones_like(w) if wfun is None else np.sqrt(wfun(f))
    n = np.arange(N)
    c = N / 2 - 1 + d
    A = np.exp(-1j * np.outer(w, n)) * W[:, None]
    b = np.exp(-1j * w * c) * W
    h, *_ = np.linalg.lstsq(np.vstack([A.real, A.imag]),
                            np.concatenate([b.real, b.imag]), rcond=None)
    return h


def weight(f):
    # flat to ~8 kHz, then falling: more accuracy where music has its energy
    return 1.0 / (1e-3 + (f / 16000.0) ** 4)


def catmull_rom(d):
    t = d
    return np.array([(-t ** 3 + 2 * t ** 2 - t) / 2,
                     (3 * t ** 3 - 5 * t ** 2 + 2) / 2,
                     (-3 * t ** 3 + 4 * t ** 2 + t) / 2,
                     (t ** 3 - t ** 2) / 2])


def make_table(design, NT, MPH_SH, fmax=20500.0):
    M = 1 << MPH_SH
    H = []
    for j in range(M + 1):
        d = j / M
        if design == "cr":
            h = catmull_rom(d)
        else:
            h = wls_fd(NT, d, fmax=fmax, wfun=weight)
        H.append(h)
    H = np.array(H)
    C = np.round(H * (1 << CSH)).astype(np.int64)
    # exact DC gain in every phase: the rounding residue goes to the largest tap
    for j in range(M + 1):
        C[j, np.argmax(np.abs(C[j]))] += (1 << CSH) - C[j].sum()
    C[0] = 0
    C[0][NT // 2 - 1] = 1 << CSH                      # phase 0 is an exact delta
    C[M] = 0
    C[M][NT // 2] = 1 << CSH                          # phase M: delta one tap later
    assert np.abs(C).max() < (1 << (CW - 1)), "coefficient overflow"
    return C


def write_hex(C, path):
    NT = C.shape[1]
    with open(path, "w") as fh:
        for j in range(C.shape[0]):
            for n in range(NT):
                fh.write(f"{int(C[j, n]) & ((1 << CW) - 1):0{(CW + 3) // 4}x}\n")


def coefs_fixed(C, frac, MPH_SH):
    """Exactly what the RTL computes for one read position."""
    MU = FW - MPH_SH
    j = frac >> MU
    mu = frac & ((1 << MU) - 1)
    if MU == 0:
        return C[j].copy()
    return C[j] + (((C[j + 1] - C[j]) * mu + (1 << (MU - 1))) >> MU)


def error_profile(C, MPH_SH, fs, freqs, n_frac=64):
    NT = C.shape[1]
    out = []
    for f in freqs:
        w = 2 * np.pi * f / fs
        worst = 0.0
        for frac in np.linspace(1, (1 << FW) - 1, n_frac).astype(int):
            c = coefs_fixed(C, int(frac), MPH_SH) / (1 << CSH)
            H = np.exp(-1j * w * np.arange(NT)) @ c
            d = frac / (1 << FW)
            worst = max(worst, abs(H - np.exp(-1j * w * (NT / 2 - 1 + d))))
        out.append(20 * np.log10(worst))
    return out


def main():
    freqs = (1000, 5000, 10000, 15000, 20000)
    rep = ["# Таблицы коэффициентов интерполятора\n\n",
           "Худшая ошибка интерполяции по всем дробным позициям, дБ относительно "
           "сигнала (с квантованием коэффициентов и линейной интерполяцией между "
           "фазами, как в RTL).\n\n",
           "Таблица ROM: (M + 1) · NT слов по 30 бит, читается двумя портами (C[j] и C[j+1]), "
           "то есть BSRAM в режиме true dual port 1024×18, два блока по ширине. Блоков: "
           "fir_ls64_m128 — 18, fir_ls64_m64 — 10, fir_ls32_m128 — 10, fir_cr4 — 10 "
           "(точное число смотрите в отчёте синтеза).\n\n",
           "| Таблица | Fs | 1 кГц | 5 кГц | 10 кГц | 15 кГц | 20 кГц |\n"
           "|---|---|---:|---:|---:|---:|---:|\n"]
    # the 32-tap table is meant for high sample rates (768 kHz with a 49.152 MHz
    # read clock), where the audio band is a small part of Nyquist: a narrower
    # design band (16 kHz at 44.1 kHz) buys accuracy there
    for design, NT, MPH, fmax, name in (("ls", 64, 7, 20000.0, "fir_ls64_m128.hex"),
                                        ("ls", 64, 6, 20000.0, "fir_ls64_m64.hex"),
                                        ("ls", 32, 7, 16000.0, "fir_ls32_m128.hex"),
                                        ("cr", 4, 10, 0.0, "fir_cr4.hex")):
        C = make_table(design, NT, MPH, fmax)
        write_hex(C, os.path.join(RTL, name))
        for fs in (44100, 48000, 96000, 192000, 768000):
            e = error_profile(C, MPH, fs, freqs)
            rep.append(f"| {name} (NT={NT}, M={1 << MPH}) | {fs/1000:g} кГц | "
                       + " | ".join(f"{v:.1f}" for v in e) + " |\n")
            print(name, fs, [round(v, 1) for v in e], flush=True)
    with open(os.path.join(HERE, "..", "coefs.md"), "w") as fh:
        fh.writelines(rep)


if __name__ == "__main__":
    main()
