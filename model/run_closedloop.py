"""
Closed-loop scenario sweep -> results/closedloop.md, loop_gain.png

    python model/run_closedloop.py
"""

import os
import sys
import numpy as np
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from classd_model import Params, db  # noqa: E402
from closedloop import (LoopParams, simulate, plant_discrete, plant_freq,  # noqa: E402
                        controller_tf, ctrl_freq, closed_loop_stable)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

# controllers from loop_design.py (modulus margin >= 0.5 on 4/8 Ohm/open)
PRE = {1: dict(n_int=2, fc=74.7e3, fz_ratio=0.3),
       0: dict(n_int=3, fc=144.4e3, fz_ratio=0.2)}
# global loop: "fast" = the most loop gain the LC resonance allows,
# "slow" = 4 kHz integrator (modulus margin 0.63), used in "dual"
POST = {"fast": {1: dict(n_int=2, fc=14.1e3, fz_ratio=0.1, f_lc=50e3, zeta_z=0.3, f_p=300e3),
                 0: dict(n_int=1, fc=25.2e3, fz_ratio=0.1, f_lc=50e3, zeta_z=0.3, f_p=300e3)},
        "slow": {1: dict(n_int=1, fc=4e3, fz_ratio=0.1),
                 0: dict(n_int=1, fc=4e3, fz_ratio=0.1)}}


def lp(mode, D=1, post="slow", **kw):
    return LoopParams(mode=mode, delay=D, pre=PRE[D], post=POST[post][D], **kw)


def adc_sigma(sqnr_db):
    """rms per 1.536 MHz sample of white noise whose 20 kHz in-band part
    gives the stated full-scale SQNR."""
    return 0.707 * np.sqrt(768e3 / 20e3) * 10 ** (-sqnr_db / 20)


BASE = dict(n_fft=2 ** 14, pad=2048)
SCEN = [
    ("Dead time 2 нс, 4 Ом", dict(deadtime=2e-9, r_load=4.0), [
        ("без ООС", lp("none")),
        ("pre, D=1", lp("pre")),
        ("post (макс. усиление), D=1", lp("post", post="fast")),
        ("dual, D=1", lp("dual")),
        ("dual, D=0", lp("dual", 0)),
        ("dual, D=1, АЦП 104 дБ", lp("dual", adc_noise=adc_sigma(104))),
        ("dual, D=1, АЦП 115 дБ", lp("dual", adc_noise=adc_sigma(115))),
    ]),
    ("Dead time 2 нс, 4 Ом, IMD 18+19 кГц", dict(deadtime=2e-9, r_load=4.0,
                                               f_sig=18000, f_sig2=19000), [
        ("без ООС", lp("none")),
        ("pre, D=1", lp("pre")),
        ("post (макс. усиление), D=1", lp("post", post="fast")),
        ("dual, D=1", lp("dual"))]),
    ("Dead time 2 нс, 8 Ом", dict(deadtime=2e-9), [
        ("без ООС", lp("none")), ("dual, D=1", lp("dual"))]),
    ("Dead time 5 нс, 4 Ом", dict(deadtime=5e-9, r_load=4.0), [
        ("без ООС", lp("none")), ("dual, D=1", lp("dual"))]),
    ("Dead time 2 нс, без нагрузки", dict(deadtime=2e-9, r_load=np.inf), [
        ("без ООС", lp("none")), ("dual, D=1", lp("dual"))]),
    ("Пульсации шины 1 % / 100 Гц", dict(bus_ripple=0.01), [
        ("без ООС", lp("none")), ("pre, D=1", lp("pre"))]),
    ("Джиттер 50 пс", dict(jitter_rms=50e-12), [
        ("без ООС", lp("none")), ("dual, D=1", lp("dual"))]),
]


def plot_loop_gains(p):
    f = np.logspace(2, np.log10(0.499 * p.f_upd), 1500)
    w = 2 * np.pi * f / p.f_upd
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, D in zip(axes, [1, 0]):
        b, a = controller_tf(PRE[D], p.f_upd)
        L = ctrl_freq(b, a, w) * np.exp(-1j * w * (1 + D))
        ax.semilogx(f, 20 * np.log10(abs(L)), color=S1, lw=1.2, label="pre")
        for R, c, ls in [(4.0, S2, "-"), (8.0, S2, "--"), (np.inf, S2, ":")]:
            ss = plant_discrete(replace(p, r_load=R), 2)
            P = plant_freq(*ss, w)
            ssn = plant_discrete(replace(p, r_load=8.0), 2)
            dc = abs(ssn[2] @ np.linalg.solve(np.eye(len(ssn[0])) - ssn[0], ssn[1]) + ssn[3])
            b2, a2 = controller_tf(POST["fast"][D], p.f_upd, gain_dc=dc)
            L2 = ctrl_freq(b2, a2, w) * P * np.exp(-1j * w * (1 + D))
            lab = {4.0: "post 4 Ом", 8.0: "post 8 Ом", np.inf: "post без нагр."}[R] + " (макс.)"
            ax.semilogx(f, 20 * np.log10(abs(L2)), color=c, lw=1.0, ls=ls, label=lab)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(20e3, color=GRID, lw=1.0)
        ax.set_xlim(100, 0.5 * p.f_upd)
        ax.set_ylim(-40, 120)
        ax.set_title(f"Петлевое усиление, задержка D = {D}", loc="left", color=INK)
        ax.set_xlabel("Частота, Гц")
    axes[0].set_ylabel("|L|, дБ")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loop_gain.png"), dpi=150)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False})
    p0 = Params()
    plot_loop_gains(p0)
    rows = []
    for title, pk, runs in SCEN:
        p = Params(**BASE, **pk)
        for lab, l in runs:
            r = simulate(p, l)
            rows.append((title, lab, db(r["thdn"]), db(r["thd"])))
            print(f"{title:38s} {lab:24s} THD+N {db(r['thdn']):7.1f}  THD/IMD {db(r['thd']):7.1f}")
    with open(os.path.join(OUT, "closedloop.md"), "w") as fh:
        fh.write("# Closed-loop: результаты модели\n\n")
        fh.write("1 кГц −1 дБFS, если не указано иное; полоса 20 Гц–20 кГц. "
                 "pre — локальная петля по площади импульсов моста, post — глобальная "
                 "после LC (2-DOF, эталон — модель 8 Ом, треугольное ядро измерения), "
                 "dual — обе. D — задержка коррекции в полупериодах сверх одного "
                 "(D=1: коррекция по измерению полупериода k приходит в k+2). "
                 "«АЦП 104 дБ» — шум АЦП ООС, эквивалентный SQNR 104 дБ в полосе 20 кГц.\n\n")
        fh.write("| Условия | ООС | THD+N, дБ | THD (IMD), дБ |\n|---|---|---:|---:|\n")
        for t, lab, tn, th in rows:
            fh.write(f"| {t} | {lab} | {tn:.1f} | {th:.1f} |\n")


if __name__ == "__main__":
    main()
