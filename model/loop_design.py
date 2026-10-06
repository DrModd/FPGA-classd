"""
Grid search of the loop controllers on the exact discrete plant.

Constraints, for every load (4 Ohm, 8 Ohm, open):
  closed loop stable (eigenvalues), modulus margin min|1+L| >= MM_MIN
Objective: the largest worst-case loop gain at 20 kHz.

    python model/loop_design.py      -> prints the best designs, writes
                                        results/loop_design.md and loop_gain.png
"""

import os
import sys
import itertools
import numpy as np
from dataclasses import replace

sys.path.insert(0, os.path.dirname(__file__))
from classd_model import Params  # noqa: E402
from closedloop import (plant_discrete, plant_freq, controller_tf,  # noqa: E402
                        closed_loop_stable, ctrl_freq)

MM_MIN = 0.5            # -> GM >= 6 dB, PM >= 29 deg
LOADS = [4.0, 8.0, np.inf]
DELAY = 1
POST_ORDER = 2


def freq_grid(p):
    f = np.logspace(1, np.log10(0.499 * p.f_upd), 2500)
    return f, 2 * np.pi * f / p.f_upd


def evaluate(cfg, plants, p, f, w, delay=None):
    delay = DELAY if delay is None else delay
    b, a = controller_tf(cfg, p.f_upd)
    C = ctrl_freq(b, a, w)
    zd = np.exp(-1j * w * (1 + delay))
    worst20, worst1k, mm_min = np.inf, np.inf, np.inf
    i20 = np.argmin(abs(f - 20e3))
    i1k = np.argmin(abs(f - 1e3))
    for (Phi, Gam, Cz, Dz, P) in plants:
        L = C * P * zd
        mm = np.min(np.abs(1 + L))
        mm_min = min(mm_min, mm)
        if mm < MM_MIN:
            return None
        worst20 = min(worst20, abs(L[i20]))
        worst1k = min(worst1k, abs(L[i1k]))
    for (Phi, Gam, Cz, Dz, P) in plants:
        if closed_loop_stable(Phi, Gam, Cz, Dz, b, a, delay) >= 1.0:
            return None
    return dict(cfg=cfg, g20=20 * np.log10(worst20), g1k=20 * np.log10(worst1k),
                mm=mm_min, b=b, a=a)


def search_pre(p, f, w):
    one = (np.zeros((1, 1)), np.zeros(1), np.zeros(1), 1.0, np.ones(len(w)))
    plants = [one]
    best = []
    for n_int, fc, fzr in itertools.product([1, 2, 3], np.geomspace(20e3, 250e3, 24),
                                            [0.1, 0.15, 0.2, 0.3, 0.5]):
        if n_int == 1 and fzr != 0.1:
            continue
        r = evaluate(dict(n_int=n_int, fc=fc, fz_ratio=fzr), plants, p, f, w)
        if r:
            best.append(r)
    return sorted(best, key=lambda r: -r["g20"])


def search_post(p, f, w):
    plants = []
    for R in LOADS:
        pr = replace(p, r_load=R)
        Phi, Gam, Cz, Dz = plant_discrete(pr, POST_ORDER)
        plants.append((Phi, Gam, Cz, Dz, plant_freq(Phi, Gam, Cz, Dz, w)))
    best = []
    comp = [dict()] + [dict(f_lc=flc, zeta_z=zz, f_p=flc * pr)
                       for flc in [40e3, 45e3, 50e3]
                       for zz in [0.3, 0.5, 0.8] for pr in [3, 6]]
    for n_int, fc, fzr, cmp in itertools.product([1, 2, 3], np.geomspace(2e3, 120e3, 22),
                                                 [0.1, 0.2, 0.3, 0.5], comp):
        if n_int == 1 and fzr != 0.1:
            continue
        cfg = dict(n_int=n_int, fc=fc, fz_ratio=fzr, **cmp)
        r = evaluate(cfg, plants, p, f, w)
        if r:
            best.append(r)
    return sorted(best, key=lambda r: -r["g20"]), plants


def fmt(cfg):
    s = f"n_int={cfg['n_int']}, fc={cfg['fc']/1e3:.1f} кГц, fz=fc·{cfg['fz_ratio']}"
    if cfg.get("f_lc"):
        s += (f", LC-компенсация {cfg['f_lc']/1e3:.0f} кГц ζ={cfg['zeta_z']}, "
              f"полюса {cfg['f_p']/1e3:.0f} кГц")
    return s


def main():
    p = Params()
    f, w = freq_grid(p)
    pre = search_pre(p, f, w)
    post, plants = search_post(p, f, w)
    out = os.path.join(os.path.dirname(__file__), "..", "results")
    with open(os.path.join(out, "loop_design.md"), "w") as fh:
        fh.write("# Подбор регуляторов\n\n")
        fh.write(f"Задержка D = {DELAY} полупериод(а) сверх измерения; запас по модулю "
                 f"min|1+L| ≥ {MM_MIN} (GM ≥ 6 дБ, PM ≥ 29°) на 4 Ом, 8 Ом и без нагрузки; "
                 "устойчивость проверена по собственным числам замкнутой системы.\n\n")
        for name, res in [("Локальная петля (pre, до LC)", pre),
                          ("Глобальная петля (post, после LC)", post)]:
            fh.write(f"## {name}\n\n| Регулятор | Петлевое усиление 1 кГц, дБ | 20 кГц, дБ | min|1+L| |\n|---|---:|---:|---:|\n")
            for r in res[:5]:
                fh.write(f"| {fmt(r['cfg'])} | {r['g1k']:.1f} | {r['g20']:.1f} | {r['mm']:.2f} |\n")
            fh.write("\n")
    for name, res in [("PRE", pre), ("POST", post)]:
        print(name)
        for r in res[:5]:
            print(f"  {fmt(r['cfg']):70s} 1k {r['g1k']:6.1f} dB  20k {r['g20']:6.1f} dB  mm {r['mm']:.2f}")
    return pre, post, plants


if __name__ == "__main__":
    main()
