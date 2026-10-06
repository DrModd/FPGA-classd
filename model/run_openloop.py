"""
Open-loop scenario sweep. Writes results/summary.md and spectra PNGs.

    python model/run_openloop.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from classd_model import Params, run, db  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

SCENARIOS = [
    # (group, label, kwargs)
    ("Модулятор", "UPWM без коррекции, 1 кГц", dict(pwm_corr="none")),
    ("Модулятор", "UPWM без коррекции, 6,6 кГц", dict(pwm_corr="none", f_sig=6600)),
    ("Модулятор", "UPWM без коррекции, 18+19 кГц (IMD)", dict(pwm_corr="none", f_sig=18000, f_sig2=19000)),
    ("Модулятор", "С коррекцией, 1 кГц", {}),
    ("Модулятор", "С коррекцией, 6,6 кГц", dict(f_sig=6600)),
    ("Модулятор", "С коррекцией, 18+19 кГц (IMD)", dict(f_sig=18000, f_sig2=19000)),
    ("Модулятор", "BD-разбиение parity (шаг 1 тик)", dict(bd_split="parity")),
    ("Джиттер", "Джиттер 10 пс RMS", dict(jitter_rms=10e-12)),
    ("Джиттер", "Джиттер 50 пс RMS", dict(jitter_rms=50e-12)),
    ("Джиттер", "Джиттер 100 пс RMS", dict(jitter_rms=100e-12)),
    ("Dead time", "1 нс, 8 Ом", dict(deadtime=1e-9)),
    ("Dead time", "2 нс, 8 Ом", dict(deadtime=2e-9)),
    ("Dead time", "5 нс, 8 Ом", dict(deadtime=5e-9)),
    ("Dead time", "2 нс, 4 Ом", dict(deadtime=2e-9, r_load=4.0)),
    ("Dead time", "5 нс, 4 Ом", dict(deadtime=5e-9, r_load=4.0)),
    ("Dead time", "2 нс, без нагрузки", dict(deadtime=2e-9, r_load=np.inf)),
    ("Dead time", "2 нс, 8 Ом, −20 дБFS", dict(deadtime=2e-9, level_dbfs=-20)),
    ("Питание", "Пульсации шины 1 % / 100 Гц", dict(bus_ripple=0.01)),
    ("Питание", "То же + feedforward", dict(bus_ripple=0.01, feedforward=True)),
]


def spectrum_db(r):
    return db(r["mag"] / 1.0)


def plot_overlay(results, labels, colors, fname, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    for r, lab, c in zip(results, labels, colors):
        f = r["f"]
        m = (f >= 20) & (f <= 22000)
        ax.semilogx(f[m], spectrum_db(r)[m], color=c, lw=1.0, label=lab)
    ax.set_xlim(20, 22000)
    ax.set_ylim(-200, 5)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("дБFS")
    ax.set_title(title, loc="left", color=INK)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def plot_small_multiples(results, labels, fname, title):
    fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.3 * len(results)),
                             sharex=True)
    for ax, r, lab in zip(axes, results, labels):
        f = r["f"]
        m = (f >= 20) & (f <= 22000)
        ax.semilogx(f[m], spectrum_db(r)[m], color=S1, lw=0.9)
        ax.set_ylim(-200, 5)
        ax.set_ylabel("дБFS")
        ax.text(0.01, 0.92, f"{lab}   THD+N {db(r['thdn']):.1f} дБ",
                transform=ax.transAxes, va="top", color=INK)
    axes[-1].set_xlim(20, 22000)
    axes[-1].set_xlabel("Частота, Гц")
    axes[0].set_title(title, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def main():
    rows, res = [], {}
    for group, label, kw in SCENARIOS:
        r = run(Params(**kw))
        res[label] = r
        h = r["harm_db"]
        h3 = f"{h[1]:.1f}" if len(h) > 1 else "—"
        rows.append((group, label, db(r["thdn"]), db(r["thd"]), h3))
        print(f"{label:45s} THD+N {db(r['thdn']):7.1f} dB  THD {db(r['thd']):7.1f} dB")

    p = Params()
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("# Open-loop: результаты модели\n\n")
        fh.write(f"Несущая {p.f_pwm/1e3:.0f} кГц (BD), обновление {p.f_upd/1e6:.3f} МГц, "
                 f"шаг фронта {p.tick*1e9:.3f} нс ({p.nh} тиков на полупериод), "
                 f"NTF {p.ntf_order}-го порядка, OBG {p.ntf_obg}. "
                 f"Шина {p.vbus:.0f} В, L {p.L*1e6:.1f} мкГн, C на землю "
                 f"{p.c_gnd*1e9:.0f} нФ, C дифф. {p.c_diff*1e9:.0f} нФ, Zobel "
                 f"{p.zobel_r} Ом + {p.zobel_c*1e9:.0f} нФ, Coss {p.coss*1e12:.0f} пФ. "
                 f"Сигнал −1 дБFS, 1 кГц, если не указано иное. Полоса 20 Гц–20 кГц.\n\n")
        fh.write("Для двухтонового сигнала колонка THD — это IMD (2f1−f2, 2f2−f1, f2−f1).\n\n")
        fh.write("| Группа | Сценарий | THD+N, дБ | THD, дБ | H3, дБ |\n")
        fh.write("|---|---|---:|---:|---:|\n")
        for g, lab, tn, t, h3 in rows:
            fh.write(f"| {g} | {lab} | {tn:.1f} | {t:.1f} | {h3} |\n")

    plot_overlay([res["BD-разбиение parity (шаг 1 тик)"], res["С коррекцией, 1 кГц"]],
                 ["parity (шаг 1 тик, плавающий CM)", "even (шаг 2 тика, CM постоянный)"],
                 [S2, S1], "bd_split.png",
                 "Разбиение BD между плечами: идеальный тракт, 1 кГц −1 дБFS")
    plot_overlay([res["UPWM без коррекции, 6,6 кГц"], res["С коррекцией, 6,6 кГц"]],
                 ["без коррекции", "коррекция m = x − D²(x³)/24"], [S2, S1],
                 "pwm_corr.png", "Коррекция нелинейности UPWM, 6,6 кГц −1 дБFS")
    plot_small_multiples([res["2 нс, 8 Ом"], res["2 нс, 4 Ом"], res["2 нс, без нагрузки"]],
                         ["8 Ом", "4 Ом", "без нагрузки"], "deadtime_loads.png",
                         "Dead time 2 нс (ISG3208, Coss(TR) 805 пФ), open-loop, 1 кГц −1 дБFS")
    plot_overlay([res["Джиттер 100 пс RMS"], res["Джиттер 10 пс RMS"]],
                 ["100 пс RMS", "10 пс RMS"], [S2, S1], "jitter.png",
                 "Джиттер фронтов, 1 кГц −1 дБFS")


if __name__ == "__main__":
    main()
