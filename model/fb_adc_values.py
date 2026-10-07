"""
Component values of the recommended feedback ADC ("AP2 + replica x8 +
sinc2", one BTL channel, fully differential) and a check of the real E96
values against the behavioural model.

    python model/fb_adc_values.py   -> results/fb_adc_values.md
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from classd_model import Params  # noqa: E402
from fb_adc import AdcParams, design, measure  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

VBUS, VL, G = 50.0, 3.3, 8.0
FS = 49.152e6
T = 1 / FS
TAU2 = 0.5e-6
R_BUF, R_FF = 15.0, 20.0            # output resistance of 74LVC buffer / flip-flop
S1 = 0.5                            # FDA differential volts per state unit
E96 = np.array([100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 137,
                140, 143, 147, 150, 154, 158, 162, 165, 169, 174, 178, 182, 187, 191,
                196, 200, 205, 210, 215, 221, 226, 232, 237, 243, 249, 255, 261, 267,
                274, 280, 287, 294, 301, 309, 316, 324, 332, 340, 348, 357, 365, 374,
                383, 392, 402, 412, 422, 432, 442, 453, 464, 475, 487, 499, 511, 523,
                536, 549, 562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732,
                750, 768, 787, 806, 825, 845, 866, 887, 909, 931, 953, 976])
E12C = np.array([10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82])


def e96(x):
    d = 10 ** np.floor(np.log10(x) - 2)
    return E96[np.argmin(abs(E96 * d - x))] * d


def e12c(x):
    d = 10 ** np.floor(np.log10(x) - 1)
    return E12C[np.argmin(abs(E12C * d - x))] * d


def ideal():
    a1, a2 = design(AdcParams(kind="AP2", tau2=TAU2))["a"]
    k0 = design(AdcParams(kind="AP2", tau2=TAU2))["k0"]
    Rin = 47e3
    Rrep = Rin * VL / VBUS - R_BUF
    Rdac1 = G * (Rrep + R_BUF) - R_FF
    C1 = VL * T / ((Rdac1 + R_FF) * a1 * S1)
    R2 = 1e3
    g = a2 * S1 / (R2 * VL)                     # 1/Rdac2 + 1/(Rs + Rk0)
    C2x2 = TAU2 * (1 / R2 + g)                  # "2 C2" (floating C2)
    S2 = T * S1 / (C2x2 * R2)
    Rk0 = 100e3
    Rs = k0 * S2 / VL * Rk0
    Rdac2 = 1 / (g - 1 / (Rs + Rk0)) - R_FF
    return dict(Rin=Rin, Rrep=Rrep, Rdac1=Rdac1, C1=C1, R2=R2, C2=C2x2 / 2,
                Rdac2=Rdac2, Rs=Rs, Rk0=Rk0), (a1, a2, k0)


def effective(v):
    """Normalised loop coefficients realised by the component values v."""
    C2x2 = 2 * v["C2"]
    S2 = T * S1 / (C2x2 * v["R2"])
    a1 = VL * T / ((v["Rdac1"] + R_FF) * v["C1"] * S1)
    X = v["Rs"] + v["Rk0"]
    g = 1 / (v["Rdac2"] + R_FF) + 1 / X
    a2 = T * VL * g / (S2 * C2x2)
    k0 = VL * v["Rs"] / (S2 * v["Rk0"])
    tau2 = C2x2 / (1 / v["R2"] + g)
    s_cmp = S2 * v["Rk0"] / X                   # volts per unit at the comparator
    # replica balance (gain error of the subtraction) and ADC full scale
    bal = (VL / (v["Rrep"] + R_BUF)) / (VBUS / v["Rin"]) - 1
    g_eff = (v["Rdac1"] + R_FF) / (v["Rrep"] + R_BUF)
    return dict(a1=a1, a2=a2, k0=k0, tau2=tau2, s_cmp=s_cmp, bal=bal, G=g_eff)


def main():
    v_id, (a1, a2, k0) = ideal()
    v = dict(Rin=47.0e3, Rrep=e96(v_id["Rrep"]), Rdac1=e96(v_id["Rdac1"]),
             C1=e12c(v_id["C1"]), R2=v_id["R2"], C2=e12c(v_id["C2"]),
             Rdac2=e96(v_id["Rdac2"]), Rs=e96(v_id["Rs"]), Rk0=v_id["Rk0"])
    ef = effective(v)
    vicm = 1.0
    i_cm = VBUS / v["Rin"] + VL / 2 / (v["Rdac1"] + R_FF)
    gsum = 1 / v["Rin"] + 1 / (v["Rrep"] + R_BUF) + 1 / (v["Rdac1"] + R_FF)
    v["Rcm"] = e96(vicm / (i_cm - vicm * gsum))
    print("ideal :", {k: f"{x:.4g}" for k, x in v_id.items()}, f"a=({a1:.3f},{a2:.3f}) k0={k0:.3f}")
    print("E96   :", {k: f"{x:.4g}" for k, x in v.items()})
    print("eff   :", {k: f"{x:.4g}" for k, x in ef.items()})

    p = Params(deadtime=2e-9, r_load=4.0, n_fft=4096 + 256, pad=512)
    rows = []
    for lab, noise_v, hyst_v in [("компаратор идеальный", 0.0, 0.0),
                                 ("шум 0,3 мВ", 0.3e-3, 0.0),
                                 ("шум 0,3 мВ + гистерезис 2 мВ", 0.3e-3, 2e-3),
                                 ("шум 1 мВ + гистерезис 6 мВ", 1e-3, 6e-3)]:
        ap = AdcParams(kind="AP2", gain=ef["G"], tau2=ef["tau2"], jitter=5e-12,
                       coef=(ef["a1"], ef["a2"], ef["k0"]),
                       comp_noise=noise_v / ef["s_cmp"], hyst=hyst_v / ef["s_cmp"])
        r = measure(p, ap, kernels=("sinc2",))
        rows.append((lab, r["sinc2"]))
        print(f"{lab:32s} sinc2 floor {r['sinc2']:.1f} dB", flush=True)

    names = [("Rin", "Ом", "вход с ноги моста, E192, 0,1 %, 0,25 Вт"),
             ("Rrep", "Ом", "от буфера реплики; Rin/Rrep задаёт усиление"),
             ("Rdac1", "Ом", "ЦАП 1-го интегратора; шкала АЦП = Vшины/8"),
             ("Rcm", "Ом", "к земле, синфазный ток, Vicm ≈ 1 В"),
             ("C1", "Ф", "C0G, ОС интегратора, ×2"),
             ("R2", "Ом", "выход ОУ → 2-й интегратор"),
             ("C2", "Ф", "C0G, плавающий между N2+ и N2−"),
             ("Rdac2", "Ом", "ЦАП 2-го интегратора"),
             ("Rs", "Ом", "N2 → вход компаратора"),
             ("Rk0", "Ом", "компенсация ELD")]
    with open(os.path.join(OUT, "fb_adc_values.md"), "w") as fh:
        fh.write("# Номиналы АЦП обратной связи (AP2 + реплика ×8 + sinc2)\n\n"
                 f"Шина {VBUS:.0f} В, опора {VL} В (LDO), такт {FS/1e6:.3f} МГц, τ2 = "
                 f"{TAU2*1e6:.1f} мкс. Значения на одну сторону дифференциальной схемы "
                 "(всё ×2, кроме C2).\n\n| Элемент | Номинал | Назначение |\n|---|---:|---|\n")
        for n, unit, why in names:
            x = v[n]
            if unit == "Ф":
                s = f"{x*1e12:.0f} пФ"
            elif x >= 1e3:
                s = f"{x/1e3:.3g} кОм"
            else:
                s = f"{x:.3g} Ом"
            fh.write(f"| {n} | {s} | {why} |\n")
        fh.write(f"\nРеализованные коэффициенты: a1 = {ef['a1']:.3f} (цель {a1:.3f}), "
                 f"a2 = {ef['a2']:.3f} ({a2:.3f}), k0 = {ef['k0']:.3f} ({k0:.3f}), "
                 f"τ2 = {ef['tau2']*1e6:.2f} мкс, G = {ef['G']:.2f}, разбаланс реплики "
                 f"{ef['bal']*100:+.2f} %, масштаб на входе компаратора "
                 f"{ef['s_cmp']*1e3:.1f} мВ/ед.\n\n")
        fh.write("Проверка на модели (джиттер 5 пс, sinc2, dead time 2 нс, 4 Ом):\n\n"
                 "| Компаратор | Шум измерения, дБ |\n|---|---:|\n")
        for lab, f in rows:
            fh.write(f"| {lab} | {f:.1f} |\n")


if __name__ == "__main__":
    main()
