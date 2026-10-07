"""
Closed loop (dual, D=1) with a realistic feedback ADC: the per-half
measurement error of the ADC model (fb_adc.py, driven by the same switch-node
waveform) is injected into the pre-loop measurement.

    python model/run_fbadc_loop.py   -> results/fbadc_loop.md
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from classd_model import Params, db  # noqa: E402
from closedloop import LoopParams, simulate  # noqa: E402
from fb_adc import AdcParams, error_sequence  # noqa: E402
import run_closedloop as rc  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

# pre-loop controller for the sinc2 kernel (loop_design.search_pre, kernel
# "sinc2", ADC delay 0.1 half): 57 dB at 1 kHz, 10.7 dB at 20 kHz, mm 0.52
PRE_SINC2 = dict(n_int=2, fc=60e3, fz_ratio=0.2)

CASES = [
    ("Идеальное измерение, sinc1", None, "sinc1"),
    ("Идеальное измерение, sinc2 (только цена задержки)", None, "sinc2"),
    ("P1 пассивный RC 98 МГц, sinc1 (первая схема)",
     dict(kind="P1", fs=98.304e6, mode="direct", comp_noise=0.03), "sinc1"),
    ("A2 2-й порядок 49 МГц, sinc1", dict(kind="A2", mode="direct"), "sinc1"),
    ("A2 + реплика ×16, sinc1", dict(kind="A2", gain=16.0), "sinc1"),
    ("A2, sinc2, джиттер 1 пс", dict(kind="A2", mode="direct", jitter=1e-12), "sinc2"),
    ("A2, sinc2, джиттер 5 пс", dict(kind="A2", mode="direct", jitter=5e-12), "sinc2"),
    ("A2 + реплика ×16, sinc2, джиттер 5 пс", dict(kind="A2", gain=16.0, jitter=5e-12), "sinc2"),
]


def main():
    p = Params(deadtime=2e-9, r_load=4.0, n_fft=2 ** 14, pad=2048)
    rows = []
    for lab, adc, kern in CASES:
        err, floor, delay = (None, None, None)
        if adc is not None:
            err, floor, delay = error_sequence(p, AdcParams(**adc), kern)
        pre = rc.PRE[1] if kern == "sinc1" else PRE_SINC2
        lp = LoopParams(mode="dual", delay=1, pre=pre, post=rc.POST["slow"][1],
                        pre_kernel=kern, adc_err=err)
        r = simulate(p, lp)
        rows.append((lab, floor, db(r["thdn"]), db(r["thd"])))
        fl = "—" if floor is None else f"{floor:.1f}"
        print(f"{lab:50s} floor {fl:>7s}  THD+N {db(r['thdn']):7.1f}  THD {db(r['thd']):7.1f}",
              flush=True)
    with open(os.path.join(OUT, "fbadc_loop.md"), "w") as fh:
        fh.write("# Замкнутая петля с реальным АЦП обратной связи\n\n"
                 "dual, D=1, dead time 2 нс, 4 Ом, 1 кГц −1 дБFS. Ошибка измерения АЦП "
                 "(модель fb_adc.py на том же напряжении моста) добавляется к измерению "
                 "pre-петли. «Шум АЦП» — его уровень в полосе 20 кГц относительно полной "
                 "шкалы усилителя.\n\n")
        fh.write("| Измерение | Шум АЦП, дБ | THD+N, дБ | THD, дБ |\n|---|---:|---:|---:|\n")
        for lab, fl, tn, th in rows:
            fls = "—" if fl is None else f"{fl:.1f}"
            fh.write(f"| {lab} | {fls} | {tn:.1f} | {th:.1f} |\n")


if __name__ == "__main__":
    main()
