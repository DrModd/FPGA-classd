"""Runs the testbench scenario of tb/tb_jitter_fifo.v on the model and
applies the same checks (expected result of the RTL simulation)."""
import sys
import numpy as np
from jfifo_model import JFifo, simulate, S_RUN, M_SLOW, M_FAST, S_PAUSE

P1A, P1B, P2A, P2B, HALF = 30000, 32000, 40000, 47000, 32


def sample(i):
    if P1A <= i < P1B or P2A <= i < P2B:
        return 0, 0
    return i * 256, -i * 256


def check(rd_per):
    ppm = (rd_per / 20.345 - 1) * 1e6          # writer faster when reader period longer
    f = JFifo(AW=6, STEP_SH=8, pause_thr=0, pause_len=200, lat_s=60e-9)
    fs = 49.152e6 / 64
    r = simulate(f, lambda i: sample(i + 1), fs, ppm, 60000 / fs)
    out, fill, st, mode = r["out"][:, 0], r["fill"], r["st"], r["mode"]
    errors = pend = edges = resumed = 0
    n = {255: 0, 256: 0, 257: 0}
    have_prev = in_gap = False
    prev = 0
    for k, o in enumerate(out):
        if o == 0:
            if pend > 3:
                errors += 1
            if pend:
                edges += 1
            pend = 0
            if have_prev:
                in_gap = True
            have_prev = False
            continue
        if have_prev:
            d = o - prev
            if pend == 0 and d in n:
                n[d] += 1
            else:
                pend += 1
            if pend > 3:
                errors += 1
        elif in_gap:
            resumed += 1
            if o not in (P1B * 256, P2B * 256):
                errors += 1
                print("lost samples", o // 256)
            if not (HALF - 2 <= fill[k] <= HALF + 2):
                errors += 1
                print("fill at resume", fill[k])
            in_gap = False
        prev, have_prev = o, True
    run = st == S_RUN
    ev = lambda m: int(np.sum(np.diff(np.concatenate(([0], m.astype(int)))) == 1))
    print(f"rd_per {rd_per}: ppm {ppm:+.0f}, exact {n[256]}, slow {n[255]}, fast {n[257]}, "
          f"low ev {ev(run & (mode == M_SLOW))}, high ev {ev(run & (mode == M_FAST))}, "
          f"pauses {ev(st == S_PAUSE)}, resumed {resumed}, edges {edges}, xrun {f.xruns}, "
          f"drops {f.dropped}, errors {errors}")


for p in (20.365, 20.325):
    check(p)
