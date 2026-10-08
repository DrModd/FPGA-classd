"""Runs the testbench scenarios of tb/tb_jitter_fifo.v on the model and
applies the same checks (expected result of the RTL simulation).

Input is a ramp L = i * 256. Bit-exact playback gives steps of exactly 256;
while interpolating the step is 256 -/+ 1 (rate 1 -/+ 2^-8) plus the
interpolation error of the table on a ramp, which the check allows as TOL.
The script prints the largest deviation seen, so TOL in the testbench can be
checked against it."""
import numpy as np
from jfifo_model import JFifo, simulate, S_RUN, M_SLOW, M_FAST, S_PAUSE

P1A, P1B, P2A, P2B = 30000, 32000, 40000, 47000
WR_PER = 20.345

# name, AW, table, DIV (rd clocks per sample), TOL, reader periods (ns)
CONFIGS = [
    ("cr4", 6, "fir_cr4.hex", 64, 2, (20.365, 20.325)),
    ("ls64", 8, "fir_ls64_m128.hex", 128, 40, (20.406, 20.284)),
]


def sample(i):
    if P1A <= i < P1B or P2A <= i < P2B:
        return 0, 0
    return i * 256, -i * 256


def check(name, AW, table, div, tol, rd_per):
    half = 1 << (AW - 1)
    ppm = (rd_per / WR_PER - 1) * 1e6          # writer faster when reader period longer
    f = JFifo(AW=AW, STEP_SH=8, pause_thr=0, pause_len=200, lat_s=60e-9, table=table)
    nh = f.NH
    fs = 1e9 / WR_PER / div
    r = simulate(f, lambda i: sample(i + 1), fs, ppm, 60000 / fs)
    out, outr, fill, st, mode = r["out"][:, 0], r["out"][:, 1], r["fill"], r["st"], r["mode"]
    errors = pend = edges = resumed = 0
    n_exact = n_slow = n_fast = 0
    worst = 0
    have_prev = in_gap = False
    prev = 0
    for k, o in enumerate(out):
        if abs(int(outr[k]) + int(o)) > 1:
            errors += 1
        if o == 0:
            if pend > f.NT + 2:
                errors += 1
            if pend:
                edges += 1
            pend = 0
            if have_prev:
                in_gap = True
            have_prev = False
            continue
        if have_prev:
            d = int(o - prev)
            if pend == 0 and abs(d - 256) <= 1 + tol:
                worst = max(worst, abs(d - 256))
                if d == 256:
                    n_exact += 1
                elif d < 256:
                    n_slow += 1
                else:
                    n_fast += 1
            else:
                pend += 1
            if pend > f.NT + 2:
                errors += 1
        elif in_gap:
            if o in (P1B * 256, P2B * 256):
                resumed += 1
                if not (half - 2 <= fill[k] <= half + 2):
                    errors += 1
                    print("fill at resume", fill[k])
                in_gap = False
            elif o > P1A * 256:
                errors += 1
                print("lost samples", o // 256)
                in_gap = False
        if not in_gap:
            prev, have_prev = o, True
    run = st == S_RUN
    ev = lambda m: int(np.sum(np.diff(np.concatenate(([0], m.astype(int)))) == 1))
    evp = ev(st == S_PAUSE)
    ok = errors == 0 and f.xruns == 0 and f.dropped == 0 and evp == 2 and resumed == 2 \
        and ev(run & (mode != 0)) > 0
    print(f"{name} rd_per {rd_per}: ppm {ppm:+.0f}, exact {n_exact}, slow {n_slow}, "
          f"fast {n_fast}, max |step-256| {worst} (TOL {tol}), "
          f"low ev {ev(run & (mode == M_SLOW))}, high ev {ev(run & (mode == M_FAST))}, "
          f"pauses {evp}, resumed {resumed}, edges {edges}, xrun {f.xruns}, "
          f"drops {f.dropped}, errors {errors} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    for name, AW, table, div, tol, pers in CONFIGS:
        for p in pers:
            check(name, AW, table, div, tol, p)
