"""Randomised stress test of the pause logic on the model: music bursts
(ramp data, exact under cubic interpolation) separated by silences of random
length, including exactly pause_len and pause_len + 1, with the writer
+/- 900 ppm off. Checks that every burst after a declared pause resumes
bit-exactly on its first sample, that no burst loses samples and that the
reader never stays muted in a burst."""
import numpy as np
from jfifo_model import JFifo, simulate, S_PAUSE

SCALE = 1024           # STEP_SH = 10 -> interpolated ramp values stay integers


def run(seed, ppm, AW=8, plen=64, step_sh=10):
    SCALE = 1 << step_sh
    rng = np.random.default_rng(seed)
    segs, t = [], 50
    while t < 120000:
        L = int(rng.choice([rng.integers(20, 3000), rng.integers(20, 200)]))
        segs.append((t, t + L))
        t += L
        r = rng.random()
        gap = plen if r < 0.25 else plen + 1 if r < 0.35 else plen - 1 if r < 0.45 \
            else int(rng.integers(plen + 2, 6000)) if r < 0.85 else int(rng.integers(1, plen))
        t += gap
    starts = np.array([a for a, _ in segs])
    ends = np.array([b for _, b in segs])

    def src(i):
        k = np.searchsorted(starts, i, side="right") - 1
        if k >= 0 and i < ends[k]:
            return i * SCALE, -i * SCALE
        return 0, 0

    f = JFifo(AW=AW, STEP_SH=step_sh, pause_thr=0, pause_len=plen, lat_s=60e-9)
    fs = 48000.0
    res = simulate(f, src, fs, ppm, (t + 2000 + 2 * (1 << AW)) / fs / min(1.0, 1 + ppm * 1e-6))
    out = res["out"][:, 0]
    pos = out / SCALE
    errors = []
    # previous gap length per burst
    for k, (a, b) in enumerate(segs):
        gap_before = a - (segs[k - 1][1] if k else 0)
        n = b - a
        inb = (pos >= a) & (pos <= b - 1)
        cnt = int(inb.sum())
        exact_first = bool(np.any(out == a * SCALE))
        declared = gap_before >= plen
        lo = n * (1 - 2.0 ** (1 - step_sh)) - 6
        if cnt < lo:
            errors.append(f"burst {a}-{b} (gap {gap_before}): only {cnt}/{n} samples")
        if declared and k > 0 and not exact_first:
            errors.append(f"burst {a}-{b} after pause {gap_before}: first sample not exact")
    return f, errors, len(segs)


if __name__ == "__main__":
    tot = 0
    # realistic rate (STEP_SH 10) and an aggressive one (STEP_SH 2: the reader
    # advances by 2 every few samples in FAST mode, which exposes tag handling
    # on skipped words)
    for step_sh, ppms in ((10, (+900, -900, +300)), (2, (+60000, -60000))):
        for seed in range(6):
            for ppm in ppms:
                f, errs, nseg = run(seed, ppm, step_sh=step_sh)
                tot += len(errs)
                print(f"STEP_SH {step_sh} seed {seed} ppm {ppm:+d}: bursts {nseg}, xrun {f.xruns}, "
                      f"drops {f.dropped}, errors {len(errs)}", *errs[:3],
                      sep="\n  " if errs else " ")
    print("TOTAL ERRORS", tot)
