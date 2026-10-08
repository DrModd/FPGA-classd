"""
Sample-level model of rtl/jitter_fifo.v (same state machine, same integer
arithmetic of the Catmull-Rom interpolator), driven by two independent
sample clocks. Used to check the algorithm: start-up, drift in both
directions, pauses, interpolation error.

    python jitter_fifo/model/jfifo_model.py   -> jitter_fifo/results.md
"""

import os
import bisect
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results.md")

S_INIT, S_RUN, S_PAUSE = 0, 1, 2
M_NORM, M_SLOW, M_FAST = 0, 1, 2


def rnd(p, s):
    return (p + (1 << (s - 1))) >> s


class JFifo:
    def __init__(self, AW=10, DW=32, LO_PCT=20, HI_PCT=80, FW=16, STEP_SH=10,
                 pause_thr=0, pause_len=0, lat_s=0.0):
        self.AW, self.DW, self.FW = AW, DW, FW
        self.D = 1 << AW
        self.HALF = self.D // 2
        self.LO = self.D * LO_PCT // 100
        self.HI = self.D * HI_PCT // 100
        self.ONE = 1 << FW
        self.DELTA = 1 << (FW - STEP_SH)
        self.thr, self.plen = pause_thr, pause_len
        self.lat = lat_s
        self.mem = [(0, 0, 0, 0)] * self.D
        # write side
        self.wptr = 0
        self.wtimes = []
        self.qcnt = 0
        self.in_pause = False
        self.pend_s = self.pend_e = 0
        self.dropped = 0
        # read side
        self.st, self.mode = S_INIT, M_NORM
        self.rd = 0
        self.frac = 0
        self.t1_s = 0
        self.y = (0, 0)
        self.xruns = 0

    # ---------------- write domain ----------------
    def write(self, t, l, r):
        quiet = abs(l) <= self.thr and abs(r) <= self.thr
        tag_s = tag_e = 0
        if self.plen:
            if quiet:
                if not self.in_pause:
                    if self.qcnt + 1 >= self.plen:
                        tag_s = 1
                        self.in_pause = True
                        self.qcnt = 0
                    else:
                        self.qcnt += 1
            else:
                if self.in_pause:
                    tag_e = 1
                self.in_pause = False
                self.qcnt = 0
        else:
            tag_e = 1 if self.in_pause else 0
            self.in_pause, self.qcnt = False, 0
        used = self.wptr - self.rd
        if used < self.D - 1:
            self.mem[self.wptr % self.D] = (l, r, tag_s | self.pend_s, tag_e | self.pend_e)
            self.wptr += 1
            self.wtimes.append(t)
            self.pend_s = self.pend_e = 0
        else:
            self.dropped += 1
            self.pend_s |= tag_s
            self.pend_e |= tag_e

    # ---------------- read domain -----------------
    def word(self, i):
        return self.mem[i % self.D]

    def read(self, t):
        """Returns (out_l, out_r, info) presented at this strobe."""
        out = self.y
        wvis = bisect.bisect_right(self.wtimes, t - self.lat)
        avail = wvis - self.rd
        info = dict(st=self.st, mode=self.mode, fill=avail, pos=None, interp=False)
        if self.st == S_INIT:
            if avail >= self.HALF:
                self.rd = wvis - self.HALF
                self.frac, self.mode, skip_s = 0, M_NORM, 0
                self.st = S_RUN
                self._fetch(skip_s)
            else:
                self.y = (0, 0)
        elif self.st == S_RUN:
            step = self.ONE - self.DELTA if self.mode == M_SLOW else \
                self.ONE + self.DELTA if self.mode == M_FAST else self.ONE
            psum = self.frac + step
            adv, fr = psum >> self.FW, psum & (self.ONE - 1)
            if avail <= adv + 2 or avail >= self.D - 2:
                self.st, self.mode = S_INIT, M_NORM
                self.xruns += 1
                self.y = (0, 0)
            else:
                skip_s = self.t1_s if adv == 2 else 0
                self.rd += adv
                self.frac = fr
                if self.mode == M_NORM:
                    if avail < self.LO:
                        self.mode = M_SLOW
                    elif avail > self.HI:
                        self.mode = M_FAST
                elif self.mode == M_SLOW:
                    if avail >= self.HALF and fr == 0:
                        self.mode = M_NORM
                else:
                    if avail <= self.HALF and fr == 0:
                        self.mode = M_NORM
                self._fetch(skip_s)
        else:  # pause: consume pc silent words, stop on the first END tag
            ev = [self.word(self.rd + j)[3] for j in (1, 2, 3, 4)]
            pc = 0 if avail < self.HALF else min(avail - self.HALF + 1, 4)
            hit = next((j for j in range(1, pc + 1) if ev[j - 1]), None)
            if hit:
                self.rd += hit
                self.frac, self.mode, self.st = 0, M_NORM, S_RUN
                self._fetch(0)
            else:
                self.rd += pc
                self.y = (0, 0)
        info["pos_next"] = self.rd + self.frac / self.ONE if self.st == S_RUN else None
        info["interp_next"] = self.st == S_RUN and self.frac != 0
        return out, info

    def _fetch(self, skip_s):
        w = [self.word(self.rd + k) for k in (-1, 0, 1, 2)]
        self.t1_s = w[2][2]
        if self.st == S_RUN and (w[1][2] or skip_s) and not w[1][3]:
            self.st = S_PAUSE
        if self.frac == 0:
            self.y = (w[1][0], w[1][1])
            return
        t = self.frac
        ys = []
        for c in (0, 1):
            pm1, p0, p1, p2 = (w[k][c] for k in range(4))
            a = p1 - pm1
            b = 2 * pm1 - 5 * p0 + 4 * p1 - p2
            cc = 3 * p0 - 3 * p1 + p2 - pm1
            h = cc
            h = b + rnd(h * t, self.FW)
            h = a + rnd(h * t, self.FW)
            y = p0 + rnd(h * t, self.FW + 1)
            lim = 1 << (self.DW - 1)
            ys.append(max(-lim, min(lim - 1, y)))
        self.y = tuple(ys)


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def simulate(fifo, x, fs_w, ppm, dur_s, jitter_s=0.0, phase=0.37, seed=1):
    """x: function(index) -> (l, r). Writer runs at fs_w * (1 + ppm),
    reader at fs_w. Returns per-read arrays."""
    rng = np.random.default_rng(seed)
    fw = fs_w * (1 + ppm * 1e-6)
    nw = int(dur_s * fw)
    nr = int(dur_s * fs_w)
    tw = np.arange(nw) / fw + rng.normal(0, jitter_s, nw) if jitter_s else np.arange(nw) / fw
    tr = (np.arange(nr) + phase) / fs_w
    out = np.zeros((nr, 2), dtype=np.int64)
    st = np.zeros(nr, np.int8)
    mode = np.zeros(nr, np.int8)
    fill = np.zeros(nr, np.int32)
    pos = np.full(nr, np.nan)
    i = 0
    prev_pos = None
    for j in range(nr):
        while i < nw and tw[i] < tr[j]:
            fifo.write(tw[i], *x(i))
            i += 1
        o, info = fifo.read(tr[j])
        out[j] = o
        st[j], mode[j], fill[j] = info["st"], info["mode"], info["fill"]
        pos[j] = prev_pos if prev_pos is not None else np.nan
        prev_pos = info["pos_next"]
    return dict(out=out, st=st, mode=mode, fill=fill, pos=pos, fw=fw)


def sine_src(f, fs, amp=0.5, DW=32):
    A = amp * (2 ** (DW - 1) - 1)
    def x(i):
        v = int(round(A * np.sin(2 * np.pi * f * i / fs)))
        return v, -v
    return x, A


def interp_error_db(res, f, fs_w_nominal, A):
    """RMS error of interpolated outputs vs the ideal sine at the read
    position, relative to the sine rms. Position is in input-sample units."""
    pos = res["pos"]
    m = ~np.isnan(pos)
    ideal = A * np.sin(2 * np.pi * f * pos[m] / fs_w_nominal)   # source is index-based
    err = res["out"][m, 0] - ideal
    interp = (pos[m] % 1) != 0
    def db(e):
        return 20 * np.log10(np.sqrt(np.mean(e ** 2)) / (A / np.sqrt(2)) + 1e-30)
    return (db(err[interp]) if interp.any() else None), db(err[~interp]), interp.mean()


def segments(mask):
    """number of contiguous True runs"""
    m = mask.astype(int)
    return int(np.sum(np.diff(np.concatenate(([0], m))) == 1))


def main():
    lines = ["# jitter_fifo: проверка алгоритма на модели\n",
             "Модель `model/jfifo_model.py` повторяет автомат и целочисленную арифметику "
             "RTL (`rtl/jitter_fifo.v`) на уровне отсчётов; запись и чтение идут от двух "
             "независимых тактов, писатель быстрее/медленнее на заданные ppm.\n"]

    # ---- 1. start-up and drift both ways ----
    lines.append("## Старт и уход частоты (AW = 8, 256 отсчётов, STEP_SH = 10 → 977 ppm)\n")
    lines.append("| Сценарий | Нулей на старте | xrun | Включений интерполяции | "
                 "Доля времени с интерполяцией | Мин/макс заполнение после старта | "
                 "Ошибка интерполяции 1 кГц, дБ |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for ppm in (+500, -500, +900, -900, +1200):
        f = JFifo(AW=8)
        x, A = sine_src(1000, 48000)
        r = simulate(f, x, 48000, ppm, 30.0)
        run = r["st"] == S_RUN
        first = int(np.argmax(run))
        zeros = int(np.sum(np.all(r["out"][:first + 1] == 0, axis=1)))
        imask = (r["mode"] != M_NORM) & run
        e_int, e_ex, frac = interp_error_db(r, 1000, 48000, A)
        fl = r["fill"][first + 10:]
        lines.append(f"| писатель {ppm:+d} ppm | {zeros} | {f.xruns} | "
                     f"{segments(imask)} | {imask.mean()*100:.0f} % | {fl.min()}/{fl.max()} | "
                     f"{e_int:.1f} |\n" if e_int is not None else
                     f"| писатель {ppm:+d} ppm | {zeros} | {f.xruns} | {segments(imask)} | "
                     f"{imask.mean()*100:.0f} % | {fl.min()}/{fl.max()} | — |\n")
        print(ppm, zeros, f.xruns, segments(imask), f"{imask.mean():.2f}", fl.min(), fl.max(),
              e_int, flush=True)

    # ---- 2. interpolation error vs frequency ----
    lines.append("\nПри +1200 ppm скорость коррекции (977 ppm) меньше расхождения тактов, "
                 "буфер переполняется — нужен STEP_SH = 9.\n")
    lines.append("\n## Ошибка кубической интерполяции (во время коррекции)\n")
    lines.append("| Частота | 48 кГц, дБ | 192 кГц, дБ |\n|---|---:|---:|\n")
    for fsig in (1000, 5000, 10000, 20000):
        row = []
        for fs in (48000, 192000):
            f = JFifo(AW=8)
            x, A = sine_src(fsig, fs)
            r = simulate(f, x, fs, +700, 6.0)
            e_int, _, _ = interp_error_db(r, fsig, fs, A)
            row.append(e_int)
        lines.append(f"| {fsig/1000:g} кГц | {row[0]:.1f} | {row[1]:.1f} |\n")
        print("interp", fsig, row, flush=True)

    # ---- 3. pauses ----
    lines.append("\n## Паузы (AW = 10, 48 кГц, писатель +300 ppm, порог 16, пауза ≥ 4800 отсчётов)\n")
    fs = 48000
    A = 0.5 * (2 ** 31 - 1)
    music = [(0, 1.0), (3.0, 4.0), (6.0, 6.05), (7.0, 9.0)]   # sine on these intervals (s)

    def src(i):
        tt = i / fs
        on = any(a <= tt < b for a, b in music)
        if not on:
            return 0, 0
        v = int(round(A * np.sin(2 * np.pi * 1000 * i / fs + 0.3)))
        return v, -v

    f = JFifo(AW=10, pause_thr=16, pause_len=4800)
    r = simulate(f, src, fs, +300, 10.0)
    out = r["out"][:, 0]
    st = r["st"]
    # no loss check: every music burst must appear in the output bit-exact and complete
    rep = []
    for a, b in music[1:]:
        i0 = int(np.ceil(a * fs))
        n = int(np.ceil(b * fs)) - i0
        ref = np.array([src(i)[0] for i in range(i0, i0 + n)])
        nz = np.flatnonzero(out != 0)
        cand = nz[np.searchsorted(nz, int(a * fs)):]
        k0 = cand[0] if len(cand) else None
        ok = k0 is not None and np.array_equal(out[k0:k0 + n], ref) if k0 is not None else False
        fill_at = r["fill"][k0] if k0 is not None else None
        rep.append((a, b, ok, fill_at, k0))
        print("burst", a, b, ok, fill_at, flush=True)
    pauses = segments(st == S_PAUSE)
    lines.append(f"Обнаружено пауз: {pauses} (в сигнале 4, включая хвост после 9 с); xrun: {f.xruns}; "
                 f"отброшено при записи: {f.dropped}.\n\n")
    lines.append("| Фрагмент после паузы | Выдан целиком и бит-в-бит | Заполнение в момент "
                 "возобновления (цель 512) |\n|---|---|---:|\n")
    for a, b, ok, fl, _ in rep:
        lines.append(f"| {a:g}–{b:g} с | {'да' if ok else 'НЕТ'} | {fl} |\n")
    with open(OUT, "w") as fh:
        fh.writelines(lines)


if __name__ == "__main__":
    main()
