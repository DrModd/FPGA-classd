// =============================================================================
// jitter_fifo - asynchronous ring buffer with fill-level control for audio
//
//  * 2 channels x DW bits, stored in block RAM (one word = L, R, 2 tag bits)
//  * write and read in independent clock domains, pointers cross via Gray code
//  * start-up: outputs zeros until the buffer is 50 % full, then plays
//  * fill < LO_PCT or > HI_PCT: the read side resamples at a rate of
//    1 -/+ 2^-STEP_SH with a polyphase FIR interpolator (NT taps, table of
//    2^MPH_SH + 1 phases in block RAM, coefficients linearly interpolated
//    between neighbouring phases) until the fill is back at 50 %, then it
//    returns to bit-exact playback
//  * pause detector on the input: after pause_len samples with
//    |L|,|R| <= pause_thr a START tag is written with the sample, the first
//    loud sample gets an END tag. The read side acts when it reaches the
//    tagged sample, i.e. exactly where the pause is in the stream:
//      START -> outputs zeros and pulls the fill to 50 % (drops up to 3
//               extra silent samples or holds one per output sample),
//               never skipping past an END tag
//      END   -> waits (zeros) until the fill is >= 50 %, then plays again
//    so the buffer is re-initialised at the start of every pause and is at
//    50 % when the music comes back, however long the pause was.
//
// Interpolator: y = sum_n x[rd_int - NT/2 + 1 + n] * c_n(frac), n = 0..NT-1,
//   c_n = C[j][n] + ((C[j+1][n] - C[j][n]) * mu + 2^(MU-1)) >>> MU,
//   j = frac[FW-1:MU], mu = frac[MU-1:0], MU = FW - MPH_SH,
//   C in Q.(CW-2), file COEF_FILE (hex, address j*NT + n, j = 0..2^MPH_SH),
//   y = sat((sum + 2^(CW-3)) >>> (CW-2)).
// Tables (jitter_fifo/model/gen_coefs.py): fir_ls64_m128.hex (NT 64, MPH_SH 7,
// default), fir_ls64_m64.hex (64, 6), fir_ls32_m128.hex (32, 7, for high
// rates), fir_cr4.hex (4, 10, Catmull-Rom, the old cubic interpolator).
// One tap per rd_clk cycle (the FIFO RAM has one read port); the ROM is read
// through two ports (C[j] and C[j+1]): a true dual-port block RAM.
//
// Strobes: wr_stb / rd_stb are the sample clocks (one rising edge per sample)
// and may be asynchronous to wr_clk / rd_clk (SYNC_STB synchroniser stages).
// in_l / in_r must be stable for SYNC_STB + 3 wr_clk cycles after the wr_stb
// rising edge. out_l / out_r change SYNC_STB + 2 rd_clk cycles after the
// rd_stb rising edge (out_valid pulses for one cycle) and then hold for the
// whole sample period.
//
// Clock requirements: wr_clk >= 8 x Fs, rd_clk >= (NT + 16) x Fs
//   NT = 64: 80 x Fs  (49.152 MHz up to 384 kHz, 98.304 MHz up to 768 kHz)
//   NT = 32: 48 x Fs  (49.152 MHz up to 768 kHz)
//
// Depth: 2**AW words. Gray-code pointers need a power of two; the largest AW
// is limited by the block RAM of the device (memory width 2*DW + 2 = 66 bits
// for DW = 32). LO must be >= NT/2 + 3 and HI <= DEPTH - NT/2 - 2
// (NT = 64: AW >= 8).
//
// Both resets must be applied together (a write-side reset alone loses the
// pause state of the stream).
//
// CDC constraints: the Gray buses wptr_gray -> wg_s1 (rd_clk) and
// rpub_gray -> rg_s1 (wr_clk) need a max-delay / bus-skew constraint of one
// destination clock period (not a plain false path); keep the synchroniser
// flip-flops (wg_s*, rg_s*, wsr, rsr) out of retiming.
//
// Rate offset 2^-STEP_SH must be larger than the worst clock mismatch
// between source and output: STEP_SH = 10 -> 977 ppm (1.7 cents),
// STEP_SH = 9 -> 1953 ppm (3.4 cents, S/PDIF level III sources).
// =============================================================================

`timescale 1ns / 1ps

module jitter_fifo #(
    parameter integer AW       = 10,   // depth = 2**AW samples (L+R)
    parameter integer DW       = 32,   // bits per channel, two's complement
    parameter integer LO_PCT   = 20,   // below: slow-down interpolation
    parameter integer HI_PCT   = 80,   // above: speed-up interpolation
    parameter integer FW       = 16,   // fraction bits of the read position
    parameter integer STEP_SH  = 10,   // rate offset 2^-STEP_SH (STEP_SH <= FW)
    parameter integer SYNC_STB = 2,    // strobe synchroniser stages, 0 = in-domain
    parameter integer NT       = 64,   // interpolator taps (even, >= 4)
    parameter integer MPH_SH   = 7,    // 2^MPH_SH stored phases (1 .. FW-1)
    parameter integer CW       = 30,   // coefficient width, Q.(CW-2)
    parameter          COEF_FILE = "fir_ls64_m128.hex"
) (
    // ---------------- write domain ----------------
    input  wire          wr_clk,
    input  wire          wr_rst,      // synchronous to wr_clk, active high
    input  wire          wr_stb,      // rising edge: write in_l / in_r
    input  wire [DW-1:0] in_l,
    input  wire [DW-1:0] in_r,
    input  wire [DW-1:0] pause_thr,   // |x| <= pause_thr counts as silence
    input  wire [23:0]   pause_len,   // silent samples to declare a pause, 0 = off
    output reg           wr_full,     // pulse: sample dropped, buffer full
    output wire          wr_in_pause, // pause detector state (write side)
    // ---------------- read domain -----------------
    input  wire          rd_clk,
    input  wire          rd_rst,      // synchronous to rd_clk, active high
    input  wire          rd_stb,      // rising edge: next output sample
    output reg  [DW-1:0] out_l,
    output reg  [DW-1:0] out_r,
    output reg           out_valid,   // one rd_clk pulse when out_* are updated
    output wire          flag_low,    // fill < LO_PCT: slowing down (interpolating)
    output wire          flag_high,   // fill > HI_PCT: speeding up (interpolating)
    output reg           xrun,        // pulse: under/overflow, re-initialised
    output reg           rd_late,     // pulse: rd_stb edge while still busy (rd_clk too slow)
    output wire          running,     // initialised, data flowing
    output wire          pause,       // inside a pause (outputs zero)
    output reg  [AW:0]   fill         // samples in the buffer, read-side view
);

    // ------------------------------------------------------------------
    // constants
    // ------------------------------------------------------------------
    localparam integer DEPTH = 1 << AW;
    localparam integer HALF  = DEPTH / 2;
    localparam integer LO    = (DEPTH * LO_PCT) / 100;
    localparam integer HI    = (DEPTH * HI_PCT) / 100;
    localparam integer MW    = 2 * DW + 2;            // {end, start, R, L}
    localparam integer WS    = SYNC_STB + 1;
    localparam integer NH    = NT / 2;                // taps before/after x0: NH-1 / NH
    localparam integer MU    = FW - MPH_SH;           // phase interpolation bits
    localparam integer NPH   = (1 << MPH_SH) + 1;     // stored phases
    localparam integer CSH   = CW - 2;                // coefficient scale 2^CSH

    function integer clog2;
        input integer v;
        integer r;
        begin
            r = 0;
            while ((1 << r) < v) r = r + 1;
            clog2 = r;
        end
    endfunction
    localparam integer RAW   = clog2(NPH * NT);       // coefficient ROM address bits
    localparam integer CNW   = clog2(NT + 8);         // sequencer counter bits

    localparam [FW+1:0] ONE   = {2'b01, {FW{1'b0}}};
    localparam [FW+1:0] DELTA = {{(FW+1){1'b0}}, 1'b1} << (FW - STEP_SH);

    // parameter checks: an unknown module stops elaboration
    generate
        if (AW < 5 || AW > 24) begin : bad_aw
            jitter_fifo_ERROR_AW_must_be_5_to_24 bad();
        end
        if (STEP_SH < 1 || STEP_SH > FW) begin : bad_step
            jitter_fifo_ERROR_STEP_SH_must_be_1_to_FW bad();
        end
        if (LO < NH + 3 || HI > DEPTH - NH - 2 || LO >= HALF || HI <= HALF) begin : bad_thr
            jitter_fifo_ERROR_LO_HI_PCT_out_of_range_or_AW_too_small_for_NT bad();
        end
        if (NT < 4 || (NT % 2) != 0) begin : bad_nt
            jitter_fifo_ERROR_NT_must_be_even_and_at_least_4 bad();
        end
        if (MPH_SH < 1 || MPH_SH > FW - 1) begin : bad_mph
            jitter_fifo_ERROR_MPH_SH_must_be_1_to_FW_minus_1 bad();
        end
    endgenerate

    // ------------------------------------------------------------------
    // Gray code helpers
    // ------------------------------------------------------------------
    function [AW:0] bin2gray;
        input [AW:0] b;
        begin
            bin2gray = b ^ (b >> 1);
        end
    endfunction

    function [AW:0] gray2bin;
        input [AW:0] g;
        integer i;
        begin
            gray2bin[AW] = g[AW];
            for (i = AW - 1; i >= 0; i = i - 1)
                gray2bin[i] = gray2bin[i + 1] ^ g[i];
        end
    endfunction

    // |x| as an unsigned number (|-2^(DW-1)| = 2^(DW-1) fits)
    function [DW-1:0] mag;
        input [DW-1:0] x;
        begin
            mag = x[DW-1] ? (~x + 1'b1) : x;
        end
    endfunction

    // pointers that cross clock domains (declared before use)
    reg  [AW:0] wptr, wptr_gray;              // write domain
    reg  [AW:0] rpub, rpub_gray;              // read domain

    // ------------------------------------------------------------------
    // block RAM: simple dual port, write on wr_clk, read on rd_clk
    // ------------------------------------------------------------------
    (* syn_ramstyle = "block_ram" *)
    reg [MW-1:0] mem [0:DEPTH-1];

    // the word is written on the same edge that advances wptr, so the read
    // side can never see a pointer ahead of the data
    wire          w_wr;
    wire [MW-1:0] wdata;
    always @(posedge wr_clk)
        if (w_wr) mem[wptr[AW-1:0]] <= wdata;

    wire [AW:0]   raddr;
    reg  [MW-1:0] rdata;
    always @(posedge rd_clk)
        rdata <= mem[raddr[AW-1:0]];

    // ==================================================================
    // WRITE DOMAIN
    // ==================================================================
    reg [WS:0] wsr;
    always @(posedge wr_clk)
        if (wr_rst) wsr <= {(WS+1){1'b0}};
        else        wsr <= {wsr[WS-1:0], wr_stb};
    wire w_edge = wsr[WS-1] & ~wsr[WS];

    reg  [AW:0] rg_s1, rg_s2;                 // read pointer (Gray) synchroniser
    always @(posedge wr_clk)
        if (wr_rst) begin rg_s1 <= 0; rg_s2 <= 0; end
        else        begin rg_s1 <= rpub_gray; rg_s2 <= rg_s1; end
    wire [AW:0] rptr_w = gray2bin(rg_s2);
    wire [AW:0] used_w = wptr - rptr_w;
    // keep the word at rd_int-(NH-1) (oldest one the interpolator needs) intact
    wire        full_w = (used_w >= DEPTH - NH + 1);

    // pause detector
    reg [23:0] qcnt;
    reg        in_pause;
    reg        pend_s, pend_e;                // tags waiting for a written word
    assign wr_in_pause = in_pause;

    wire quiet = (mag(in_l) <= pause_thr) && (mag(in_r) <= pause_thr);

    reg tag_s, tag_e;                         // combinational, this sample
    always @* begin
        tag_s = 1'b0;
        tag_e = 1'b0;
        if (pause_len == 24'd0) begin
            tag_e = in_pause;                 // switched off inside a pause
        end else begin
            if (quiet) begin
                if (!in_pause && (qcnt + 24'd1 >= pause_len)) tag_s = 1'b1;
            end else if (in_pause) begin
                tag_e = 1'b1;
            end
        end
    end

    assign w_wr  = w_edge & ~full_w;
    assign wdata = {tag_e | pend_e, tag_s | pend_s, in_r, in_l};

    always @(posedge wr_clk) begin
        if (wr_rst) begin
            wptr      <= 0;
            wptr_gray <= 0;
            wr_full   <= 1'b0;
            qcnt      <= 0;
            in_pause  <= 1'b0;
            pend_s    <= 1'b0;
            pend_e    <= 1'b0;
        end else begin
            wr_full <= 1'b0;
            if (w_edge) begin
                // ---- pause detector state ----
                if (pause_len == 24'd0) begin
                    qcnt     <= 0;
                    in_pause <= 1'b0;
                end else if (quiet) begin
                    if (!in_pause) begin
                        if (tag_s) begin
                            in_pause <= 1'b1;
                            qcnt     <= 0;
                        end else begin
                            qcnt <= qcnt + 24'd1;
                        end
                    end
                end else begin
                    qcnt     <= 0;
                    in_pause <= 1'b0;
                end
                // ---- write (tags survive a dropped sample) ----
                if (!full_w) begin
                    wptr      <= wptr + 1'b1;
                    wptr_gray <= bin2gray(wptr + 1'b1);
                    pend_s    <= 1'b0;
                    pend_e    <= 1'b0;
                end else begin
                    wr_full <= 1'b1;
                    pend_s  <= pend_s | tag_s;
                    pend_e  <= pend_e | tag_e;
                end
            end
        end
    end

    // ==================================================================
    // READ DOMAIN
    // ==================================================================
    reg [WS:0] rsr;
    always @(posedge rd_clk)
        if (rd_rst) rsr <= {(WS+1){1'b0}};
        else        rsr <= {rsr[WS-1:0], rd_stb};
    wire r_edge = rsr[WS-1] & ~rsr[WS];

    reg  [AW:0] wg_s1, wg_s2;                 // write pointer (Gray) synchroniser
    always @(posedge rd_clk)
        if (rd_rst) begin wg_s1 <= 0; wg_s2 <= 0; end
        else        begin wg_s1 <= wptr_gray; wg_s2 <= wg_s1; end
    wire [AW:0] wptr_r = gray2bin(wg_s2);

    // states
    localparam [1:0] S_INIT = 2'd0, S_RUN = 2'd1, S_PAUSE = 2'd2;
    localparam [1:0] M_NORM = 2'd0, M_SLOW = 2'd1, M_FAST = 2'd2;
    // sequencer
    localparam [2:0] Q_IDLE = 3'd0, Q_DEC = 3'd1, Q_FETCH = 3'd2,
                     Q_PFETCH = 3'd3, Q_PDEC = 3'd4;

    reg  [1:0]     st, mode;
    reg  [2:0]     seq;
    reg  [CNW-1:0] cnt;
    reg  [AW:0]    rd_int;                    // index of x0
    reg  [FW-1:0]  frac;                      // position = rd_int + frac / 2^FW
    reg  [AW:0]    avail;                     // wptr_r - rd_int at the strobe
    reg            t1_s, skip_s, t0_s;        // START tags (x1, skipped word, x0)
    reg            t0_e;                      // END tag of x0
    reg  [3:0]     ev;                        // END tags of rd_int+1 .. +4 (pause)
    reg  [DW-1:0]  x0_l, x0_r;
    reg  [DW-1:0]  y_l, y_r;                  // next output sample

    assign flag_low  = (st == S_RUN) && (mode == M_SLOW);
    assign flag_high = (st == S_RUN) && (mode == M_FAST);
    assign running   = (st != S_INIT);
    assign pause     = (st == S_PAUSE);

    // silent words to consume per output sample in pause
    wire [AW:0] over = avail - HALF;
    wire [2:0]  pc   = (avail < HALF) ? 3'd0 :
                       (over >= 3)    ? 3'd4 : over[2:0] + 3'd1;

    // RAM read address for the current sequencer step:
    // FETCH reads tap n = cnt, word rd_int - (NH-1) + n
    assign raddr = (seq == Q_FETCH)  ? rd_int - (NH - 1) + cnt :
                   (seq == Q_PFETCH) ? rd_int + cnt + 1'b1 :
                                       rd_int;

    // rate step of the RUN state
    wire [FW+1:0] step = (mode == M_SLOW) ? ONE - DELTA :
                         (mode == M_FAST) ? ONE + DELTA : ONE;
    wire [FW+1:0] psum = {2'b00, frac} + step;
    wire [1:0]    adv  = psum[FW+1:FW];

    // ------------------------------------------------------------------
    // polyphase FIR datapath, one tap per clock
    //   edge(cnt = n)     : RAM and ROM latch the addresses of tap n
    //   edge(cnt = n + 1) : d1 <= data, cb <= C[j][n], cd <= C[j+1][n] - C[j][n]
    //   edge(cnt = n + 2) : cdm <= cd * mu, cb2 <= cb, d2 <= d1
    //   edge(cnt = n + 3) : coef <= cb2 + round(cdm / 2^MU), d3 <= d2
    //   edge(cnt = n + 4) : p <= d3 * coef
    //   edge(cnt = n + 5) : acc <= acc + p          (cnt = 5 .. NT+4)
    //   cnt = NT + 5      : y = sat(round(acc / 2^CSH))
    // ------------------------------------------------------------------
    (* syn_romstyle = "block_rom" *)
    reg [CW-1:0] rom [0:NPH*NT-1];
    initial $readmemh(COEF_FILE, rom);

    wire [MPH_SH-1:0] ph = frac[FW-1:MU];
    wire [MU-1:0]     mu = frac[MU-1:0];
    // addresses only matter for cnt < NT; kept inside the table otherwise
    wire [RAW-1:0]    ra0 = ph * NT + ((cnt < NT) ? cnt : 0);
    wire [RAW-1:0]    ra1 = ra0 + NT;
    reg  [CW-1:0]     c0q, c1q;
    always @(posedge rd_clk) begin              // dual-port ROM, registered output
        c0q <= rom[ra0];
        c1q <= rom[ra1];
    end

    localparam integer PW = DW + CW + 1;        // product width
    reg  signed [DW-1:0] d1_l, d1_r, d2_l, d2_r, d3_l, d3_r;
    reg  signed [CW-1:0] cb, cb2;
    reg  signed [CW:0]   cd;
    reg  signed [CW+MU+1:0] cdm;
    reg  signed [CW:0]   coef;
    reg  signed [PW-1:0] p_l, p_r;
    reg  signed [63:0]   acc_l, acc_r;

    wire signed [CW+MU+1:0] cdr  = (cdm + (1 <<< (MU - 1))) >>> MU;

    localparam signed [63:0] YMAX =  (64'sd1 <<< (DW - 1)) - 1;
    localparam signed [63:0] YMIN = -(64'sd1 <<< (DW - 1));
    wire signed [63:0] yr_l = (acc_l + (64'sd1 <<< (CSH - 1))) >>> CSH;
    wire signed [63:0] yr_r = (acc_r + (64'sd1 <<< (CSH - 1))) >>> CSH;
    wire [DW-1:0] ys_l = (yr_l > YMAX) ? YMAX[DW-1:0] :
                         (yr_l < YMIN) ? YMIN[DW-1:0] : yr_l[DW-1:0];
    wire [DW-1:0] ys_r = (yr_r > YMAX) ? YMAX[DW-1:0] :
                         (yr_r < YMIN) ? YMIN[DW-1:0] : yr_r[DW-1:0];

    always @(posedge rd_clk) begin              // free-running pipeline stages
        d1_l <= rdata[DW-1:0];
        d1_r <= rdata[2*DW-1:DW];
        cb   <= c0q;
        cd   <= $signed({c1q[CW-1], c1q}) - $signed({c0q[CW-1], c0q});
        cdm  <= cd * $signed({1'b0, mu});
        cb2  <= cb;
        d2_l <= d1_l;
        d2_r <= d1_r;
        coef <= $signed({cb2[CW-1], cb2}) + $signed(cdr[CW:0]);
        d3_l <= d2_l;
        d3_r <= d2_r;
        p_l  <= d3_l * coef;
        p_r  <= d3_r * coef;
    end

    // ---- published read pointer for the write side ----
    // rd_int only moves forward; the published copy follows it one step per
    // clock so that only one Gray bit changes at a time.
    always @(posedge rd_clk)
        if (rd_rst) begin
            rpub      <= 0;
            rpub_gray <= 0;
        end else begin
            if (rpub != rd_int) rpub <= rpub + 1'b1;
            rpub_gray <= bin2gray(rpub);
        end

    // pause entry at the end of a fetch: play this (silent) sample, then
    // pause; a pause that already ended on this very word (END on x0) is not
    // entered
    wire enter_pause = (st == S_RUN) && (t0_s || skip_s) && !t0_e;

    always @(posedge rd_clk) begin
        if (rd_rst) begin
            st <= S_INIT; mode <= M_NORM; seq <= Q_IDLE; cnt <= 0;
            rd_int <= 0; frac <= 0; avail <= 0; fill <= 0;
            t1_s <= 1'b0; skip_s <= 1'b0; t0_s <= 1'b0; t0_e <= 1'b0; ev <= 4'd0;
            x0_l <= 0; x0_r <= 0;
            y_l <= 0; y_r <= 0; out_l <= 0; out_r <= 0;
            out_valid <= 1'b0; xrun <= 1'b0; rd_late <= 1'b0;
            acc_l <= 0; acc_r <= 0;
        end else begin
            out_valid <= 1'b0;
            xrun      <= 1'b0;
            rd_late   <= r_edge && (seq != Q_IDLE);

            case (seq)
            // ---------------------------------------------------------
            Q_IDLE:
                if (r_edge) begin
                    out_l     <= y_l;
                    out_r     <= y_r;
                    out_valid <= 1'b1;
                    avail     <= wptr_r - rd_int;
                    fill      <= wptr_r - rd_int;
                    seq       <= Q_DEC;
                end
            // ---------------------------------------------------------
            Q_DEC:
                case (st)
                S_INIT:
                    if (avail >= HALF) begin
                        rd_int <= wptr_r - HALF;      // start (or re-start) at 50 %
                        frac   <= 0;
                        mode   <= M_NORM;
                        skip_s <= 1'b0;
                        st     <= S_RUN;
                        cnt    <= 0;
                        seq    <= Q_FETCH;
                    end else begin
                        y_l <= 0; y_r <= 0;
                        seq <= Q_IDLE;
                    end
                S_RUN:
                    if ((avail <= {{(AW-1){1'b0}}, adv} + NH) || (avail >= DEPTH - NH)) begin
                        // underflow / overflow: start again from 50 %
                        st   <= S_INIT;
                        mode <= M_NORM;
                        xrun <= 1'b1;
                        y_l  <= 0; y_r <= 0;
                        seq  <= Q_IDLE;
                    end else begin
                        rd_int <= rd_int + adv;
                        frac   <= psum[FW-1:0];
                        skip_s <= (adv == 2'd2) & t1_s;   // START on a skipped word
                        case (mode)
                        M_NORM: if (avail < LO)      mode <= M_SLOW;
                                else if (avail > HI) mode <= M_FAST;
                        M_SLOW: if (avail >= HALF && psum[FW-1:0] == 0) mode <= M_NORM;
                        M_FAST: if (avail <= HALF && psum[FW-1:0] == 0) mode <= M_NORM;
                        default: mode <= M_NORM;
                        endcase
                        cnt <= 0;
                        seq <= Q_FETCH;
                    end
                default: begin                        // S_PAUSE
                    cnt <= 0;
                    seq <= Q_PFETCH;
                end
                endcase
            // ---------------------------------------------------------
            // walk the NT-tap window; x0 = tap NH-1, x1 = tap NH
            Q_FETCH: begin
                cnt <= cnt + 1'b1;
                if (cnt == 0) begin
                    acc_l <= 0;
                    acc_r <= 0;
                end else if (cnt >= 5 && cnt <= NT + 4) begin
                    acc_l <= acc_l + p_l;
                    acc_r <= acc_r + p_r;
                end
                if (cnt == NH) begin                  // data of tap NH-1
                    x0_l <= rdata[DW-1:0];
                    x0_r <= rdata[2*DW-1:DW];
                    t0_s <= rdata[MW-2];
                    t0_e <= rdata[MW-1];
                end
                if (cnt == NH + 1)                    // data of tap NH
                    t1_s <= rdata[MW-2];
                if (frac == 0 && cnt == NH + 1) begin
                    // on a sample: bit-exact copy, the rest of the window is not needed
                    y_l <= x0_l;
                    y_r <= x0_r;
                    if (enter_pause) st <= S_PAUSE;
                    seq <= Q_IDLE;
                end else if (cnt == NT + 5) begin
                    y_l <= ys_l;
                    y_r <= ys_r;
                    if (enter_pause) st <= S_PAUSE;
                    seq <= Q_IDLE;
                end
            end
            // ---------------------------------------------------------
            // pause: look at the END tags of the next four words
            Q_PFETCH: begin
                cnt <= cnt + 1'b1;
                if (cnt != 0) ev[cnt - 1'b1] <= rdata[MW-1];
                if (cnt == 4) seq <= Q_PDEC;
            end
            Q_PDEC: begin
                // consume pc silent words (0 = hold, up to 4 = drop 3 extra) to
                // pull the fill to 50 %; stop on the first END tag: music resumes
                if (pc >= 3'd1 && ev[0]) begin
                    rd_int <= rd_int + 1'b1;
                    frac <= 0; mode <= M_NORM; skip_s <= 1'b0; st <= S_RUN;
                    cnt <= 0; seq <= Q_FETCH;
                end else if (pc >= 3'd2 && ev[1]) begin
                    rd_int <= rd_int + 2'd2;
                    frac <= 0; mode <= M_NORM; skip_s <= 1'b0; st <= S_RUN;
                    cnt <= 0; seq <= Q_FETCH;
                end else if (pc >= 3'd3 && ev[2]) begin
                    rd_int <= rd_int + 2'd3;
                    frac <= 0; mode <= M_NORM; skip_s <= 1'b0; st <= S_RUN;
                    cnt <= 0; seq <= Q_FETCH;
                end else if (pc >= 3'd4 && ev[3]) begin
                    rd_int <= rd_int + 3'd4;
                    frac <= 0; mode <= M_NORM; skip_s <= 1'b0; st <= S_RUN;
                    cnt <= 0; seq <= Q_FETCH;
                end else begin
                    rd_int <= rd_int + pc;
                    y_l <= 0; y_r <= 0;
                    seq <= Q_IDLE;
                end
            end
            default: seq <= Q_IDLE;
            endcase
        end
    end

endmodule
