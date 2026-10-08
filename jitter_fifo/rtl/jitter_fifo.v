// =============================================================================
// jitter_fifo - asynchronous ring buffer with fill-level control for audio
//
//  * 2 channels x DW bits, stored in block RAM (one word = L, R, 2 tag bits)
//  * write and read in independent clock domains, pointers cross via Gray code
//  * start-up: outputs zeros until the buffer is 50 % full, then plays
//  * fill < LO_PCT or > HI_PCT: the read side resamples with a 4-point
//    cubic (Catmull-Rom) interpolator at a rate of 1 -/+ 2^-STEP_SH until
//    the fill is back at 50 % (then it returns to bit-exact playback)
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
// Strobes: wr_stb / rd_stb are the sample clocks (one rising edge per sample)
// and may be asynchronous to wr_clk / rd_clk (SYNC_STB synchroniser stages).
// in_l / in_r must be stable for SYNC_STB + 3 wr_clk cycles after the wr_stb
// rising edge. out_l / out_r change SYNC_STB + 2 rd_clk cycles after the
// rd_stb rising edge (out_valid pulses for one cycle) and then hold for the
// whole sample period.
//
// Clock requirements: wr_clk >= 8 x Fs, rd_clk >= 32 x Fs
// (e.g. 49.152 MHz covers 44.1 ... 768 kHz with margin).
//
// Depth: 2**AW words (AW >= 4). Gray-code pointers need a power of two;
// the largest AW is limited by the block RAM of the device
// (memory width 2*DW + 2 = 66 bits for DW = 32).
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
    parameter integer SYNC_STB = 2     // strobe synchroniser stages, 0 = in-domain
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
        if (LO < 5 || HI > DEPTH - 4 || LO >= HALF || HI <= HALF) begin : bad_thr
            jitter_fifo_ERROR_LO_HI_PCT_out_of_range bad();
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
    // keep the word at rd_int-1 (oldest one the interpolator needs) intact
    wire        full_w = (used_w >= DEPTH - 1);

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
                     Q_PFETCH = 3'd3, Q_PDEC = 3'd4, Q_CALC = 3'd5;

    reg  [1:0]    st, mode;
    reg  [2:0]    seq;
    reg  [2:0]    cnt;
    reg  [AW:0]   rd_int;                     // index of x0
    reg  [FW-1:0] frac;                       // position = rd_int + frac / 2^FW
    reg  [AW:0]   avail;                      // wptr_r - rd_int at the strobe
    reg           t1_s, skip_s, t0_s;         // START tags (x1, skipped word, x0)
    reg           t0_e;                       // END tag of x0
    reg  [3:0]    ev;                         // END tags of rd_int+1 .. +4 (pause)
    reg  [DW-1:0] xm1_l, x0_l, x1_l, x2_l;
    reg  [DW-1:0] xm1_r, x0_r, x1_r, x2_r;
    reg  [DW-1:0] y_l, y_r;                   // next output sample

    assign flag_low  = (st == S_RUN) && (mode == M_SLOW);
    assign flag_high = (st == S_RUN) && (mode == M_FAST);
    assign running   = (st != S_INIT);
    assign pause     = (st == S_PAUSE);

    // RAM read address for the current sequencer step
    // silent words to consume per output sample in pause
    wire [AW:0] over = avail - HALF;
    wire [2:0]  pc   = (avail < HALF) ? 3'd0 :
                       (over >= 3)    ? 3'd4 : over[2:0] + 3'd1;

    assign raddr = (seq == Q_FETCH)  ? rd_int + cnt - 1'b1 :
                   (seq == Q_PFETCH) ? rd_int + cnt + 1'b1 :
                                       rd_int;

    // rate step of the RUN state
    wire [FW+1:0] step = (mode == M_SLOW) ? ONE - DELTA :
                         (mode == M_FAST) ? ONE + DELTA : ONE;
    wire [FW+1:0] psum = {2'b00, frac} + step;
    wire [1:0]    adv  = psum[FW+1:FW];

    // ---- cubic interpolation datapath (one multiplier, sequential) ----
    localparam integer HW = DW + 8;           // Horner accumulator width
    reg               ch;                     // 0 = L, 1 = R
    reg  [2:0]        k;
    reg  signed [HW-1:0] ca, cb, h;
    reg  signed [63:0]   prod;
    wire signed [FW:0]   t = {1'b0, frac};

    wire signed [HW-1:0] pm1 = $signed(ch ? xm1_r : xm1_l);
    wire signed [HW-1:0] p0  = $signed(ch ? x0_r  : x0_l);
    wire signed [HW-1:0] p1  = $signed(ch ? x1_r  : x1_l);
    wire signed [HW-1:0] p2  = $signed(ch ? x2_r  : x2_l);

    // Catmull-Rom: y = x0 + t/2 * (a + t * (b + t * c))
    wire signed [HW-1:0] a_c = p1 - pm1;
    wire signed [HW-1:0] b_c = (pm1 <<< 1) - (p0 <<< 2) - p0 + (p1 <<< 2) - p2;
    wire signed [HW-1:0] c_c = (p0 <<< 1) + p0 - (p1 <<< 1) - p1 + p2 - pm1;

    wire signed [63:0] rnd_fw  = (prod + (64'sd1 <<< (FW - 1))) >>> FW;
    wire signed [63:0] rnd_fw1 = (prod + (64'sd1 <<< FW)) >>> (FW + 1);
    wire signed [63:0] yfull   = p0 + rnd_fw1;
    localparam signed [63:0] YMAX =  (64'sd1 <<< (DW - 1)) - 1;
    localparam signed [63:0] YMIN = -(64'sd1 <<< (DW - 1));
    wire [DW-1:0] ysat = (yfull > YMAX) ? YMAX[DW-1:0] :
                         (yfull < YMIN) ? YMIN[DW-1:0] : yfull[DW-1:0];

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

    always @(posedge rd_clk) begin
        if (rd_rst) begin
            st <= S_INIT; mode <= M_NORM; seq <= Q_IDLE; cnt <= 0;
            rd_int <= 0; frac <= 0; avail <= 0; fill <= 0;
            t1_s <= 1'b0; skip_s <= 1'b0; t0_s <= 1'b0; t0_e <= 1'b0; ev <= 4'd0;
            xm1_l <= 0; x0_l <= 0; x1_l <= 0; x2_l <= 0;
            xm1_r <= 0; x0_r <= 0; x1_r <= 0; x2_r <= 0;
            y_l <= 0; y_r <= 0; out_l <= 0; out_r <= 0;
            out_valid <= 1'b0; xrun <= 1'b0;
            ch <= 1'b0; k <= 0; ca <= 0; cb <= 0; h <= 0; prod <= 0;
        end else begin
            out_valid <= 1'b0;
            xrun      <= 1'b0;

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
                    if ((avail <= {{(AW-1){1'b0}}, adv} + 2) || (avail >= DEPTH - 2)) begin
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
            // read x[-1], x0, x1, x2 (RAM output one clock after the address)
            Q_FETCH: begin
                cnt <= cnt + 1'b1;
                case (cnt)
                3'd1: begin xm1_l <= rdata[DW-1:0]; xm1_r <= rdata[2*DW-1:DW]; end
                3'd2: begin x0_l  <= rdata[DW-1:0]; x0_r  <= rdata[2*DW-1:DW];
                            t0_s  <= rdata[MW-2]; t0_e <= rdata[MW-1]; end
                3'd3: begin x1_l  <= rdata[DW-1:0]; x1_r  <= rdata[2*DW-1:DW];
                            t1_s  <= rdata[MW-2]; end
                3'd4: begin
                    x2_l <= rdata[DW-1:0]; x2_r <= rdata[2*DW-1:DW];
                    // play this (silent) sample, then pause; a pause that already
                    // ended on this very word (END on x0) is not entered
                    if (st == S_RUN && (t0_s || skip_s) && !t0_e)
                        st <= S_PAUSE;
                    if (frac == 0) begin              // on a sample: bit-exact copy
                        y_l <= x0_l;
                        y_r <= x0_r;
                        seq <= Q_IDLE;
                    end else begin
                        ch  <= 1'b0;
                        k   <= 0;
                        seq <= Q_CALC;
                    end
                end
                default: ;
                endcase
            end
            // ---------------------------------------------------------
            // pause: look at the END tags of the next four words
            Q_PFETCH: begin
                cnt <= cnt + 1'b1;
                if (cnt != 3'd0) ev[cnt - 1'b1] <= rdata[MW-1];
                if (cnt == 3'd4) seq <= Q_PDEC;
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
            // ---------------------------------------------------------
            // Catmull-Rom, Horner form, one multiply per step
            Q_CALC: begin
                k <= k + 1'b1;
                case (k)
                3'd0: begin ca <= a_c; cb <= b_c; h <= c_c; end
                3'd1: prod <= h * t;
                3'd2: h <= cb + $signed(rnd_fw[HW-1:0]);
                3'd3: prod <= h * t;
                3'd4: h <= ca + $signed(rnd_fw[HW-1:0]);
                3'd5: prod <= h * t;
                3'd6: begin
                    if (ch == 1'b0) begin
                        y_l <= ysat;
                        ch  <= 1'b1;
                        k   <= 0;
                    end else begin
                        y_r <= ysat;
                        seq <= Q_IDLE;
                    end
                end
                default: ;
                endcase
            end
            default: seq <= Q_IDLE;
            endcase
        end
    end

endmodule
