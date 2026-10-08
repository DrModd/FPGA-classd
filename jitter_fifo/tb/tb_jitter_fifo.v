// Self-checking testbench for jitter_fifo.
//
// Two free-running clocks with a ppm offset, strobes at Fs = clk / DIV.
// Input data is a ramp L = idx * 256, R = -L.
//   normal playback   : L[n] - L[n-1] = 256 exactly (bit-exact copy)
//   slowing down      : 255 (rate 1 - 2^-8) } +/- TOL: interpolation error
//   speeding up       : 257 (rate 1 + 2^-8) } of the table on a ramp
// Catmull-Rom (fir_cr4.hex) is nearly exact on a straight line (coefficient
// rounding: <= 2 LSB, the model shows |step - 256| <= 1): TOL = 2. The 64-tap
// table has an error of a few tens of LSB at most on the 1.5e7 ramp values
// (about -130 dB): TOL = 40, still well below the 256 of a one-tap shift.
// Interpolating across the jump from the ramp into a silent block rings
// over the whole window: up to NT + 2 off-ramp samples are forgiven there.
// Independently of the ramp, a reference model in the testbench recomputes
// every output bit-exactly from the DUT's read position (rd_int, frac at the
// start of each fetch), the words in its RAM and its own copy of the
// coefficient table (formula of jitter_fifo/model/gen_coefs.py), and compares
// it with the DUT result: this catches phase / mu / pipeline errors that are
// far below TOL.
// Two blocks of digital silence test the pause logic: the first non-zero
// sample after each pause must be exactly the first input sample after it
// (nothing lost) and the fill must be back at 50 %.
//
// Runs (see the Makefile, target jf_sim; expected results from
// jitter_fifo/model/tb_scenario.py):
//   CR4 : AW 6, NT 4,  DIV 64,  RD_PER 20.365 / 20.325 (~ -/+ 1000 ppm)
//   LS64: AW 8, NT 64, DIV 128, RD_PER 20.406 / 20.284 (~ -/+ 3000 ppm)

`timescale 1ns / 1ps

module tb_jitter_fifo;

    parameter real WR_PER = 20.345;          // 49.152 MHz
    parameter real RD_PER = 20.365;          // ~ -1000 ppm: reader slower
    parameter integer AW      = 6;           // 64 samples
    parameter integer NT      = 4;           // interpolator taps
    parameter integer MPH_SH  = 10;          // table phases 2^MPH_SH
    parameter          COEF_FILE = "jitter_fifo/rtl/fir_cr4.hex";
    parameter integer DIV     = 64;          // clocks per sample (>= NT + 16)
    parameter integer TOL     = 2;           // allowed |step - 256| beyond +/- 1
    localparam integer STEP_SH = 8;          // 3906 ppm correction rate
    localparam integer N_SAMP  = 60000;
    localparam integer P1A = 30000, P1B = 32000;   // pause blocks (input index)
    localparam integer P2A = 40000, P2B = 47000;
    localparam integer HALF = (1 << AW) / 2;

    reg wr_clk = 0, rd_clk = 0;
    always #(WR_PER / 2) wr_clk = ~wr_clk;
    always #(RD_PER / 2) rd_clk = ~rd_clk;

    reg wr_rst = 1, rd_rst = 1;

    // ---------------- write side source ----------------
    reg [7:0]  wdiv = 0;
    reg        wr_stb = 0;
    reg [31:0] in_l = 0, in_r = 0;
    integer    idx = 0;

    function [31:0] sample_l;
        input integer i;
        begin
            if ((i >= P1A && i < P1B) || (i >= P2A && i < P2B)) sample_l = 0;
            else sample_l = i * 256;
        end
    endfunction

    always @(posedge wr_clk) begin
        if (wr_rst) begin
            wdiv <= 0; wr_stb <= 0; idx <= 1;
        end else begin
            wdiv <= (wdiv == DIV - 1) ? 0 : wdiv + 1;
            if (wdiv == 0) begin
                // new data, then the strobe rises half a sample later
                in_l <= sample_l(idx);
                in_r <= -sample_l(idx);
                idx  <= idx + 1;
            end
            wr_stb <= (wdiv >= DIV / 4) && (wdiv < 3 * DIV / 4);
        end
    end

    // ---------------- read side strobe ----------------
    reg [7:0] rdiv = 0;
    reg       rd_stb = 0;
    always @(posedge rd_clk) begin
        if (rd_rst) begin rdiv <= 0; rd_stb <= 0; end
        else begin
            rdiv   <= (rdiv == DIV - 1) ? 0 : rdiv + 1;
            rd_stb <= (rdiv < DIV / 2);
        end
    end

    // ---------------- DUT ----------------
    wire        wr_full, wr_in_pause;
    wire [31:0] out_l, out_r;
    wire        out_valid, flag_low, flag_high, xrun, rd_late, running, pause;
    wire [AW:0] fill;

    jitter_fifo #(.AW(AW), .DW(32), .STEP_SH(STEP_SH), .FW(16), .SYNC_STB(2),
                  .NT(NT), .MPH_SH(MPH_SH), .COEF_FILE(COEF_FILE)) dut (
        .wr_clk(wr_clk), .wr_rst(wr_rst), .wr_stb(wr_stb), .in_l(in_l), .in_r(in_r),
        .pause_thr(32'd0), .pause_len(24'd200), .wr_full(wr_full), .wr_in_pause(wr_in_pause),
        .rd_clk(rd_clk), .rd_rst(rd_rst), .rd_stb(rd_stb),
        .out_l(out_l), .out_r(out_r), .out_valid(out_valid),
        .flag_low(flag_low), .flag_high(flag_high), .xrun(xrun), .rd_late(rd_late),
        .running(running), .pause(pause), .fill(fill));

    // ---------------- checks ----------------
    integer n_out = 0, errors = 0, n_exact = 0, n_slow = 0, n_fast = 0, worst = 0;
    integer ev_low = 0, ev_high = 0, ev_pause = 0, n_xrun = 0, n_full = 0;
    integer resumed = 0, pend_bad = 0, edges = 0;
    reg     have_prev = 0, was_low = 0, was_high = 0, was_pause = 0, in_gap = 0;
    reg signed [31:0] prev_l;
    integer d, ad, lr;

    task err;
        input [8*64-1:0] msg;
        begin
            errors = errors + 1;
            if (errors <= 20)
                $display("ERROR @%0t out#%0d: %0s  L=%0d prev=%0d R=%0d fill=%0d",
                         $time, n_out, msg, $signed(out_l), prev_l, $signed(out_r), fill);
        end
    endtask

    always @(posedge wr_clk) if (!wr_rst && wr_full) n_full = n_full + 1;

    // ---------------- bit-exact reference of the interpolator ----------------
    localparam integer NH    = NT / 2;
    localparam integer MU    = 16 - MPH_SH;
    localparam integer DEPTH = 1 << AW;
    reg [29:0] trom [0:((1 << MPH_SH) + 1) * NT - 1];
    initial $readmemh(COEF_FILE, trom);

    integer           n_ref = 0, n_ref_int = 0, ref_err = 0, n_late = 0;
    integer           tn, tj, tmu;
    reg               chk_pend = 0, ref_x = 0;
    reg  [AW-1:0]     ta;
    reg  [65:0]       tw;
    reg  signed [63:0] tc0, tc1, tcc, tacc_l, tacc_r, ty_l, ty_r;
    reg  [31:0]       ex_l, ex_r;

    function signed [63:0] sx30;
        input [29:0] v;
        begin
            sx30 = {{34{v[29]}}, v};
        end
    endfunction

    function [31:0] sat32;
        input signed [63:0] v;
        begin
            if (v > 64'sh7fffffff)       sat32 = 32'h7fffffff;
            else if (v < -64'sh80000000) sat32 = 32'h80000000;
            else                         sat32 = v[31:0];
        end
    endfunction

    always @(posedge rd_clk) if (!rd_rst) begin
        if (rd_late) n_late = n_late + 1;
        // start of a fetch (seq == Q_FETCH, cnt == 0): position is final
        if (dut.seq == 3'd2 && dut.cnt == 0) begin
            if (dut.frac == 0) begin
                ta = dut.rd_int;
                tw = dut.mem[ta];
                ex_l = tw[31:0];
                ex_r = tw[63:32];
                ref_x = 0;
            end else begin
                tj  = dut.frac >> MU;
                tmu = dut.frac & ((1 << MU) - 1);
                tacc_l = 0;
                tacc_r = 0;
                for (tn = 0; tn < NT; tn = tn + 1) begin
                    ta  = dut.rd_int - (NH - 1) + tn;
                    tw  = dut.mem[ta];
                    tc0 = sx30(trom[tj * NT + tn]);
                    tc1 = sx30(trom[(tj + 1) * NT + tn]);
                    tcc = tc0 + (((tc1 - tc0) * tmu + (64'sd1 <<< (MU - 1))) >>> MU);
                    tacc_l = tacc_l + $signed({{32{tw[31]}}, tw[31:0]})  * tcc;
                    tacc_r = tacc_r + $signed({{32{tw[63]}}, tw[63:32]}) * tcc;
                end
                ty_l = (tacc_l + (64'sd1 <<< 27)) >>> 28;     // CW = 30: Q.28
                ty_r = (tacc_r + (64'sd1 <<< 27)) >>> 28;
                ex_l = sat32(ty_l);
                ex_r = sat32(ty_r);
                ref_x = 1;
            end
            chk_pend <= 1'b1;
        end
        // the fetch has finished when the sequencer is back in Q_IDLE
        if (chk_pend && dut.seq == 3'd0) begin
            chk_pend <= 1'b0;
            if (^{ex_l, ex_r} !== 1'bx) begin      // window fully written
                n_ref = n_ref + 1;
                if (ref_x) n_ref_int = n_ref_int + 1;
                if (dut.y_l !== ex_l || dut.y_r !== ex_r) begin
                    ref_err = ref_err + 1;
                    if (ref_err <= 10)
                        $display("REF MISMATCH @%0t rd_int=%0d frac=%0d: dut %0d/%0d model %0d/%0d",
                                 $time, dut.rd_int, dut.frac, $signed(dut.y_l), $signed(dut.y_r),
                                 $signed(ex_l), $signed(ex_r));
                end
            end
        end
    end

    always @(posedge rd_clk) if (!rd_rst) begin
        if (xrun) n_xrun = n_xrun + 1;
        if (flag_low  && !was_low)   ev_low   = ev_low + 1;
        if (flag_high && !was_high)  ev_high  = ev_high + 1;
        if (pause     && !was_pause) ev_pause = ev_pause + 1;
        was_low <= flag_low; was_high <= flag_high; was_pause <= pause;

        if (out_valid) begin
            n_out = n_out + 1;
            // R = -L up to one LSB (rounding of a negated sum)
            lr = $signed(out_r) + $signed(out_l);
            if (lr > 1 || lr < -1) err("R != -L");
            if (out_l == 0) begin
                // zeros: start-up, pause, or the silent input itself. Up to
                // NT + 2 interpolated samples around the jump into a silent
                // block are not on the ramp (the window sees it): forgiven.
                if (pend_bad > NT + 2) err("too many off-ramp samples at silence");
                if (pend_bad > 0) edges = edges + 1;
                pend_bad = 0;
                if (have_prev) in_gap = 1;
                have_prev = 0;
            end else begin
                if (have_prev) begin
                    d  = $signed(out_l) - prev_l;
                    ad = (d > 256) ? d - 256 : 256 - d;
                    if (pend_bad == 0 && ad <= 1 + TOL) begin
                        if (ad > worst) worst = ad;
                        if (d == 256)     n_exact = n_exact + 1;
                        else if (d < 256) n_slow  = n_slow + 1;
                        else              n_fast  = n_fast + 1;
                    end else pend_bad = pend_bad + 1;
                    if (pend_bad > NT + 2) err("step off the ramp");
                end else if (in_gap) begin
                    // first sample after a pause must be exactly the first input
                    // after it; small interpolated values right at the start of a
                    // silent block are ignored (a lost start shows up as resumed < 2)
                    if ($signed(out_l) == P1B * 256 || $signed(out_l) == P2B * 256) begin
                        resumed = resumed + 1;
                        if (fill < HALF - 2 || fill > HALF + 2)
                            err("fill not at 50 % when the music resumes");
                        in_gap = 0;
                    end else if ($signed(out_l) > P1A * 256) begin
                        err("lost samples at the end of a pause");
                        in_gap = 0;
                    end
                end
                if (!in_gap) begin            // artifacts inside a gap are not a reference
                    prev_l    = $signed(out_l);
                    have_prev = 1;
                end
            end
        end
    end

    initial begin
        repeat (20) @(negedge wr_clk);
        wr_rst <= 0;
        repeat (7) @(negedge rd_clk);
        rd_rst <= 0;
        wait (idx >= N_SAMP);
        repeat (DIV * (HALF + 8)) @(posedge rd_clk);
        $display("------------------------------------------------------------");
        $display("NT %0d, AW %0d, %0s", NT, AW, COEF_FILE);
        $display("outputs %0d: exact %0d, slowed %0d, sped up %0d, max |step-256| %0d (TOL %0d)",
                 n_out, n_exact, n_slow, n_fast, worst, TOL);
        $display("interpolation events: low %0d, high %0d; pauses %0d, resumed %0d",
                 ev_low, ev_high, ev_pause, resumed);
        $display("xrun %0d, write drops %0d, late strobes %0d, edge samples forgiven %0d, errors %0d",
                 n_xrun, n_full, n_late, edges, errors);
        $display("reference model: %0d outputs compared (%0d interpolated), %0d mismatches",
                 n_ref, n_ref_int, ref_err);
        if (errors == 0 && n_xrun == 0 && n_full == 0 && n_late == 0 && ev_pause == 2
            && resumed == 2 && (ev_low + ev_high) > 0 && ref_err == 0 && n_ref_int > 0)
            $display("PASS");
        else
            $display("FAIL");
        $finish;
    end

endmodule
