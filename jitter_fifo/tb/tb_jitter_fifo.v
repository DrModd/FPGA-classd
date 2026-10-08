// Self-checking testbench for jitter_fifo.
//
// Two free-running clocks with a ppm offset, strobes at Fs = clk / 64
// (768 kHz at 49.152 MHz). Input data is a ramp L = idx * 256, R = -L.
// A cubic interpolator reproduces a straight line exactly, so every output
// can be checked bit-exactly:
//   normal playback   : L[n] - L[n-1] = 256
//   slowing down      : 255   (rate 1 - 2^-8)
//   speeding up       : 257   (rate 1 + 2^-8)
// Two blocks of digital silence test the pause logic: the first non-zero
// sample after each pause must be exactly the first input sample after it
// (nothing lost) and the fill must be back at 50 %.
//
//   iverilog -g2005 -o build/tb_jf jitter_fifo/rtl/jitter_fifo.v jitter_fifo/tb/tb_jitter_fifo.v
//   vvp build/tb_jf                         (reader slower: speed-up path)
//   iverilog ... -Ptb_jitter_fifo.RD_PER=20.325 ...   (reader faster: slow-down path)

`timescale 1ns / 1ps

module tb_jitter_fifo;

    parameter real WR_PER = 20.345;          // 49.152 MHz
    parameter real RD_PER = 20.365;          // ~ -1000 ppm: reader slower
    localparam integer AW      = 6;          // 64 samples
    localparam integer STEP_SH = 8;          // 3906 ppm correction rate
    localparam integer DIV     = 64;         // clocks per sample
    localparam integer N_SAMP  = 60000;
    localparam integer P1A = 30000, P1B = 32000;   // pause blocks (input index)
    localparam integer P2A = 40000, P2B = 47000;
    localparam integer HALF = 32;

    reg wr_clk = 0, rd_clk = 0;
    always #(WR_PER / 2) wr_clk = ~wr_clk;
    always #(RD_PER / 2) rd_clk = ~rd_clk;

    reg wr_rst = 1, rd_rst = 1;

    // ---------------- write side source ----------------
    reg [6:0]  wdiv = 0;
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
    reg [6:0] rdiv = 0;
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
    wire        out_valid, flag_low, flag_high, xrun, running, pause;
    wire [AW:0] fill;

    jitter_fifo #(.AW(AW), .DW(32), .STEP_SH(STEP_SH), .FW(16), .SYNC_STB(2)) dut (
        .wr_clk(wr_clk), .wr_rst(wr_rst), .wr_stb(wr_stb), .in_l(in_l), .in_r(in_r),
        .pause_thr(32'd0), .pause_len(24'd200), .wr_full(wr_full), .wr_in_pause(wr_in_pause),
        .rd_clk(rd_clk), .rd_rst(rd_rst), .rd_stb(rd_stb),
        .out_l(out_l), .out_r(out_r), .out_valid(out_valid),
        .flag_low(flag_low), .flag_high(flag_high), .xrun(xrun),
        .running(running), .pause(pause), .fill(fill));

    // ---------------- checks ----------------
    integer n_out = 0, errors = 0, n_exact = 0, n_slow = 0, n_fast = 0;
    integer ev_low = 0, ev_high = 0, ev_pause = 0, n_xrun = 0, n_full = 0;
    integer resumed = 0, pend_bad = 0, edges = 0;
    reg     have_prev = 0, was_low = 0, was_high = 0, was_pause = 0, in_gap = 0;
    reg signed [31:0] prev_l;
    integer d;

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

    always @(posedge rd_clk) if (!rd_rst) begin
        if (xrun) n_xrun = n_xrun + 1;
        if (flag_low  && !was_low)   ev_low   = ev_low + 1;
        if (flag_high && !was_high)  ev_high  = ev_high + 1;
        if (pause     && !was_pause) ev_pause = ev_pause + 1;
        was_low <= flag_low; was_high <= flag_high; was_pause <= pause;

        if (out_valid) begin
            n_out = n_out + 1;
            if ($signed(out_r) != -$signed(out_l)) err("R != -L");
            if (out_l == 0) begin
                // zeros: start-up, pause, or the silent input itself. Up to 4
                // interpolated samples just before a silent block are not on
                // the ramp (the cubic window already sees the zeros): forgiven.
                if (pend_bad > 4) err("too many off-ramp samples before silence");
                if (pend_bad > 0) edges = edges + 1;
                pend_bad = 0;
                if (have_prev) in_gap = 1;
                have_prev = 0;
            end else begin
                if (have_prev) begin
                    d = $signed(out_l) - prev_l;
                    if (pend_bad == 0 && d == 256)      n_exact = n_exact + 1;
                    else if (pend_bad == 0 && d == 255) n_slow  = n_slow + 1;
                    else if (pend_bad == 0 && d == 257) n_fast  = n_fast + 1;
                    else pend_bad = pend_bad + 1;
                    if (pend_bad > 4) err("step not 255/256/257");
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
        $display("outputs %0d: exact %0d, slowed %0d, sped up %0d", n_out, n_exact, n_slow, n_fast);
        $display("interpolation events: low %0d, high %0d; pauses %0d, resumed %0d",
                 ev_low, ev_high, ev_pause, resumed);
        $display("xrun %0d, write drops %0d, edge samples forgiven %0d, errors %0d",
                 n_xrun, n_full, edges, errors);
        if (errors == 0 && n_xrun == 0 && n_full == 0 && ev_pause == 2 && resumed == 2
            && (ev_low + ev_high) > 0)
            $display("PASS");
        else
            $display("FAIL");
        $finish;
    end

endmodule
