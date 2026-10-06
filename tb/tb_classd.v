// Self-checking testbench for classd_channel against the bit-exact Python
// reference (model/fixed_golden.py -> tb/vectors/*.hex).
//
// Checks:
//   1. interpolator output stream  == interp.hex
//   2. PWM-correction output stream == corr.hex
//   3. shaper output q stream       == q.hex
//   4. every half period: high ticks of leg A / B == 256 +- q_cur, and q_cur
//      is the last q delivered by the shaper
//   5. gate words: H = s & s(t - dt), L = ~s & ~s(t - dt), H & L == 0
//   6. shortest high / low run on each leg >= 12 ticks (ISG3208 min pulse)
//
//   make sim        (from the repository root)

`timescale 1ns / 1ps

module tb_classd;

`include "counts.vh"

    localparam integer N_MIN = 12;

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #10.1725 clk = ~clk;       // 49.152 MHz

    reg [23:0] in_mem     [0:N_IN-1];
    reg [27:0] exp_interp [0:N_INTERP-1];
    reg [27:0] exp_corr   [0:N_CORR-1];
    reg [8:0]  exp_q      [0:N_Q-1];

    reg signed [23:0] pcm;
    reg [3:0]         dt;

    wire        sample_req, interp_stb, corr_stb, q_stb, half_start;
    wire [15:0] a_h, a_l, b_h, b_l, sa, sb;
    wire signed [27:0] interp_x, corr_m;
    wire signed [9:0]  q, q_cur;

    classd_channel dut (
        .clk(clk), .rst(rst), .en(1'b1), .dt(dt), .in_pcm(pcm),
        .sample_req(sample_req),
        .a_h(a_h), .a_l(a_l), .b_h(b_h), .b_l(b_l),
        .interp_stb(interp_stb), .interp_x(interp_x),
        .corr_stb(corr_stb), .corr_m(corr_m),
        .q_stb(q_stb), .q(q),
        .sa(sa), .sb(sb), .half_start(half_start), .q_cur(q_cur));

    integer in_idx, ni, nc, nq;
    integer err_i, err_c, err_q, err_pwm, err_gate, halves;
    integer cnt_a, cnt_b, run_a, run_b, min_run;
    reg     lvl_a, lvl_b, have_q, run_valid;
    reg signed [9:0] last_q, q_half;
    reg [15:0] sa_p, sb_p;
    integer k;

    function [15:0] delayed;
        input [15:0] cur;
        input [15:0] prev;
        input [3:0]  d;
        reg   [31:0] c;
        begin
            c = {cur, prev};
            delayed = c >> (16 - d);
        end
    endfunction

    function integer popcount;
        input [15:0] w;
        integer i;
        begin
            popcount = 0;
            for (i = 0; i < 16; i = i + 1) popcount = popcount + w[i];
        end
    endfunction

    initial begin
        $readmemh("tb/vectors/in.hex", in_mem);
        $readmemh("tb/vectors/interp.hex", exp_interp);
        $readmemh("tb/vectors/corr.hex", exp_corr);
        $readmemh("tb/vectors/q.hex", exp_q);
        pcm = in_mem[0];
        in_idx = 1;
        dt = 4'd3;
        ni = 0; nc = 0; nq = 0;
        err_i = 0; err_c = 0; err_q = 0; err_pwm = 0; err_gate = 0; halves = 0;
        cnt_a = 0; cnt_b = 0; have_q = 1'b0; last_q = 0; q_half = 0;
        run_a = 0; run_b = 0; lvl_a = 1'b0; lvl_b = 1'b0; run_valid = 1'b0;
        min_run = 1 << 20;
        sa_p = 16'h0; sb_p = 16'h0;
        repeat (8) @(posedge clk);
        rst <= 1'b0;
    end

    // PCM source: a new sample every 256 clocks
    always @(posedge clk)
        if (!rst && sample_req) begin
            pcm    <= (in_idx < N_IN) ? in_mem[in_idx] : 24'd0;
            in_idx <= in_idx + 1;
        end

    // streams vs. reference
    always @(posedge clk) if (!rst) begin
        if (interp_stb && ni < N_INTERP) begin
            if (interp_x !== exp_interp[ni]) begin
                if (err_i < 10) $display("interp[%0d]: got %h exp %h", ni, interp_x, exp_interp[ni]);
                err_i = err_i + 1;
            end
            ni = ni + 1;
        end
        if (corr_stb && nc < N_CORR) begin
            if (corr_m !== exp_corr[nc]) begin
                if (err_c < 10) $display("corr[%0d]: got %h exp %h", nc, corr_m, exp_corr[nc]);
                err_c = err_c + 1;
            end
            nc = nc + 1;
        end
        if (q_stb) begin
            if (nq < N_Q) begin
                if (q !== {exp_q[nq][8], exp_q[nq]}) begin
                    if (err_q < 10) $display("q[%0d]: got %0d exp %0d", nq, q, $signed(exp_q[nq]));
                    err_q = err_q + 1;
                end
                nq = nq + 1;
            end
            last_q <= q;
            have_q <= 1'b1;
        end
    end

    // PWM words
    always @(posedge clk) if (!rst) begin
        if (half_start) begin
            if (halves > 2) begin
                if (cnt_a != 256 + q_half || cnt_b != 256 - q_half) begin
                    if (err_pwm < 10)
                        $display("half %0d: A %0d B %0d, q %0d", halves, cnt_a, cnt_b, q_half);
                    err_pwm = err_pwm + 1;
                end
                if (have_q && q_cur !== last_q) begin
                    if (err_pwm < 10) $display("half %0d: q_cur %0d, last q %0d", halves, q_cur, last_q);
                    err_pwm = err_pwm + 1;
                end
            end
            halves = halves + 1;
            q_half = q_cur;
            cnt_a = 0;
            cnt_b = 0;
        end
        cnt_a = cnt_a + popcount(sa);
        cnt_b = cnt_b + popcount(sb);

        if ((a_h & a_l) != 0 || (b_h & b_l) != 0
            || a_h !==  (sa & delayed(sa, sa_p, dt)) || a_l !== (~sa & ~delayed(sa, sa_p, dt))
            || b_h !==  (sb & delayed(sb, sb_p, dt)) || b_l !== (~sb & ~delayed(sb, sb_p, dt))) begin
            if (err_gate < 10) $display("gate words wrong at half %0d", halves);
            err_gate = err_gate + 1;
        end
        sa_p <= sa;
        sb_p <= sb;

        // run lengths, bit 0 first
        if (halves > 2) begin
            for (k = 0; k < 16; k = k + 1) begin
                if (sa[k] == lvl_a) run_a = run_a + 1;
                else begin
                    if (run_valid && run_a < min_run) min_run = run_a;
                    lvl_a = sa[k];
                    run_a = 1;
                end
                if (sb[k] == lvl_b) run_b = run_b + 1;
                else begin
                    if (run_valid && run_b < min_run) min_run = run_b;
                    lvl_b = sb[k];
                    run_b = 1;
                end
            end
            run_valid = 1'b1;
        end
    end

    initial begin
        wait (!rst);
        wait (nq >= N_Q);
        repeat (200) @(posedge clk);
        $display("--------------------------------------------------");
        $display("interp: %0d/%0d checked, %0d errors", ni, N_INTERP, err_i);
        $display("corr  : %0d/%0d checked, %0d errors", nc, N_CORR, err_c);
        $display("q     : %0d/%0d checked, %0d errors", nq, N_Q, err_q);
        $display("pwm   : %0d halves, %0d errors; gate words %0d errors", halves, err_pwm, err_gate);
        $display("shortest leg pulse/gap: %0d ticks (limit %0d)", min_run, N_MIN);
        if (err_i == 0 && err_c == 0 && err_q == 0 && err_pwm == 0 && err_gate == 0
            && min_run >= N_MIN && nq == N_Q)
            $display("PASS");
        else
            $display("FAIL");
        $finish;
    end

    initial begin
        #20000000;
        $display("FAIL: timeout");
        $finish;
    end

endmodule
