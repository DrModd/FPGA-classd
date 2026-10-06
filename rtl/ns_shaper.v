// 5th-order error-feedback noise shaper (NTF with optimised zeros, OBG 3).
//
//   v = m * 256 + w,  w = H e  (H = NTF - 1, transposed DF-II, states z[])
//   q = clamp(round(v), +-LIM)            units of 2 ticks (even BD split)
//   e = clamp(q - v, +-2 LSB)             clamp keeps the loop bounded on overload
//   z[i] = sat( rnd(hb[i]*e) - rnd(a[i]*w) + z[i+1] ),  i = 0..N-1, z[N] = 0
//
// Fractional bits: m Q.24, v/w/e/z Q.24 (in LSB units), coefficients Q.28.
// The state update takes N clocks after in_stb (one pair of multiplies per
// clock); q is available one clock after in_stb.
// Bit-exact reference: shaper_fixed() in model/fixed_golden.py.

`timescale 1ns / 1ps

module ns_shaper #(
    parameter integer W   = 28,
    parameter integer N   = 5,
    parameter integer SW  = 44,       // state width
    parameter integer LIM = 250       // (512 - 12) / 2
) (
    input  wire                clk,
    input  wire                rst,
    input  wire                in_stb,
    input  wire signed [W-1:0] in_m,
    output reg                 out_stb,
    output reg  signed [9:0]   out_q
);

`include "ns_coefs.vh"

    localparam signed [47:0] EMAX = 48'sd2 <<< 24;
    localparam signed [SW-1:0] ZMAX = {1'b0, {(SW-1){1'b1}}};
    localparam signed [SW-1:0] ZMIN = {1'b1, {(SW-1){1'b0}}};

    reg signed [SW-1:0] z [0:N];      // z[N] stays 0
    reg signed [47:0]   e_r;
    reg signed [SW-1:0] w_r;
    reg [3:0]           ui;           // state-update index
    reg                 upd;

    // ---- quantiser (combinational, on in_stb) ----
    wire signed [47:0] v  = ($signed({{(48-W){in_m[W-1]}}, in_m}) <<< 8)
                          + $signed({{(48-SW){z[0][SW-1]}}, z[0]});
    wire signed [47:0] vr = (v + (48'sd1 <<< 23)) >>> 24;
    wire signed [47:0] qc = (vr > LIM) ? LIM : (vr < -LIM) ? -LIM : vr;
    wire signed [47:0] ev = (qc <<< 24) - v;
    wire signed [47:0] ec = (ev > EMAX) ? EMAX : (ev < -EMAX) ? -EMAX : ev;

    // ---- state update term i ----
    wire signed [33:0] chb = ns_hb(ui);
    wire signed [33:0] ca  = ns_a(ui);
    wire signed [79:0] p1  = chb * e_r;
    wire signed [79:0] p2  = ca * w_r;
    wire signed [79:0] r1  = (p1 + (80'sd1 <<< 27)) >>> 28;
    wire signed [79:0] r2  = (p2 + (80'sd1 <<< 27)) >>> 28;
    wire signed [79:0] zn  = r1 - r2 + z[ui + 1];
    wire signed [SW-1:0] zsat = (zn > ZMAX) ? ZMAX : (zn < ZMIN) ? ZMIN : zn[SW-1:0];

    integer j;

    always @(posedge clk) begin
        if (rst) begin
            for (j = 0; j <= N; j = j + 1) z[j] <= {SW{1'b0}};
            e_r <= 0; w_r <= 0; ui <= 0; upd <= 1'b0;
            out_stb <= 1'b0; out_q <= 10'sd0;
        end else begin
            out_stb <= 1'b0;
            if (in_stb) begin
                out_q   <= qc[9:0];
                out_stb <= 1'b1;
                e_r     <= ec;
                w_r     <= z[0];
                ui      <= 4'd0;
                upd     <= 1'b1;
            end else if (upd) begin
                z[ui] <= zsat;        // uses the not yet updated z[ui+1]
                if (ui == N - 1)
                    upd <= 1'b0;
                ui <= ui + 4'd1;
            end
        end
    end

endmodule
