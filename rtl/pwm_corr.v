// Input clip + uniform-PWM linearisation for the BD modulator.
//
//   xc   = clip(x, +-XMAX)                 (min pulse of the ISG3208, headroom)
//   m[k] = xc[k] - (c[k-1] - 2 c[k] + c[k+1]) / 24,   c = xc^3
//
// With the constant-CM BD split every half period carries one pulse with a
// fixed centre; its baseband adds D2(w^3)/24, which this pre-correction
// removes. m[k] is produced when x[k+1] arrives (one sample latency).
// Bit-exact reference: pwm_corr_fixed() in model/fixed_golden.py.

`timescale 1ns / 1ps

module pwm_corr #(
    parameter integer W     = 28,
    parameter integer XMAX  = 16187392,   // (512-12-6)/512 * 2^24
    parameter integer INV24 = 699051      // round(2^24 / 24)
) (
    input  wire                clk,
    input  wire                rst,
    input  wire                in_stb,
    input  wire signed [W-1:0] in_x,
    output reg                 out_stb,
    output reg  signed [W-1:0] out_y
);

    localparam signed [W-1:0] XP = XMAX;
    localparam signed [W-1:0] XN = -XMAX;
    localparam signed [24:0]  INVS = INV24;

    reg [2:0]          ph;            // pipeline phase after in_stb
    reg                busy, have;
    reg signed [W-1:0] xn, xcur;
    reg signed [W-1:0] c1, cn, ccur, cprev;
    reg signed [W-1:0] corr;

    wire signed [W-1:0] xclip = (in_x > XP) ? XP : (in_x < XN) ? XN : in_x;

    wire signed [63:0] sq    = xn * xn;
    wire signed [63:0] cb    = c1 * xn;
    wire signed [W+2:0] d2   = cprev - (ccur <<< 1) + cn;
    wire signed [63:0] cprod = d2 * INVS;

    wire signed [63:0] sq_r  = (sq    + (64'sd1 <<< 23)) >>> 24;
    wire signed [63:0] cb_r  = (cb    + (64'sd1 <<< 23)) >>> 24;
    wire signed [63:0] cp_r  = (cprod + (64'sd1 <<< 23)) >>> 24;

    always @(posedge clk) begin
        if (rst) begin
            ph <= 3'd0; busy <= 1'b0; have <= 1'b0;
            xn <= 0; xcur <= 0; c1 <= 0; cn <= 0; ccur <= 0; cprev <= 0; corr <= 0;
            out_stb <= 1'b0; out_y <= 0;
        end else begin
            out_stb <= 1'b0;
            if (in_stb) begin
                xn   <= xclip;
                ph   <= 3'd1;
                busy <= 1'b1;
            end else if (busy) begin
                ph <= ph + 3'd1;
                case (ph)
                    3'd1: c1   <= sq_r[W-1:0];
                    3'd2: cn   <= cb_r[W-1:0];
                    3'd3: corr <= cp_r[W-1:0];
                    3'd4: begin
                        if (have) begin
                            out_y   <= xcur - corr;
                            out_stb <= 1'b1;
                        end
                        have  <= 1'b1;
                        xcur  <= xn;
                        cprev <= ccur;
                        ccur  <= cn;
                        busy  <= 1'b0;
                    end
                    default: ;
                endcase
            end
        end
    end

endmodule
