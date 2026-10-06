// x8 interpolator: 192 kHz -> 384 -> 768 -> 1536 kHz, three halfband stages.
// System clock 49.152 MHz: input every 256 clocks, output every 32 clocks.
// Stage timing (clocks after the input strobe): outputs of stage s are
// registered at +DLY+1 and +DLY+1+PERIOD/2 and are the input strobes of
// stage s+1, so the chain output arrives at a fixed phase:
// 3*(DLY+1) = 75 clocks -> slot 11 of every 32.

`timescale 1ns / 1ps

module interp8 #(
    parameter integer W   = 28,
    parameter integer DLY = 24
) (
    input  wire                clk,
    input  wire                rst,
    input  wire                in_stb,       // every 256 clocks
    input  wire signed [23:0]  in_pcm,       // 24-bit PCM, s.23
    output wire                out_stb,      // every 32 clocks
    output wire signed [W-1:0] out_x         // value = int / 2^24
);

    wire signed [W-1:0] x0 = {{(W-25){in_pcm[23]}}, in_pcm, 1'b0};   // s.23 -> Q.24

    wire                s1_stb, s2_stb;
    wire signed [W-1:0] s1_y, s2_y;

    hb_stage #(.STAGE(1), .M(8), .PERIOD(256), .DLY(DLY), .W(W)) u_hb1 (
        .clk(clk), .rst(rst), .in_stb(in_stb), .in_x(x0),
        .out_stb(s1_stb), .out_y(s1_y));

    hb_stage #(.STAGE(2), .M(7), .PERIOD(128), .DLY(DLY), .W(W)) u_hb2 (
        .clk(clk), .rst(rst), .in_stb(s1_stb), .in_x(s1_y),
        .out_stb(s2_stb), .out_y(s2_y));

    hb_stage #(.STAGE(3), .M(7), .PERIOD(64), .DLY(DLY), .W(W)) u_hb3 (
        .clk(clk), .rst(rst), .in_stb(s2_stb), .in_x(s2_y),
        .out_stb(out_stb), .out_y(out_x));

endmodule
