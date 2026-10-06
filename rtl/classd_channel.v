// One BTL channel: 24-bit PCM @192 kHz -> BD PWM words for 4 gate inputs.
//
// clk = 49.152 MHz (also PCLK of the OSER16 serialisers, FCLK 393.216 MHz DDR).
// A free-running counter cnt (0..255 = one input sample) sets every phase:
//   cnt == 0        : in_pcm is sampled (sample_req pulses one clock before)
//   slot = cnt[4:0] : position within the PWM half period
// The PCM source must be synchronous to the same 49.152 MHz oscillator
// (ASRC output of ax-bridge) and present a new sample every 256 clocks.

`timescale 1ns / 1ps

module classd_channel #(
    parameter integer W = 28
) (
    input  wire               clk,
    input  wire               rst,
    input  wire               en,
    input  wire [3:0]         dt,          // dead time, ticks of 1.27 ns
    input  wire signed [23:0] in_pcm,
    output wire               sample_req,  // load the next sample now
    output wire [15:0]        a_h, a_l, b_h, b_l,
    // debug / testbench taps
    output wire               interp_stb,
    output wire signed [W-1:0] interp_x,
    output wire               corr_stb,
    output wire signed [W-1:0] corr_m,
    output wire               q_stb,
    output wire signed [9:0]  q,
    output wire [15:0]        sa, sb,
    output wire               half_start,
    output wire signed [9:0]  q_cur
);

    reg [7:0] cnt;
    always @(posedge clk)
        if (rst) cnt <= 8'd0;
        else     cnt <= cnt + 8'd1;

    wire in_stb = (cnt == 8'd0) && !rst;
    assign sample_req = (cnt == 8'd255);

    interp8 #(.W(W)) u_interp (
        .clk(clk), .rst(rst), .in_stb(in_stb), .in_pcm(in_pcm),
        .out_stb(interp_stb), .out_x(interp_x));

    pwm_corr #(.W(W)) u_corr (
        .clk(clk), .rst(rst), .in_stb(interp_stb), .in_x(interp_x),
        .out_stb(corr_stb), .out_y(corr_m));

    ns_shaper #(.W(W)) u_ns (
        .clk(clk), .rst(rst), .in_stb(corr_stb), .in_m(corr_m),
        .out_stb(q_stb), .out_q(q));

    bd_pwm u_pwm (
        .clk(clk), .rst(rst), .en(en), .dt(dt), .slot(cnt[4:0]),
        .q_stb(q_stb), .q(q),
        .a_h(a_h), .a_l(a_l), .b_h(b_h), .b_l(b_l),
        .sa(sa), .sb(sb), .half_start(half_start), .q_cur(q_cur));

endmodule
