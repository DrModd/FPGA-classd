// BD (3-level) centre-aligned PWM for one full bridge, 16 ticks per clock.
//
// Half period = 32 clocks = 512 ticks of 1.27 ns (OSER16 at 786.432 Mbit/s).
// Constant-CM split: nA = 256 + q, nB = 256 - q (q in units of 2 ticks).
//   even half: leg high for the last  n ticks (rising edge at 512 - n)
//   odd  half: leg high for the first n ticks (falling edge at n)
// so every leg pulse is centred on the half-period boundary between an even
// and an odd half.
//
// Per clock the module outputs 16-bit words, bit 0 = first tick in time
// (check the OSER16 bit order on the target). Dead time dt (0..15 ticks):
//   H = s & s(t - dt),  L = ~s & ~s(t - dt)
// so each turn-on is delayed by dt and H, L never overlap. en = 0 forces
// both switches of both legs off.
//
// slot (0..31) must come from the master counter; q is latched at slot 31
// and used for the half starting at the next slot 0.

`timescale 1ns / 1ps

module bd_pwm (
    input  wire              clk,
    input  wire              rst,
    input  wire              en,
    input  wire [3:0]        dt,
    input  wire [4:0]        slot,
    input  wire              q_stb,
    input  wire signed [9:0] q,
    output reg  [15:0]       a_h, a_l, b_h, b_l,
    output reg  [15:0]       sa, sb,       // ideal leg waveforms (debug / test)
    output reg               half_start,   // first word of a half period
    output reg  signed [9:0] q_cur         // q of the current half
);

    reg signed [9:0] q_next;
    reg [9:0]        na, nb;          // 0..512
    reg              odd;             // parity of the current half
    reg [15:0]       sa_prev, sb_prev;

    // bits b < T set
    function [15:0] thermo_lo;
        input signed [11:0] t;
        begin
            if (t <= 0)       thermo_lo = 16'h0000;
            else if (t >= 16) thermo_lo = 16'hFFFF;
            else              thermo_lo = (16'h0001 << t) - 16'h0001;
        end
    endfunction

    // word of a leg for the current slot
    function [15:0] leg_word;
        input [9:0] n;
        input       odd_h;
        input [4:0] j;
        reg signed [11:0] t;
        begin
            if (odd_h) begin
                t = $signed({2'b00, n}) - $signed({3'b000, j, 4'b0000});
                leg_word = thermo_lo(t);                  // high while tick < n
            end else begin
                t = 12'sd512 - $signed({2'b00, n}) - $signed({3'b000, j, 4'b0000});
                leg_word = ~thermo_lo(t);                 // high from tick 512-n
            end
        end
    endfunction

    // s(t - dt) from the current and previous word
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

    wire [15:0] wa  = leg_word(na, odd, slot);
    wire [15:0] wb  = leg_word(nb, odd, slot);
    wire [15:0] wad = delayed(wa, sa_prev, dt);
    wire [15:0] wbd = delayed(wb, sb_prev, dt);

    always @(posedge clk) begin
        if (rst) begin
            q_next <= 10'sd0; q_cur <= 10'sd0;
            na <= 10'd256; nb <= 10'd256; odd <= 1'b1;
            sa_prev <= 16'h0; sb_prev <= 16'h0;
            a_h <= 0; a_l <= 0; b_h <= 0; b_l <= 0; sa <= 0; sb <= 0;
            half_start <= 1'b0;
        end else begin
            if (q_stb)
                q_next <= q;
            // latch for the next half at the last slot of this one
            if (slot == 5'd31) begin
                na    <= 10'd256 + {{1{q_next[9]}}, q_next[8:0]};
                nb    <= 10'd256 - {{1{q_next[9]}}, q_next[8:0]};
                q_cur <= q_next;
                odd   <= ~odd;
            end
            half_start <= (slot == 5'd0);
            sa <= wa;
            sb <= wb;
            sa_prev <= wa;
            sb_prev <= wb;
            a_h <= en ?  (wa &  wad) : 16'h0;
            a_l <= en ? (~wa & ~wad) : 16'h0;
            b_h <= en ?  (wb &  wbd) : 16'h0;
            b_l <= en ? (~wb & ~wbd) : 16'h0;
        end
    end

endmodule
