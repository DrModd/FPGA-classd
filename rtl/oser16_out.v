// 16:1 serialiser for one gate signal.
//
// GOWIN defined : Gowin OSER16 primitive (PCLK = 49.152 MHz, FCLK = 393.216 MHz
//                 from a PLL locked to the same oscillator). Check the port
//                 names, bit order (D0 first?) and FCLK/PCLK ratio for GW5A in
//                 UG289 before synthesis.
// otherwise      : behavioural model for simulation (sends d[0] first).

`timescale 1ns / 1ps

module oser16_out (
    input  wire        pclk,
    input  wire        fclk,
    input  wire        rst,
    input  wire [15:0] d,
    output wire        q
);

`ifdef GOWIN
    OSER16 #(
        .GSREN("false"),
        .LSREN("true")
    ) u_oser (
        .Q(q),
        .D0(d[0]),   .D1(d[1]),   .D2(d[2]),   .D3(d[3]),
        .D4(d[4]),   .D5(d[5]),   .D6(d[6]),   .D7(d[7]),
        .D8(d[8]),   .D9(d[9]),   .D10(d[10]), .D11(d[11]),
        .D12(d[12]), .D13(d[13]), .D14(d[14]), .D15(d[15]),
        .PCLK(pclk), .FCLK(fclk), .RESET(rst)
    );
`else
    // simulation only: fclk here is the full bit clock (786.432 MHz),
    // d[0] leaves first, q lags the parallel word by one bit period
    reg [15:0] sh;
    reg [3:0]  bcnt;
    reg [15:0] d_r;
    always @(posedge pclk) d_r <= d;
    always @(posedge fclk) begin
        if (rst) begin
            bcnt <= 4'd0;
            sh   <= 16'h0;
        end else begin
            sh   <= (bcnt == 4'd0) ? d_r : (sh >> 1);
            bcnt <= bcnt + 4'd1;
        end
    end
    assign q = sh[0];
`endif

endmodule
