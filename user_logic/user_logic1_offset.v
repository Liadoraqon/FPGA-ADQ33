/*
 * Copyright 2021 Teledyne Signal Processing Devices Sweden AB
 * Modified 2025 by Liad Panker — Added ZERO block functionality
 */

`default_nettype none
`timescale 1ns / 1ps

`include "databus_type1_parameters.vh"

module user_logic1 #(
   parameter integer S_AXI_DATA_WIDTH = 32,
   parameter integer S_AXI_ADDR_WIDTH = 16,
   parameter integer S_DATABUS_WIDTH  = -1,
   parameter integer M_DATABUS_WIDTH  = -1
)(
   input  wire data_clk_i,
   input  wire data_rst_i,
   input  wire [S_DATABUS_WIDTH-1:0]  s_databus_type1_payload,
   output reg  [M_DATABUS_WIDTH-1:0]  m_databus_type1_payload,

   input  wire s_axi_aclk,
   input  wire s_axi_aresetn,
   input  wire [S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
   input  wire [2:0] s_axi_awprot,
   input  wire s_axi_awvalid,
   output wire s_axi_awready,
   input  wire [S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
   input  wire [(S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
   input  wire s_axi_wvalid,
   output wire s_axi_wready,
   output wire [1:0] s_axi_bresp,
   output wire s_axi_bvalid,
   input  wire s_axi_bready,
   input  wire [S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
   input  wire [2:0] s_axi_arprot,
   input  wire s_axi_arvalid,
   output wire s_axi_arready,
   output wire [S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
   output wire [1:0] s_axi_rresp,
   output wire s_axi_rvalid,
   input  wire s_axi_rready
);

   `include "dbt1_param.vh"
   `include "databus_type1_functions.vh"

   localparam BUS_PIPELINE = 1;

   // Channel data arrays
   reg  [DBT1_CHANNEL_SAMPLEDATA_MAXWIDTH-1:0] my_data[DBT1_BUS_CHANNELS-1:0];
   reg  [DBT1_CHANNEL_SAMPLEDATA_MAXWIDTH-1:0] channel_sampledata_in [DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_SAMPLEDATA_MAXWIDTH-1:0] channel_sampledata_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_SAMPLEDATA_MAXWIDTH-1:0] channel_zeroed        [DBT1_BUS_CHANNELS-1:0];

   // ---- other signal declarations omitted for brevity ----
   // (keep same as Teledyne template)
   // All extract/insert wiring identical to your version.

   wire [S_AXI_DATA_WIDTH-1:0] axi_reg[16:0];
   wire [S_AXI_DATA_WIDTH-1:0] axi_rd_reg[16:0];

   assign axi_rd_reg[0] = 32'h00abcdef;
   genvar r;
   generate
      for (r=1; r<=16; r=r+1)
         assign axi_rd_reg[r] = axi_reg[r];
   endgenerate

   // -------------------------------
   // ZERO block control registers
   // -------------------------------
   wire [31:0] zero_ctrl_axi = axi_reg[3]; // bit0: enable, bits[7:1]: len_pow2
   wire [31:0] zero_ofs_axi  = axi_reg[4]; // offset in data_clk cycles

   // CDC for ZERO control
   wire [31:0] zero_ctrl_dclk, zero_ofs_dclk;

   devkit_cdc_bus #(.WIDTH(32)) cdc_zero_ctrl (
      .src_rst_i(1'b0),
      .src_clk_i(s_axi_aclk),
      .dest_clk_i(data_clk_i),
      .src_valid_i(1'b1),
      .src_data_i(zero_ctrl_axi),
      .src_ack_o(), .src_ready_o(),
      .dest_valid_o(), .dest_data_o(zero_ctrl_dclk)
   );

   devkit_cdc_bus #(.WIDTH(32)) cdc_zero_ofs (
      .src_rst_i(1'b0),
      .src_clk_i(s_axi_aclk),
      .dest_clk_i(data_clk_i),
      .src_valid_i(1'b1),
      .src_data_i(zero_ofs_axi),
      .src_ack_o(), .src_ready_o(),
      .dest_valid_o(), .dest_data_o(zero_ofs_dclk)
   );

   // -------------------------------
   // AXI register instance
   // -------------------------------
   user_logic1_s_axi #(
      .C_S_AXI_DATA_WIDTH(S_AXI_DATA_WIDTH),
      .C_S_AXI_ADDR_WIDTH(S_AXI_ADDR_WIDTH)
   ) user_logic1_s_axi_inst (
      .S_AXI_ACLK(s_axi_aclk),
      .S_AXI_ARESETN(s_axi_aresetn),
      .S_AXI_AWADDR(s_axi_awaddr),
      .S_AXI_AWPROT(s_axi_awprot),
      .S_AXI_AWVALID(s_axi_awvalid),
      .S_AXI_AWREADY(s_axi_awready),
      .S_AXI_WDATA(s_axi_wdata),
      .S_AXI_WSTRB(s_axi_wstrb),
      .S_AXI_WVALID(s_axi_wvalid),
      .S_AXI_WREADY(s_axi_wready),
      .S_AXI_BRESP(s_axi_bresp),
      .S_AXI_BVALID(s_axi_bvalid),
      .S_AXI_BREADY(s_axi_bready),
      .S_AXI_ARADDR(s_axi_araddr),
      .S_AXI_ARPROT(s_axi_arprot),
      .S_AXI_ARVALID(s_axi_arvalid),
      .S_AXI_ARREADY(s_axi_arready),
      .S_AXI_RDATA(s_axi_rdata),
      .S_AXI_RRESP(s_axi_rresp),
      .S_AXI_RVALID(s_axi_rvalid),
      .S_AXI_RREADY(s_axi_rready),

      .slv_reg0 (axi_reg[0]),
      .slv_reg1 (axi_reg[1]),
      .slv_reg2 (axi_reg[2]),
      .slv_reg3 (axi_reg[3]),
      .slv_reg4 (axi_reg[4]),
      .slv_reg5 (axi_reg[5]),
      .slv_reg6 (axi_reg[6]),
      .slv_reg7 (axi_reg[7]),
      .slv_reg8 (axi_reg[8]),
      .slv_reg9 (axi_reg[9]),
      .slv_reg10(axi_reg[10]),
      .slv_reg11(axi_reg[11]),
      .slv_reg12(axi_reg[12]),
      .slv_reg13(axi_reg[13]),
      .slv_reg14(axi_reg[14]),
      .slv_reg15(axi_reg[15]),
      .slv_reg16(axi_reg[16]),

      .slv_reg0_read (axi_rd_reg[0]),
      .slv_reg1_read (axi_rd_reg[1]),
      .slv_reg2_read (axi_rd_reg[2]),
      .slv_reg3_read (axi_rd_reg[3]),
      .slv_reg4_read (axi_rd_reg[4]),
      .slv_reg5_read (axi_rd_reg[5]),
      .slv_reg6_read (axi_rd_reg[6]),
      .slv_reg7_read (axi_rd_reg[7]),
      .slv_reg8_read (axi_rd_reg[8]),
      .slv_reg9_read (axi_rd_reg[9]),
      .slv_reg10_read(axi_rd_reg[10]),
      .slv_reg11_read(axi_rd_reg[11]),
      .slv_reg12_read(axi_rd_reg[12]),
      .slv_reg13_read(axi_rd_reg[13]),
      .slv_reg14_read(axi_rd_reg[14]),
      .slv_reg15_read(axi_rd_reg[15]),
      .slv_reg16_read(axi_rd_reg[16])
   );

   // -------------------------------
   // Channel generate
   // -------------------------------
   genvar ch, k;
   generate
      for (ch=0; ch<DBT1_BUS_CHANNELS; ch=ch+1) begin: gen_channels
         localparam PAR_SAMPLES =
            (ch==0)?DBT1_CHANNEL0_PARALLEL_SAMPLES:
            (ch==1)?DBT1_CHANNEL1_PARALLEL_SAMPLES:
            (ch==2)?DBT1_CHANNEL2_PARALLEL_SAMPLES:
            (ch==3)?DBT1_CHANNEL3_PARALLEL_SAMPLES:
            (ch==4)?DBT1_CHANNEL4_PARALLEL_SAMPLES:
            (ch==5)?DBT1_CHANNEL5_PARALLEL_SAMPLES:
            (ch==6)?DBT1_CHANNEL6_PARALLEL_SAMPLES:
            DBT1_CHANNEL7_PARALLEL_SAMPLES;

         localparam BITS_PER_SAMPLE =
            (ch==0)?DBT1_CHANNEL0_BITS_PER_SAMPLE:
            (ch==1)?DBT1_CHANNEL1_BITS_PER_SAMPLE:
            (ch==2)?DBT1_CHANNEL2_BITS_PER_SAMPLE:
            (ch==3)?DBT1_CHANNEL3_BITS_PER_SAMPLE:
            (ch==4)?DBT1_CHANNEL4_BITS_PER_SAMPLE:
            (ch==5)?DBT1_CHANNEL5_BITS_PER_SAMPLE:
            (ch==6)?DBT1_CHANNEL6_BITS_PER_SAMPLE:
            DBT1_CHANNEL7_BITS_PER_SAMPLE;

         localparam integer CH_WIDTH = PAR_SAMPLES * BITS_PER_SAMPLE;

         // ---- ZERO window logic ----
         wire zero_en    = zero_ctrl_dclk[0];
         wire [6:0] len_pow2  = zero_ctrl_dclk[7:1];
         wire [31:0] ofs_cycles = zero_ofs_dclk;
         wire trig_rise = channel_trigger_primary_rising_in[ch][0];

         reg [31:0] delay_cnt = 0;
         reg [31:0] len_cnt   = 0;
         reg        zero_active = 0;

         always @(posedge data_clk_i) begin
            if (data_rst_i) begin
               delay_cnt   <= 0;
               len_cnt     <= 0;
               zero_active <= 0;
            end else begin
               if (zero_en && trig_rise) begin
                  delay_cnt   <= ofs_cycles;
                  len_cnt     <= (32'd1 << len_pow2);
                  zero_active <= 1'b0;
               end
               if (delay_cnt != 0)
                  delay_cnt <= delay_cnt - 1'b1;
               else if (len_cnt != 0) begin
                  len_cnt <= len_cnt - 1'b1;
                  zero_active <= 1'b1;
               end else
                  zero_active <= 1'b0;
            end
         end

         // Zero output substitution
         assign channel_zeroed[ch][CH_WIDTH-1:0] = zero_active ?
               {CH_WIDTH{1'b0}} :
               channel_sampledata_in[ch][CH_WIDTH-1:0];

         // Use zeroed data (or my_data if test pattern active)
         assign channel_sampledata_out[ch] = data_mux ? my_data[ch] : channel_zeroed[ch];
      end
   endgenerate

   // ---- all other extract/insert logic identical to Teledyne template ----
   // Keep your bus_extraction, bus_insertion, trigger wiring, etc.

endmodule
`default_nettype wire

