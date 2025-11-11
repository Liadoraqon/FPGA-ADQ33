/*
 * Copyright 2021 Teledyne Signal Processing Devices Sweden AB
 */
`default_nettype none
`timescale 1ns / 1ps

`include "databus_type1_parameters.vh"

module user_logic1 #(
   parameter integer S_AXI_DATA_WIDTH = 32,
   parameter integer S_AXI_ADDR_WIDTH = 16,
   parameter integer S_DATABUS_WIDTH = -1,
   parameter integer M_DATABUS_WIDTH = -1
)(
   input wire data_clk_i,
   input wire data_rst_i,
   input wire [S_DATABUS_WIDTH-1:0] s_databus_type1_payload,
   output reg [M_DATABUS_WIDTH-1:0] m_databus_type1_payload,

   input wire s_axi_aclk,
   input wire s_axi_aresetn,
   input wire [S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
   input wire [2:0] s_axi_awprot,
   input wire s_axi_awvalid,
   output wire s_axi_awready,
   input wire [S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
   input wire [(S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
   input wire s_axi_wvalid,
   output wire s_axi_wready,
   output wire [1:0] s_axi_bresp,
   output wire s_axi_bvalid,
   input wire s_axi_bready,
   input wire [S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
   input wire [2:0] s_axi_arprot,
   input wire s_axi_arvalid,
   output wire s_axi_arready,
   output wire [S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
   output wire [1:0] s_axi_rresp,
   output wire s_axi_rvalid,
   input wire s_axi_rready
);

  `include "dbt1_param.vh"
  `include "databus_type1_functions.vh"

   /* Set BUS_PIPELINE to the computational latency through this module. Bus
      signals that are _not inserted_ are delayed by this number of data clock
      cycles. */
   localparam BUS_PIPELINE = 1;

   reg [DBT1_CHANNEL_SAMPLEDATA_MAXWIDTH-1:0] my_data[DBT1_BUS_CHANNELS-1:0];

   reg [DBT1_CHANNEL_TRIGGER_SECONDARY_EVENT_WIDTH-1:0] channel_trigger_secondary_event_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_TRIGGER_SECONDARY_RISING_WIDTH-1:0] channel_trigger_secondary_rising_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_TRIGGER_SECONDARY_SAMPLEINDEX_FRACTION_WIDTH-1:0] channel_trigger_secondary_sampleindex_fraction_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_TRIGGER_SECONDARY_SAMPLEINDEX_WIDTH-1:0] channel_trigger_secondary_sampleindex_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_TRIGGER_PRIMARY_EVENT_WIDTH-1:0] channel_trigger_primary_event_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_TRIGGER_PRIMARY_RISING_WIDTH-1:0] channel_trigger_primary_rising_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_TRIGGER_PRIMARY_SAMPLEINDEX_FRACTION_WIDTH-1:0] channel_trigger_primary_sampleindex_fraction_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_TRIGGER_PRIMARY_SAMPLEINDEX_WIDTH-1:0] channel_trigger_primary_sampleindex_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_GENERAL_PURPOSE_WIDTH-1:0] channel_general_purpose_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_TRIGGER_PRIMARY_INHIBIT_WIDTH-1:0] channel_trigger_primary_inhibit_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_COMMON_TIMESTAMP_WIDTH-1:0] common_timestamp_in;
   reg [DBT1_COMMON_TIMESTAMP_SYNC_COUNT_WIDTH-1:0] common_timestamp_sync_count_in;
   reg [DBT1_CHANNEL_OVERRANGE_WIDTH-1:0] channel_overrange_in[DBT1_BUS_CHANNELS-1:0];
   reg [DBT1_CHANNEL_SAMPLEDATA_MAXWIDTH-1:0] channel_sampledata_in[DBT1_BUS_CHANNELS-1:0];

   wire [DBT1_CHANNEL_TRIGGER_SECONDARY_EVENT_WIDTH-1:0] channel_trigger_secondary_event_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_TRIGGER_SECONDARY_RISING_WIDTH-1:0] channel_trigger_secondary_rising_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_TRIGGER_SECONDARY_SAMPLEINDEX_FRACTION_WIDTH-1:0] channel_trigger_secondary_sampleindex_fraction_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_TRIGGER_SECONDARY_SAMPLEINDEX_WIDTH-1:0] channel_trigger_secondary_sampleindex_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_TRIGGER_PRIMARY_EVENT_WIDTH-1:0] channel_trigger_primary_event_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_TRIGGER_PRIMARY_RISING_WIDTH-1:0] channel_trigger_primary_rising_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_TRIGGER_PRIMARY_SAMPLEINDEX_FRACTION_WIDTH-1:0] channel_trigger_primary_sampleindex_fraction_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_TRIGGER_PRIMARY_SAMPLEINDEX_WIDTH-1:0] channel_trigger_primary_sampleindex_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_GENERAL_PURPOSE_WIDTH-1:0] channel_general_purpose_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_TRIGGER_PRIMARY_INHIBIT_WIDTH-1:0] channel_trigger_primary_inhibit_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_COMMON_TIMESTAMP_WIDTH-1:0] common_timestamp_out;
   wire [DBT1_COMMON_TIMESTAMP_SYNC_COUNT_WIDTH-1:0] common_timestamp_sync_count_out;
   wire [DBT1_CHANNEL_OVERRANGE_WIDTH-1:0] channel_overrange_out[DBT1_BUS_CHANNELS-1:0];
   wire [DBT1_CHANNEL_SAMPLEDATA_MAXWIDTH-1:0] channel_sampledata_out[DBT1_BUS_CHANNELS-1:0];

   wire [S_AXI_DATA_WIDTH-1:0] axi_reg[16:0];
   wire [S_AXI_DATA_WIDTH-1:0] axi_rd_reg[16:0];

   assign axi_rd_reg[0] = 32'h00abcdef;
   assign axi_rd_reg[1] = axi_reg[1];
   assign axi_rd_reg[2] = axi_reg[2];
   assign axi_rd_reg[3] = axi_reg[3];
   assign axi_rd_reg[4] = axi_reg[4];
   assign axi_rd_reg[5] = axi_reg[5];
   assign axi_rd_reg[6] = axi_reg[6];
   assign axi_rd_reg[7] = axi_reg[7];
   assign axi_rd_reg[8] = axi_reg[8];
   assign axi_rd_reg[9] = axi_reg[9];
   assign axi_rd_reg[10] = axi_reg[10];
   assign axi_rd_reg[11] = axi_reg[11];
   assign axi_rd_reg[12] = axi_reg[12];
   assign axi_rd_reg[13] = axi_reg[13];
   assign axi_rd_reg[14] = axi_reg[14];
   assign axi_rd_reg[15] = axi_reg[15];
   assign axi_rd_reg[16] = axi_reg[16];

   wire data_mux;
   devkit_cdc_bit #(
      .ENABLE_OUTPUT_REGISTER("true")
   ) cdc_data_mux_inst (
      .src_clk_i(s_axi_aclk),
      .src_data_i(axi_reg[1][31]),
      .dest_clk_i(data_clk_i),
      .dest_data_o(data_mux)
   );

   /* Multi-bit CDC synchronization triggered when the value changes. */
   reg [S_AXI_DATA_WIDTH-1:0] base_value_axi_clk = {S_AXI_DATA_WIDTH{1'b0}};
   wire [S_AXI_DATA_WIDTH-1:0] base_value_data_clk;
   wire value_changed = (base_value_axi_clk != axi_reg[2]);
   wire value_changed_ack;

   devkit_cdc_bus #(
      .WIDTH(S_AXI_DATA_WIDTH)
   ) cdc_bus_value_inst (
      .src_rst_i(1'b0),
      .src_clk_i(s_axi_aclk),
      .dest_clk_i(data_clk_i),
      .src_valid_i(value_changed),
      .src_data_i(axi_reg[2]),
      .src_ack_o(value_changed_ack),
      .src_ready_o(),
      .dest_valid_o(),
      .dest_data_o(base_value_data_clk)
   );

   always @(posedge s_axi_aclk) begin
      if (value_changed_ack)
         base_value_axi_clk <= axi_reg[2];
   end

   /* AXI register file for the user logic 1 IP. */
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
      .slv_reg10 (axi_reg[10]),
      .slv_reg11 (axi_reg[11]),
      .slv_reg12 (axi_reg[12]),
      .slv_reg13 (axi_reg[13]),
      .slv_reg14 (axi_reg[14]),
      .slv_reg15 (axi_reg[15]),
      .slv_reg16 (axi_reg[16]),

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
      .slv_reg10_read (axi_rd_reg[10]),
      .slv_reg11_read (axi_rd_reg[11]),
      .slv_reg12_read (axi_rd_reg[12]),
      .slv_reg13_read (axi_rd_reg[13]),
      .slv_reg14_read (axi_rd_reg[14]),
      .slv_reg15_read (axi_rd_reg[15]),
      .slv_reg16_read (axi_rd_reg[16])
   );

   genvar ch;
   genvar k;
   generate
      for (ch = 0; ch < DBT1_BUS_CHANNELS; ch = ch + 1) begin: assign_my_data
         localparam INSTANCE_PARALLEL_SAMPLES =
            (ch == 0) ? DBT1_CHANNEL0_PARALLEL_SAMPLES :
            (ch == 1) ? DBT1_CHANNEL1_PARALLEL_SAMPLES :
            (ch == 2) ? DBT1_CHANNEL2_PARALLEL_SAMPLES :
            (ch == 3) ? DBT1_CHANNEL3_PARALLEL_SAMPLES :
            (ch == 4) ? DBT1_CHANNEL4_PARALLEL_SAMPLES :
            (ch == 5) ? DBT1_CHANNEL5_PARALLEL_SAMPLES :
            (ch == 6) ? DBT1_CHANNEL6_PARALLEL_SAMPLES :
            DBT1_CHANNEL7_PARALLEL_SAMPLES;

         localparam INSTANCE_BITS_PER_SAMPLE =
            (ch == 0) ? DBT1_CHANNEL0_BITS_PER_SAMPLE :
            (ch == 1) ? DBT1_CHANNEL1_BITS_PER_SAMPLE :
            (ch == 2) ? DBT1_CHANNEL2_BITS_PER_SAMPLE :
            (ch == 3) ? DBT1_CHANNEL3_BITS_PER_SAMPLE :
            (ch == 4) ? DBT1_CHANNEL4_BITS_PER_SAMPLE :
            (ch == 5) ? DBT1_CHANNEL5_BITS_PER_SAMPLE :
            (ch == 6) ? DBT1_CHANNEL6_BITS_PER_SAMPLE :
            DBT1_CHANNEL7_BITS_PER_SAMPLE;

         for (k = 0; k < INSTANCE_PARALLEL_SAMPLES; k = k + 1) begin
            always @(posedge data_clk_i) begin
               my_data[ch][k * INSTANCE_BITS_PER_SAMPLE +: INSTANCE_BITS_PER_SAMPLE] <=
                  base_value_data_clk[0 +: INSTANCE_BITS_PER_SAMPLE] + k;
            end
         end
      end

      for (ch = 0; ch < DBT1_BUS_CHANNELS; ch = ch + 1) begin: assign_out_in
         assign channel_trigger_secondary_event_out[ch] = channel_trigger_secondary_event_in[ch];
         assign channel_trigger_secondary_rising_out[ch] = channel_trigger_secondary_rising_in[ch];
         assign channel_trigger_secondary_sampleindex_fraction_out[ch] = channel_trigger_secondary_sampleindex_fraction_in[ch];
         assign channel_trigger_secondary_sampleindex_out[ch] = channel_trigger_secondary_sampleindex_in[ch];
         assign channel_trigger_primary_event_out[ch] = channel_trigger_primary_event_in[ch];
         assign channel_trigger_primary_rising_out[ch] = channel_trigger_primary_rising_in[ch];
         assign channel_trigger_primary_sampleindex_fraction_out[ch] = channel_trigger_primary_sampleindex_fraction_in[ch];
         assign channel_trigger_primary_sampleindex_out[ch] = channel_trigger_primary_sampleindex_in[ch];
         assign channel_general_purpose_out[ch] = channel_general_purpose_in[ch];
         assign channel_trigger_primary_inhibit_out[ch] = channel_trigger_primary_inhibit_in[ch];
         assign channel_overrange_out[ch] = channel_overrange_in[ch];

         /* Assign the output data as either
            - the test pattern from this module: my_data[ch]; or
            - the input data: channel_sampledata_in[ch]. */
         assign channel_sampledata_out[ch] = data_mux ? my_data[ch] : channel_sampledata_in[ch];
      end
   endgenerate

   assign common_timestamp_out = common_timestamp_in;
   assign common_timestamp_sync_count_out = common_timestamp_sync_count_in;

   /* Clocked bus extraction. */
   always @(posedge data_clk_i) begin : bus_extraction
      integer ch;

      /* Extract channel-specific bus signals. */
      for (ch = 0; ch < DBT1_BUS_CHANNELS; ch = ch + 1) begin
         channel_trigger_secondary_event_in[ch] <= extract_channel_trigger_secondary_event(ch);
         channel_trigger_secondary_rising_in[ch] <= extract_channel_trigger_secondary_rising(ch);
         channel_trigger_secondary_sampleindex_fraction_in[ch] <= extract_channel_trigger_secondary_sampleindex_fraction(ch);
         channel_trigger_secondary_sampleindex_in[ch] <= extract_channel_trigger_secondary_sampleindex(ch);
         channel_trigger_primary_event_in[ch] <= extract_channel_trigger_primary_event(ch);
         channel_trigger_primary_rising_in[ch] <= extract_channel_trigger_primary_rising(ch);
         channel_trigger_primary_sampleindex_fraction_in[ch] <= extract_channel_trigger_primary_sampleindex_fraction(ch);
         channel_trigger_primary_sampleindex_in[ch] <= extract_channel_trigger_primary_sampleindex(ch);
         channel_general_purpose_in[ch] <= extract_channel_general_purpose(ch);
         channel_trigger_primary_inhibit_in[ch] <= extract_channel_trigger_primary_inhibit(ch);
         channel_overrange_in[ch] <= extract_channel_overrange(ch);
         channel_sampledata_in[ch] <= extract_channel_sampledata(ch);
      end

      /* Extract bus signals common to all channels. */
      common_timestamp_in <= extract_common_timestamp(0);
      common_timestamp_sync_count_in <= extract_common_timestamp_sync_count(0);
   end

   /* Asynchronous bus insertion. */
   always @(*) begin : bus_insertion
      integer ch;

      /* _MUST_ be called first. */
      init_bus_output(bus_output_defaults);

      /* Insert channel-specific bus signals. */
      for (ch = 0; ch < DBT1_BUS_CHANNELS; ch = ch + 1) begin
         insert_channel_trigger_secondary_event(channel_trigger_secondary_event_out[ch], ch);
         insert_channel_trigger_secondary_rising(channel_trigger_secondary_rising_out[ch], ch);
         insert_channel_trigger_secondary_sampleindex_fraction(channel_trigger_secondary_sampleindex_fraction_out[ch], ch);
         insert_channel_trigger_secondary_sampleindex(channel_trigger_secondary_sampleindex_out[ch], ch);
         insert_channel_trigger_primary_event(channel_trigger_primary_event_out[ch], ch);
         insert_channel_trigger_primary_rising(channel_trigger_primary_rising_out[ch], ch);
         insert_channel_trigger_primary_sampleindex_fraction(channel_trigger_primary_sampleindex_fraction_out[ch], ch);
         insert_channel_trigger_primary_sampleindex(channel_trigger_primary_sampleindex_out[ch], ch);
         insert_channel_general_purpose(channel_general_purpose_out[ch], ch);
         insert_channel_trigger_primary_inhibit(channel_trigger_primary_inhibit_out[ch], ch);
         insert_channel_overrange(channel_overrange_out[ch], ch);
         insert_channel_sampledata(channel_sampledata_out[ch], ch);
      end

      /* Insert bus signals common to all channels. */
      insert_common_timestamp(common_timestamp_out);
      insert_common_timestamp_sync_count(common_timestamp_sync_count_out);

      /* _MUST_ be called last. */
      finish_bus_output();
   end

endmodule
`default_nettype wire
