/*
 * Copyright 2021 Teledyne Signal Processing Devices Sweden AB
 */
`default_nettype none
`timescale 1ns / 1ps

`include "databus_type3_parameters.vh"
`include "config.vh"

module user_logic2 #(
   parameter integer S_AXI_DATA_WIDTH = 32,
   parameter integer S_AXI_ADDR_WIDTH = 16,
   parameter integer S_DATABUS_WIDTH = -1,
   parameter integer NOF_EXT_GPIOA_IN_BITS = -1,
   parameter integer NOF_EXT_GPIOA_OUT_BITS = -1,
   parameter integer NOF_EXT_GPIOA_DIRECTION_BITS = -1,
   parameter integer NOF_EXT_GPIOB_IN_BITS = -1,
   parameter integer NOF_EXT_GPIOB_OUT_BITS = -1,
   parameter integer NOF_EXT_GPIOB_DIRECTION_BITS = -1,
   parameter integer NOF_EXT_GPIOC_IN_BITS = -1,
   parameter integer NOF_EXT_GPIOC_OUT_BITS = -1,
   parameter integer NOF_EXT_GPIOC_DIRECTION_BITS = -1,
   parameter integer M_DATABUS_WIDTH = -1
)(
   input  wire data_clk_i,
   input  wire data_rst_i,
   input  wire [S_DATABUS_WIDTH-1:0] s_databus_type3_payload,
   output reg  [M_DATABUS_WIDTH-1:0] m_databus_type3_payload,

   /* NOTE: All ext_ signals are synchronous to data_clk_i */
   input  wire ext_trig_i,
   output wire ext_trig_o,
   output wire ext_trig_direction_o,

   input  wire ext_sync_i,
   output wire ext_sync_o,
   output wire ext_sync_direction_o,

`ifdef HAS_EXT_GPIOA
   input  wire [NOF_EXT_GPIOA_IN_BITS-1:0]        ext_gpioa_i,
   output wire [NOF_EXT_GPIOA_OUT_BITS-1:0]       ext_gpioa_o,
   output wire [NOF_EXT_GPIOA_DIRECTION_BITS-1:0] ext_gpioa_direction_o,
`endif

`ifdef HAS_EXT_GPIOB
   input  wire [NOF_EXT_GPIOB_IN_BITS-1:0]        ext_gpiob_i,
   output wire [NOF_EXT_GPIOB_OUT_BITS-1:0]       ext_gpiob_o,
   output wire [NOF_EXT_GPIOB_DIRECTION_BITS-1:0] ext_gpiob_direction_o,
`endif

`ifdef HAS_EXT_GPIOC
   input  wire [NOF_EXT_GPIOC_IN_BITS-1:0]        ext_gpioc_i,
   output wire [NOF_EXT_GPIOC_OUT_BITS-1:0]       ext_gpioc_o,
   output wire [NOF_EXT_GPIOC_DIRECTION_BITS-1:0] ext_gpioc_direction_o,
`endif

`ifdef HAS_EXT_PXIE_TRIG
   input  wire ext_starb_i,
   output wire ext_starc_o,
`endif

   output wire led_user_n_o,

   // AXI-Lite
   input  wire                        s_axi_aclk,
   input  wire                        s_axi_aresetn,
   input  wire [S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
   input  wire [2:0]                  s_axi_awprot,
   input  wire                        s_axi_awvalid,
   output wire                        s_axi_awready,
   input  wire [S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
   input  wire [(S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
   input  wire                        s_axi_wvalid,
   output wire                        s_axi_wready,
   output wire [1:0]                  s_axi_bresp,
   output wire                        s_axi_bvalid,
   input  wire                        s_axi_bready,
   input  wire [S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
   input  wire [2:0]                  s_axi_arprot,
   input  wire                        s_axi_arvalid,
   output wire                        s_axi_arready,
   output wire [S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
   output wire [1:0]                  s_axi_rresp,
   output wire                        s_axi_rvalid,
   input  wire                        s_axi_rready
);

  `include "dbt3_param.vh"
  `include "databus_type3_functions.vh"

   /* BUS_PIPELINE = computational latency through this block. */
   localparam BUS_PIPELINE = 1;

   reg  [DBT3_CHANNEL_DATA_MAXWIDTH-1:0] my_data[DBT3_BUS_CHANNELS-1:0];

   reg  [DBT3_CHANNEL_RECORD_NUMBER_WIDTH-1:0]              channel_record_number_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_RECORD_STOP_WIDTH-1:0]                channel_record_stop_in  [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_RECORD_STOP_INDEX_WIDTH-1:0]          channel_record_stop_index_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_RECORD_START_WIDTH-1:0]               channel_record_start_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_RECORD_START_INDEX_WIDTH-1:0]         channel_record_start_index_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TRIGGER_SECONDARY_EVENT_WIDTH-1:0]    channel_trigger_secondary_event_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TRIGGER_SECONDARY_RISING_WIDTH-1:0]   channel_trigger_secondary_rising_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TRIGGER_SECONDARY_SAMPLEINDEX_FRACTION_WIDTH-1:0] channel_trigger_secondary_sampleindex_fraction_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TRIGGER_SECONDARY_SAMPLEINDEX_WIDTH-1:0] channel_trigger_secondary_sampleindex_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TRIGGER_PRIMARY_EVENT_WIDTH-1:0]      channel_trigger_primary_event_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TRIGGER_PRIMARY_RISING_WIDTH-1:0]     channel_trigger_primary_rising_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TRIGGER_PRIMARY_SAMPLEINDEX_FRACTION_WIDTH-1:0] channel_trigger_primary_sampleindex_fraction_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TRIGGER_PRIMARY_SAMPLEINDEX_WIDTH-1:0] channel_trigger_primary_sampleindex_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_VALID_WIDTH-1:0]                      channel_valid_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_GENERAL_PURPOSE_WIDTH-1:0]            channel_general_purpose_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TRIGGER_PRIMARY_INHIBIT_WIDTH-1:0]    channel_trigger_primary_inhibit_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TIMESTAMP_WIDTH-1:0]                  channel_timestamp_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_TIMESTAMP_SYNC_COUNT_WIDTH-1:0]       channel_timestamp_sync_count_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_OVERRANGE_WIDTH-1:0]                  channel_overrange_in [DBT3_BUS_CHANNELS-1:0];
   reg  [DBT3_CHANNEL_DATA_MAXWIDTH-1:0]                    channel_data_in [DBT3_BUS_CHANNELS-1:0];

   wire [DBT3_CHANNEL_RECORD_NUMBER_WIDTH-1:0]              channel_record_number_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_RECORD_STOP_WIDTH-1:0]                channel_record_stop_out  [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_RECORD_STOP_INDEX_WIDTH-1:0]          channel_record_stop_index_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_RECORD_START_WIDTH-1:0]               channel_record_start_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_RECORD_START_INDEX_WIDTH-1:0]         channel_record_start_index_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TRIGGER_SECONDARY_EVENT_WIDTH-1:0]    channel_trigger_secondary_event_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TRIGGER_SECONDARY_RISING_WIDTH-1:0]   channel_trigger_secondary_rising_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TRIGGER_SECONDARY_SAMPLEINDEX_FRACTION_WIDTH-1:0] channel_trigger_secondary_sampleindex_fraction_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TRIGGER_SECONDARY_SAMPLEINDEX_WIDTH-1:0] channel_trigger_secondary_sampleindex_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TRIGGER_PRIMARY_EVENT_WIDTH-1:0]      channel_trigger_primary_event_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TRIGGER_PRIMARY_RISING_WIDTH-1:0]     channel_trigger_primary_rising_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TRIGGER_PRIMARY_SAMPLEINDEX_FRACTION_WIDTH-1:0] channel_trigger_primary_sampleindex_fraction_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TRIGGER_PRIMARY_SAMPLEINDEX_WIDTH-1:0] channel_trigger_primary_sampleindex_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_VALID_WIDTH-1:0]                      channel_valid_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_USER_ID_WIDTH-1:0]                    channel_user_id_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_GENERAL_PURPOSE_WIDTH-1:0]            channel_general_purpose_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TRIGGER_PRIMARY_INHIBIT_WIDTH-1:0]    channel_trigger_primary_inhibit_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TIMESTAMP_WIDTH-1:0]                  channel_timestamp_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_TIMESTAMP_SYNC_COUNT_WIDTH-1:0]       channel_timestamp_sync_count_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_OVERRANGE_WIDTH-1:0]                  channel_overrange_out [DBT3_BUS_CHANNELS-1:0];
   wire [DBT3_CHANNEL_DATA_MAXWIDTH-1:0]                    channel_data_out [DBT3_BUS_CHANNELS-1:0];

   wire [S_AXI_DATA_WIDTH-1:0] axi_reg[16:0];
   wire [S_AXI_DATA_WIDTH-1:0] axi_rd_reg[16:0];

   /* -------- CH1 POC sources (16-bit lanes) -------- */
   reg  [63:0] rec_cnt_ch1 = 64'd0;
   wire [63:0] const64     = 64'hCAFACAFA_F1FA2025; // example constant

   always @(posedge data_clk_i) begin
      if (data_rst_i)
         rec_cnt_ch1 <= 64'd0;
      else if (channel_record_start_in[1])
         rec_cnt_ch1 <= rec_cnt_ch1 + 1'b1;
   end

   // Pack into 8×16-bit words: [0..3] = const, [4..7] = counter (little-end words)
   wire [DBT3_CHANNEL1_NOF_WORDS*DBT3_CHANNEL1_WORD_SIZE-1:0] ch1_packet;
   assign ch1_packet[0*16 +: 16] = const64[15:0];
   assign ch1_packet[1*16 +: 16] = const64[31:16];
   assign ch1_packet[2*16 +: 16] = const64[47:32];
   assign ch1_packet[3*16 +: 16] = const64[63:48];
   assign ch1_packet[4*16 +: 16] = rec_cnt_ch1[15:0];
   assign ch1_packet[5*16 +: 16] = rec_cnt_ch1[31:16];
   assign ch1_packet[6*16 +: 16] = rec_cnt_ch1[47:32];
   assign ch1_packet[7*16 +: 16] = rec_cnt_ch1[63:48];

   // One-beat valid pulse per record for CH1
   reg ch1_valid_q = 1'b0;
   always @(posedge data_clk_i) begin
      if (data_rst_i) ch1_valid_q <= 1'b0;
      else            ch1_valid_q <= channel_record_start_in[1];
   end
   /* ------------------------------------------------- */

   assign axi_rd_reg[0]  = 32'h12345678;
   assign axi_rd_reg[1]  = axi_reg[1];
   assign axi_rd_reg[2]  = axi_reg[2];
   assign axi_rd_reg[3]  = axi_reg[3];
   assign axi_rd_reg[4]  = axi_reg[4];
   assign axi_rd_reg[5]  = axi_reg[5];
   assign axi_rd_reg[6]  = axi_reg[6];
   assign axi_rd_reg[7]  = axi_reg[7];
   assign axi_rd_reg[8]  = axi_reg[8];
   assign axi_rd_reg[9]  = axi_reg[9];
   assign axi_rd_reg[10] = axi_reg[10];
   assign axi_rd_reg[11] = axi_reg[11];
   assign axi_rd_reg[12] = axi_reg[12];
   assign axi_rd_reg[13] = axi_reg[13];
   assign axi_rd_reg[14] = axi_reg[14];
   assign axi_rd_reg[15] = axi_reg[15];

   wire data_mux;
   devkit_cdc_bit #(
      .ENABLE_OUTPUT_REGISTER("true")
   ) cdc_data_mux_inst (
      .src_clk_i(s_axi_aclk),
      .src_data_i(axi_reg[1][31]),
      .dest_clk_i(data_clk_i),
      .dest_data_o(data_mux)
   );

   /* Example CDC for a 32-bit value */
   reg  [S_AXI_DATA_WIDTH-1:0] base_value_axi_clk = {S_AXI_DATA_WIDTH{1'b0}};
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

   /* AXI register file */
   user_logic2_s_axi #(
      .C_S_AXI_DATA_WIDTH(S_AXI_DATA_WIDTH),
      .C_S_AXI_ADDR_WIDTH(S_AXI_ADDR_WIDTH)
   ) user_logic2_s_axi_inst (
      .S_AXI_ACLK (s_axi_aclk),
      .S_AXI_ARESETN (s_axi_aresetn),
      .S_AXI_AWADDR (s_axi_awaddr),
      .S_AXI_AWPROT (s_axi_awprot),
      .S_AXI_AWVALID (s_axi_awvalid),
      .S_AXI_AWREADY (s_axi_awready),
      .S_AXI_WDATA (s_axi_wdata),
      .S_AXI_WSTRB (s_axi_wstrb),
      .S_AXI_WVALID (s_axi_wvalid),
      .S_AXI_WREADY (s_axi_wready),
      .S_AXI_BRESP (s_axi_bresp),
      .S_AXI_BVALID (s_axi_bvalid),
      .S_AXI_BREADY (s_axi_bready),
      .S_AXI_ARADDR (s_axi_araddr),
      .S_AXI_ARPROT (s_axi_arprot),
      .S_AXI_ARVALID (s_axi_arvalid),
      .S_AXI_ARREADY (s_axi_arready),
      .S_AXI_RDATA (s_axi_rdata),
      .S_AXI_RRESP (s_axi_rresp),
      .S_AXI_RVALID (s_axi_rvalid),
      .S_AXI_RREADY (s_axi_rready),

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

   /* Front panel USER LED. Active low. */
   assign led_user_n_o = 1'b1;

`ifdef HAS_EXT_GPIOA
   assign ext_gpioa_o = {NOF_EXT_GPIOA_OUT_BITS{1'b0}};
   assign ext_gpioa_direction_o = {NOF_EXT_GPIOA_DIRECTION_BITS{1'b0}};
`endif

`ifdef HAS_EXT_GPIOB
   assign ext_gpiob_o = {NOF_EXT_GPIOB_OUT_BITS{1'b0}};
   assign ext_gpiob_direction_o = {NOF_EXT_GPIOB_DIRECTION_BITS{1'b0}};
`endif

`ifdef HAS_EXT_GPIOC
   assign ext_gpioc_o = {NOF_EXT_GPIOC_OUT_BITS{1'b0}};
   assign ext_gpioc_direction_o = {NOF_EXT_GPIOC_DIRECTION_BITS{1'b0}};
`endif

`ifdef HAS_EXT_PXIE_TRIG
   assign ext_starc_o = 1'b0;
`endif

   assign ext_trig_o = 1'b0;
   assign ext_trig_direction_o = 1'b0;

   assign ext_sync_o = 1'b0;
   assign ext_sync_direction_o = 1'b0;

   /* Test-pattern generator for channels (safe at 16-bit) */
   genvar ch, k;
   generate
      for (ch = 0; ch < DBT3_BUS_CHANNELS; ch = ch + 1) begin: assign_my_data
         localparam INSTANCE_PARALLEL_SAMPLES =
            (ch == 0) ? DBT3_CHANNEL0_NOF_WORDS :
            (ch == 1) ? DBT3_CHANNEL1_NOF_WORDS :
                        DBT3_CHANNEL7_NOF_WORDS; // unused beyond CH1

         localparam INSTANCE_BITS_PER_SAMPLE =
            (ch == 0) ? DBT3_CHANNEL0_WORD_SIZE :
            (ch == 1) ? DBT3_CHANNEL1_WORD_SIZE :
                        DBT3_CHANNEL7_WORD_SIZE; // unused beyond CH1

         for (k = 0; k < INSTANCE_PARALLEL_SAMPLES; k = k + 1) begin
            always @(posedge data_clk_i) begin
               my_data[ch][k*INSTANCE_BITS_PER_SAMPLE +: INSTANCE_BITS_PER_SAMPLE]
                 <= base_value_data_clk[0 +: INSTANCE_BITS_PER_SAMPLE] + k;
            end
         end
      end
   endgenerate

   /* Per-channel outputs: pass-through with CH1 override */
   generate
      for (ch = 0; ch < DBT3_BUS_CHANNELS; ch = ch + 1) begin: assign_out_in
         // Control/meta passthrough
         assign channel_record_number_out[ch]              = channel_record_number_in[ch];
         assign channel_record_stop_out[ch]                = channel_record_stop_in[ch];
         assign channel_record_stop_index_out[ch]          = channel_record_stop_index_in[ch];
         assign channel_record_start_out[ch]               = channel_record_start_in[ch];
         assign channel_record_start_index_out[ch]         = channel_record_start_index_in[ch];
         assign channel_trigger_secondary_event_out[ch]    = channel_trigger_secondary_event_in[ch];
         assign channel_trigger_secondary_rising_out[ch]   = channel_trigger_secondary_rising_in[ch];
         assign channel_trigger_secondary_sampleindex_fraction_out[ch] = channel_trigger_secondary_sampleindex_fraction_in[ch];
         assign channel_trigger_secondary_sampleindex_out[ch]          = channel_trigger_secondary_sampleindex_in[ch];
         assign channel_trigger_primary_event_out[ch]      = channel_trigger_primary_event_in[ch];
         assign channel_trigger_primary_rising_out[ch]     = channel_trigger_primary_rising_in[ch];
         assign channel_trigger_primary_sampleindex_fraction_out[ch]   = channel_trigger_primary_sampleindex_fraction_in[ch];
         assign channel_trigger_primary_sampleindex_out[ch]            = channel_trigger_primary_sampleindex_in[ch];
         assign channel_general_purpose_out[ch]            = channel_general_purpose_in[ch];
         assign channel_trigger_primary_inhibit_out[ch]    = channel_trigger_primary_inhibit_in[ch];
         assign channel_timestamp_out[ch]                  = channel_timestamp_in[ch];
         assign channel_timestamp_sync_count_out[ch]       = channel_timestamp_sync_count_in[ch];
         assign channel_overrange_out[ch]                  = channel_overrange_in[ch];

         if (ch == 1) begin : ch1_override
            assign channel_user_id_out[ch] = 8'hC1;        // recognizable tag for CH1 in headers
            assign channel_valid_out[ch]   = channel_valid_in[0];  // multiple beat per record
            assign channel_data_out[ch]    = ch1_packet;   // our 16b×8 payload
         end else begin : ch0_default
            assign channel_user_id_out[ch] = ch[DBT3_CHANNEL_USER_ID_WIDTH-1:0];
            assign channel_valid_out[ch]   = channel_valid_in[ch];
            assign channel_data_out[ch]    = data_mux ? my_data[ch] : channel_data_in[ch];
         end
      end
   endgenerate

   /* Clocked bus extraction. */
   always @(posedge data_clk_i) begin : bus_extraction
      integer i;
      for (i = 0; i < DBT3_BUS_CHANNELS; i = i + 1) begin
         channel_record_number_in[i]                  <= extract_channel_record_number(i);
         channel_record_stop_in[i]                    <= extract_channel_record_stop(i);
         channel_record_stop_index_in[i]              <= extract_channel_record_stop_index(i);
         channel_record_start_in[i]                   <= extract_channel_record_start(i);
         channel_record_start_index_in[i]             <= extract_channel_record_start_index(i);
         channel_trigger_secondary_event_in[i]        <= extract_channel_trigger_secondary_event(i);
         channel_trigger_secondary_rising_in[i]       <= extract_channel_trigger_secondary_rising(i);
         channel_trigger_secondary_sampleindex_fraction_in[i] <= extract_channel_trigger_secondary_sampleindex_fraction(i);
         channel_trigger_secondary_sampleindex_in[i]  <= extract_channel_trigger_secondary_sampleindex(i);
         channel_trigger_primary_event_in[i]          <= extract_channel_trigger_primary_event(i);
         channel_trigger_primary_rising_in[i]         <= extract_channel_trigger_primary_rising(i);
         channel_trigger_primary_sampleindex_fraction_in[i] <= extract_channel_trigger_primary_sampleindex_fraction(i);
         channel_trigger_primary_sampleindex_in[i]    <= extract_channel_trigger_primary_sampleindex(i);
         channel_valid_in[i]                          <= extract_channel_valid(i);
         channel_general_purpose_in[i]                <= extract_channel_general_purpose(i);
         channel_trigger_primary_inhibit_in[i]        <= extract_channel_trigger_primary_inhibit(i);
         channel_timestamp_in[i]                      <= extract_channel_timestamp(i);
         channel_timestamp_sync_count_in[i]           <= extract_channel_timestamp_sync_count(i);
         channel_overrange_in[i]                      <= extract_channel_overrange(i);
         channel_data_in[i]                           <= extract_channel_data(i);
      end
   end

   /* Asynchronous bus insertion. */
   always @(*) begin : bus_insertion
      integer j;
      init_bus_output(bus_output_defaults);
      for (j = 0; j < DBT3_BUS_CHANNELS; j = j + 1) begin
         insert_channel_record_number(               channel_record_number_out[j],               j);
         insert_channel_record_stop(                 channel_record_stop_out[j],                 j);
         insert_channel_record_stop_index(           channel_record_stop_index_out[j],           j);
         insert_channel_record_start(                channel_record_start_out[j],                j);
         insert_channel_record_start_index(          channel_record_start_index_out[j],          j);
         insert_channel_trigger_secondary_event(     channel_trigger_secondary_event_out[j],     j);
         insert_channel_trigger_secondary_rising(    channel_trigger_secondary_rising_out[j],    j);
         insert_channel_trigger_secondary_sampleindex_fraction(channel_trigger_secondary_sampleindex_fraction_out[j], j);
         insert_channel_trigger_secondary_sampleindex(channel_trigger_secondary_sampleindex_out[j], j);
         insert_channel_trigger_primary_event(       channel_trigger_primary_event_out[j],       j);
         insert_channel_trigger_primary_rising(      channel_trigger_primary_rising_out[j],      j);
         insert_channel_trigger_primary_sampleindex_fraction(channel_trigger_primary_sampleindex_fraction_out[j], j);
         insert_channel_trigger_primary_sampleindex( channel_trigger_primary_sampleindex_out[j], j);
         insert_channel_valid(                       channel_valid_out[j],                       j);
         insert_channel_user_id(                     channel_user_id_out[j],                     j);
         insert_channel_general_purpose(             channel_general_purpose_out[j],             j);
         insert_channel_trigger_primary_inhibit(     channel_trigger_primary_inhibit_out[j],     j);
         insert_channel_timestamp(                   channel_timestamp_out[j],                   j);
         insert_channel_timestamp_sync_count(        channel_timestamp_sync_count_out[j],        j);
         insert_channel_overrange(                   channel_overrange_out[j],                   j);
         insert_channel_data(                        channel_data_out[j],                        j);
      end
      finish_bus_output();
   end

endmodule
`default_nettype wire
