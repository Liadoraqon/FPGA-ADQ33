#!/usr/bin/env python3
# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
"""
 Example illustrating the data acquisition for a ADQ3 series digitizer. The
 example will list the available devices and acquire data from the first
 device.
"""

import math
import pyadq
from typing import List

# Record length in samples
RECORD_LENGTH = 1024
# Number of records to collect per channel
NOF_RECORDS = 10
# Periodic event generator frequency in Hz
PERIODIC_EVENT_GENERATOR_FREQUENCY = 10e3

# If the digitizer is running FWATD firmware, the following number of
# accumulations will be used
NOF_ACCUMULATIONS = 10

# Create the control unit
acu: pyadq.ADQControlUnit = pyadq.ADQControlUnit()

# Enable trace logging
acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, ".")

# List the available devices
device_list: List[pyadq.ADQInfoListEntry] = acu.ListDevices()

print(f"Found {len(device_list)} device(s)")

# Set up the first supported device
device_to_open = -1
for i, dl in enumerate(device_list):
    if dl.ProductID in [pyadq.PID_ADQ30, pyadq.PID_ADQ32, pyadq.PID_ADQ35, pyadq.PID_ADQ36]:
        device_to_open = i
        break

# Ensure that at least one device is available
assert device_to_open != -1, "No supported device found"

dev: pyadq.ADQ = acu.SetupDevice(device_to_open)

print(f"Setting up data collection for: {dev}")

# Initialize the parameters with default values
parameters: pyadq.ADQParameters = dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)

# Set up data collection for all channels
for ch in range(parameters.constant.nof_channels):
    parameters.acquisition.channel[ch].record_length = RECORD_LENGTH
    parameters.acquisition.channel[ch].nof_records = pyadq.ADQ_INFINITE_NOF_RECORDS
    parameters.acquisition.channel[ch].trigger_edge = pyadq.ADQ_EDGE_RISING
    parameters.acquisition.channel[ch].trigger_source = pyadq.ADQ_EVENT_SOURCE_PERIODIC

# Configure data transfer parameters for all channels
for ch in range(parameters.constant.nof_channels):
    record_buffer_size = RECORD_LENGTH * parameters.constant.channel[ch].nof_bytes_per_sample

    # Ceil the record buffer size to the nearest multiple of
    # record_buffer_size_step
    record_buffer_size = (
        math.ceil(record_buffer_size / parameters.constant.record_buffer_size_step)
        * parameters.constant.record_buffer_size_step
    )

    # Limit the record_buffer_size to approximately 1 MiB. Some systems may have
    # trouble allocating enough *contiguous* memory for large buffer sizes.
    MAX_RECORD_BUFFER_SIZE = 512 * 1024
    if record_buffer_size > MAX_RECORD_BUFFER_SIZE:
        record_buffer_size = (
            math.ceil(MAX_RECORD_BUFFER_SIZE / parameters.constant.record_buffer_size_step)
            * parameters.constant.record_buffer_size_step
        )

    parameters.transfer.channel[ch].record_size = 0
    parameters.transfer.channel[ch].infinite_record_length_enabled = 0
    parameters.transfer.channel[ch].record_buffer_size = record_buffer_size
    parameters.transfer.channel[ch].dynamic_record_length_enabled = 1
    # Enable metadata (record headers)
    parameters.transfer.channel[ch].metadata_enabled = 1
    parameters.transfer.channel[ch].metadata_buffer_size = pyadq.SIZEOF_ADQ_GEN4_HEADER
    parameters.transfer.channel[ch].nof_buffers = pyadq.ADQ_MAX_NOF_BUFFERS

    # Eject the transfer buffer on record stop. This may limit the maximum
    # throughput. Refer to the user guide for more information.
    parameters.transfer.channel[ch].eject_buffer_source = pyadq.ADQ_FUNCTION_RECORD_STOP

# Configure the periodic event generator
parameters.event_source.periodic.frequency = PERIODIC_EVENT_GENERATOR_FREQUENCY

# Configure the ATD signal processing module, if digitizer is running FWATD firmware
if parameters.constant.firmware.type == pyadq.ADQ_FIRMWARE_TYPE_FWATD:
    parameters.signal_processing.atd.common.nof_accumulations = NOF_ACCUMULATIONS

# Set parameters
dev.SetParameters(parameters)

# Start the data acquisition
print("Starting data acquisition")
result = dev.ADQ_StartDataAcquisition()
if result != pyadq.ADQ_EOK:
    raise Exception(f"ADQ_StartDataAcquisition failed with error code {result}. See log file.")

record_count = 0
records_to_collect = NOF_RECORDS * parameters.constant.nof_channels
first_timestamp = [0] * pyadq.ADQ_MAX_NOF_CHANNELS
try:
    while record_count < records_to_collect:
        # Wait for a record buffer on any channel with 1000 ms timeout
        record = dev.WaitForRecordBuffer(pyadq.ADQ_ANY_CHANNEL, 1000)

        # Use the first timestamp as the reference
        if not first_timestamp[record.header.channel]:
            first_timestamp[record.header.channel] = record.header.timestamp

        timestamp_diff = record.header.timestamp - first_timestamp[record.header.channel]
        timestamp_us = timestamp_diff * 1e6 * record.header.time_unit

        # Print some header information
        print(
            f"Channel {record.header.channel} Record {record.header.record_number} "
            f"w/ {record.header.record_length} samples @ {timestamp_us:.3f} us"
        )
        print(f"\tData: {record.data}")

        record_count += 1

except Exception as e:
    dev.ADQ_StopDataAcquisition()
    raise e

# Stop the data acquisition
print("Stopping data acquisition")
result = dev.ADQ_StopDataAcquisition()
if result not in [pyadq.ADQ_EOK, pyadq.ADQ_EINTERRUPTED]:
    raise Exception(f"ADQ_StartDataAcquisition failed with error code {result}. See log file.")
