# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
import pyadq

# Trigger mode for each channel. Valid values are:
# - pyadq.ADQ_EVENT_SOURCE_TRIG
# - pyadq.ADQ_EVENT_SOURCE_LEVEL
# - pyadq.ADQ_EVENT_SOURCE_SYNC
# - pyadq.ADQ_EVENT_SOURCE_PERIODIC
TRIGGER_SOURCE = [
    pyadq.ADQ_EVENT_SOURCE_PERIODIC,
    pyadq.ADQ_EVENT_SOURCE_PERIODIC,
    pyadq.ADQ_EVENT_SOURCE_PERIODIC,
    pyadq.ADQ_EVENT_SOURCE_PERIODIC,
]

# Trigger edge for each channel. The support for these values varies between the event sources.
# Refer to the user guide for a detailed explanation.
TRIGGER_EDGE = [
    pyadq.ADQ_EDGE_RISING,
    pyadq.ADQ_EDGE_RISING,
    pyadq.ADQ_EDGE_RISING,
    pyadq.ADQ_EDGE_RISING,
]

# Trigger period and frequency. If PERIODIC_EVENT_SOURCE_PERIOD is 0,
# PERIODIC_EVENT_SOURCE_FREQUENCY will be used.
# Otherwise PERIODIC_EVENT_SOURCE_PERIOD will be used.
PERIODIC_EVENT_SOURCE_PERIOD = 2800
PERIODIC_EVENT_SOURCE_FREQUENCY = 0

# Horizontal offset of the trigger point in samples for each channel.
HORIZONTAL_OFFSET = [0, 0, 0, 0]

# Number of channels used for acquisition.
# If this value is lower than the number of available channels, the ones with
# higher indexes are disabled.
NOF_ACTIVE_CHANNELS = 1

# Record length for each channel.
RECORD_LEN = [2048, 2048, 2048, 2048]

# Specifies the size of the buffer in number of records.
NOF_RECORDS_PER_BUFFER = 1000

# Specifies how many transfer buffers that will be allocated per channel.
NOF_GPU_BUFFERS = 2

# Number of buffers to be received for each channel before the acquisition stops.
NOF_BUFFERS_TO_RECEIVE = 10000

# Time in ms to wait in WaitForP2pBuffers(...) before timing out.
WAIT_TIMEOUT_MS = 1000

# Sample skip for each channel.
SAMPLE_SKIP_FACTOR = [1, 1, 1, 1]

# DC offset in mV for each channel.
DC_OFFSET = [0, 0, 0, 0]

# If test pattern is used, this specifies the pattern for each channel.
# Set to ADQ_TEST_PATTERN_SOURCE_DISABLE to disable the test pattern.
# Available test pattern signals are:
#  - pyadq.ADQ_TEST_PATTERN_SOURCE_DISABLE
#  - pyadq.ADQ_TEST_PATTERN_SOURCE_COUNT_UP
#  - pyadq.ADQ_TEST_PATTERN_SOURCE_COUNT_DOWN
#  - pyadq.ADQ_TEST_PATTERN_SOURCE_TRIANGLE
TEST_PATTERN_SOURCE = [
    pyadq.ADQ_TEST_PATTERN_SOURCE_TRIANGLE,
    pyadq.ADQ_TEST_PATTERN_SOURCE_TRIANGLE,
    pyadq.ADQ_TEST_PATTERN_SOURCE_TRIANGLE,
    pyadq.ADQ_TEST_PATTERN_SOURCE_TRIANGLE,
]

BYTES_PER_SAMPLE = 2

# Set to 1 to print and save an image of a single buffer at the end for verification.
PRINT_SINGLE_BUFFER = 1
