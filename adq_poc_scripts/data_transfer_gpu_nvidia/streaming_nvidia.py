# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
""" P2P streaming ADQ -> GPU in python """
import pyadq
from typing import List, Tuple
import cupy as cp
import settings as s
import ctypes as ct
import streaming_helpers as sh
import helper_cupy as hc
import gdrapi as g
from gdrapi import gdr_check_error_exit_func
import time
import numpy as np
import matplotlib.pyplot as plt
import os


gdrapi = g.GdrApi()


def allocate_and_pin_buffer(
    buffer_size: int,
    memory_handle: g.GdrMemoryHandle,
    gdr: g.Gdr,
    bar_ptr_data: sh.BarPtrData,
) -> Tuple[ct.c_void_p, ct.c_uint64, cp.ndarray]:
    """Allocate and pin buffers.

    Args:
        `buffer_size`: Size to allocate on GPU.
        `memory_handle`: Wrapped memory_handle struct from gdrapi.
        `gdr`: Wrapped gdr object from gdrapi.
        `bar_ptr_data`: Pointer to data on bar.

    Returns:
        `buffer_pointer`: Pointer to GPU buffer.
        `buffer_address`: Physical address to buffer.
        `buffer`: Buffer object.
    """
    info = g.GdrInfo()

    # GPU page size rounding.
    buffer_size = (buffer_size + g.GPU_PAGE_SIZE - 1) & g.GPU_PAGE_MASK
    buffer = cp.zeros(buffer_size // hc.sizeof(cp.short), dtype=cp.short)  # Allocate memory in GPU
    buffer_ptr = buffer.data.ptr  # Pointer to memory

    # Map device memory buffer on GPU BAR1, returning a handle.
    gdr_status = gdrapi.gdr_pin_buffer(
        gdr, ct.c_ulong(buffer_ptr), buffer_size, 0, 0, ct.byref(memory_handle)
    )
    gdr_check_error_exit_func(gdr_status or (memory_handle == 0), "gdr_pin_buffer")
    # Create a user-space mapping for the BAR1 info, length is bar1->buffer
    gdr_status = gdrapi.gdr_map(gdr, memory_handle, ct.byref(bar_ptr_data), buffer_size)
    gdr_check_error_exit_func(gdr_status or (bar_ptr_data == 0), "gdr_map")
    # Bar physical address will be aligned to the page size before being mapped in user-space
    # so the pointer returned might be affected by an offset.
    # gdr_get_info is used to calculate offset.

    gdr_status = gdrapi.gdr_get_info(gdr, memory_handle, info)

    gdr_check_error_exit_func(gdr_status, "gdr_info")
    offset_data = info.va - buffer_ptr

    buffer_address = ct.c_uint64(info.physical)
    gdr_status = gdrapi.gdr_validate_phybar(gdr, memory_handle)
    gdr_check_error_exit_func(gdr_status, "gdr_validate_phybar")
    buffer_pointer = ct.c_void_p(bar_ptr_data.value + offset_data)

    return buffer_pointer, buffer_address, buffer


def main():
    """Main streaming function"""
    if os.name == "nt":
        exit("This example is not compatible with Windows.")
    acu: pyadq.ADQControlUnit = pyadq.ADQControlUnit()
    # Enable trace logging
    acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, ".")

    # List the available devices
    device_list: List[pyadq.ADQInfoListEntry] = acu.ListDevices()

    print(f"Found {len(device_list)} device(s)")

    # Ensure that at least one device is available
    assert device_list

    # Set up the first available device
    device_to_open = 0
    dev: pyadq.ADQ = acu.SetupDevice(device_to_open)

    print(f"Setting up data collection for: {dev}")

    # Initialize the parameters with default values
    parameters: pyadq.ADQParameters = dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)

    # Setting internal trigger parameters
    parameters.event_source.periodic.period = s.PERIODIC_EVENT_SOURCE_PERIOD
    parameters.event_source.periodic.frequency = s.PERIODIC_EVENT_SOURCE_FREQUENCY

    # Setting common transfer parameters
    parameters.transfer.common.write_lock_enabled = 1
    parameters.transfer.common.record_buffer_memory_owner = pyadq.ADQ_MEMORY_OWNER_USER
    parameters.transfer.common.marker_mode = pyadq.ADQ_MARKER_MODE_HOST_MANUAL

    # Setting channel specific parameters
    for ch in range(s.NOF_ACTIVE_CHANNELS):
        # Setting test pattern, remove or set to ADQ_TEST_PATTERN_SOURCE_DISABLE
        # to disable.
        parameters.test_pattern.channel[ch].source = s.TEST_PATTERN_SOURCE[ch]

        # Setting skip factor, set to 1 to disable.
        parameters.signal_processing.sample_skip.channel[ch].skip_factor = s.SAMPLE_SKIP_FACTOR[ch]

        # Setting acquisition parameters
        parameters.acquisition.channel[ch].nof_records = (
            s.NOF_RECORDS_PER_BUFFER * s.NOF_BUFFERS_TO_RECEIVE
        )
        parameters.acquisition.channel[ch].record_length = s.RECORD_LEN[ch]
        parameters.acquisition.channel[ch].trigger_source = s.TRIGGER_SOURCE[ch]
        parameters.acquisition.channel[ch].trigger_edge = s.TRIGGER_EDGE[ch]
        parameters.acquisition.channel[ch].horizontal_offset = s.HORIZONTAL_OFFSET[ch]
        parameters.acquisition.channel[ch].nof_bits_per_sample = s.BYTES_PER_SAMPLE * 8

        # Setting transfer parameters
        parameters.transfer.channel[ch].infinite_record_length_enabled = 0
        parameters.transfer.channel[ch].record_size = (
            s.BYTES_PER_SAMPLE * parameters.acquisition.channel[ch].record_length
        )
        parameters.transfer.channel[ch].record_buffer_size = (
            s.NOF_RECORDS_PER_BUFFER * parameters.transfer.channel[ch].record_size
        )
        parameters.transfer.channel[ch].metadata_enabled = 0
        parameters.transfer.channel[ch].nof_buffers = s.NOF_GPU_BUFFERS

    # Create pointers, buffers and GDR object
    memory_handles = [
        [g.GdrMemoryHandle() for x in range(s.NOF_ACTIVE_CHANNELS)]
        for y in range(s.NOF_GPU_BUFFERS)
    ]

    bar_ptr_data = sh.BarPtrData(s.NOF_ACTIVE_CHANNELS, s.NOF_GPU_BUFFERS)
    gpu_buffer_ptr = sh.GpuBufferPointers(s.NOF_ACTIVE_CHANNELS, s.NOF_GPU_BUFFERS)
    gdr = gdrapi.gdr_open()
    gpu_buffers = sh.GpuBuffers(
        s.NOF_ACTIVE_CHANNELS,
        s.NOF_GPU_BUFFERS,
        s.RECORD_LEN,
    )
    gpu_buffer_address = 0

    # Allocate GPU buffers
    for ch in range(s.NOF_ACTIVE_CHANNELS):
        for b in range(s.NOF_GPU_BUFFERS):
            (
                gpu_buffer_ptr.pointers[b][ch],
                gpu_buffer_address,
                gpu_buffers.buffers[b][ch],
            ) = allocate_and_pin_buffer(
                parameters.transfer.channel[ch].record_buffer_size,
                memory_handles[b][ch],
                gdr,
                bar_ptr_data.pointers[b][ch],
            )

            parameters.transfer.channel[ch].record_buffer_bus_address[b] = gpu_buffer_address
            parameters.transfer.channel[ch].record_buffer[b] = gpu_buffer_ptr.pointers[b][ch]

    # Configure digitizer parameters
    dev.SetParameters(parameters)

    # Start timer for measurement
    time_start = time.time()
    # Start timer for regular printouts
    time_start_print = time.time()
    print("Start acquiring data")
    res = dev.ADQ_StartDataAcquisition()
    if res == pyadq.ADQ_EOK:
        print("Success")
    else:
        print("ADQ_StartDataAcquisition failed, exiting.")
        safe_exit(gdr, memory_handles, bar_ptr_data, exit_code=res)

    data_transfer_done = 0
    nof_buffers_received = [0, 0, 0, 0]
    bytes_received = 0
    status = pyadq.ADQP2pStatus()._to_ct()
    while not data_transfer_done:
        result = dev.ADQ_WaitForP2pBuffers(ct.byref(status), s.WAIT_TIMEOUT_MS)
        if result == pyadq.ADQ_EAGAIN:
            print("Timeout")
        elif result < 0:
            print(f"Failed with retcode {result}")
            safe_exit(gdr, memory_handles, bar_ptr_data, exit_code=result)

        else:
            buf = 0
            while (buf < status.channel[0].nof_completed_buffers) or (
                buf < status.channel[1].nof_completed_buffers
            ):
                for ch in range(s.NOF_ACTIVE_CHANNELS):
                    if buf < status.channel[ch].nof_completed_buffers:
                        buffer_index = status.channel[ch].completed_buffers[buf]
                        # gpu_buffers.buffers contains data ready for processing.
                        # It will be locked until ADQ_UnlockP2pBuffers is called.
                        # Unlock buffer when data can be overwritten.
                        dev.ADQ_UnlockP2pBuffers(ch, (1 << buffer_index))
                        nof_buffers_received[ch] += 1
                        bytes_received += (
                            s.NOF_RECORDS_PER_BUFFER * s.RECORD_LEN[0] * s.BYTES_PER_SAMPLE
                        )
                buf += 1
            data_transfer_done = np.all(
                [
                    nof_buffers_received[ch] >= s.NOF_BUFFERS_TO_RECEIVE
                    for ch in range(s.NOF_ACTIVE_CHANNELS)
                ]
            )
            time_now = time.time() - time_start
            time_print = time.time() - time_start_print
            if time_print > 5:
                # Check for overflow, stop if overflow
                overflow_status = dev.GetStatus(pyadq.ADQ_STATUS_ID_OVERFLOW)
                if overflow_status.overflow:
                    print("Overflow, stopping data acquisition...")
                    dev.ADQ_StopDataAcquisition()
                    safe_exit(gdr, memory_handles, bar_ptr_data)

                print("Nof buffers received:", nof_buffers_received)
                print("Total GB received:", bytes_received / 10**9)
                print("Average transfer speed:", bytes_received / 10**9 / time_now)
                time_start_print = time.time()

    time_stop = time.time()

    gbps = bytes_received / (time_stop - time_start)
    dev.ADQ_StopDataAcquisition()
    print(f"Total GB received: {bytes_received / 10**9}")
    print(f"Total GB/s: {gbps / 10**9}")

    if s.PRINT_SINGLE_BUFFER:
        # Prints a buffer and saves it to a figure.
        data_buffer = np.zeros(
            parameters.transfer.channel[0].record_buffer_size // 2, dtype=np.short
        )
        print(gpu_buffers.buffers[0][0])
        # Copies buffer to host with low level CUDA function via Cupy.
        hc.cudaMemcpy(
            data_buffer.ctypes.data,
            gpu_buffers.buffers[0][0].data.ptr,
            parameters.transfer.channel[0].record_buffer_size,
            hc.cudaMemcpyDeviceToHost,
        )

        data_buffer.tofile("data.bin")
        plt.plot(data_buffer)
        plt.savefig("SingleBuffer.png")
        # Saves buffer
    safe_exit(gdr, memory_handles, bar_ptr_data)


def safe_exit(gdr, memory_handles, bar_ptr_data, exit_code=None):
    for ch in range(s.NOF_ACTIVE_CHANNELS):
        buffer_size = s.NOF_RECORDS_PER_BUFFER * s.RECORD_LEN[ch] * s.BYTES_PER_SAMPLE
        # GPU page size rounding.
        buffer_size = (buffer_size + g.GPU_PAGE_SIZE - 1) & g.GPU_PAGE_MASK
        for b in range(s.NOF_GPU_BUFFERS):
            gdrapi.gdr_unmap(
                gdr,
                memory_handles[b][ch],
                bar_ptr_data.pointers[b][ch],
                buffer_size,
            )
            gdrapi.gdr_unpin_buffer(gdr, memory_handles[b][ch])
    # Free GPU memory
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    exit(exit_code)


if __name__ == "__main__":
    main()
