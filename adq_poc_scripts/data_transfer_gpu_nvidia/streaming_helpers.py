# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
import ctypes as ct
import cupy as cp


class GpuBufferPointers:
    """Pointer to each buffer on each channel"""

    def __init__(self, nof_channels, nof_gpu_buffers):
        self.pointers = [
            [ct.c_void_p() for x in range(nof_channels)] for y in range(nof_gpu_buffers)
        ]


class BarPtrData:
    """Pointer to data on bar for each channel"""

    def __init__(self, nof_channels, nof_gpu_buffers):
        self.pointers = [
            [ct.c_void_p() for x in range(nof_channels)] for y in range(nof_gpu_buffers)
        ]


class GpuBuffers:
    """Allocate space on GPU for each channel"""

    def __init__(self, nof_channels, nof_gpu_buffers, nof_samples) -> None:
        self.buffers = [
            [cp.zeros(nof_samples[x], dtype=cp.short) for x in range(nof_channels)]
            for y in range(nof_gpu_buffers)
        ]
