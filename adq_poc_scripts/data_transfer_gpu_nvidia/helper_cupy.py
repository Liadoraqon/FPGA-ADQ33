# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
import cupy as cp

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaHostRegisterIoMemory = 4


# There are many low-level cuda functions supported in CuPy, which can be found in the
# documentation.
def cudaMemcpy(dst, src, size, kind):
    return cp.cuda.runtime.memcpy(dst, src, size, kind)


def sizeof(datatype):
    return cp.dtype(datatype).itemsize
