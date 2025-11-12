# Copyright 2023 Teledyne Signal Processing Devices Sweden AB
""" Python wrapper for gdrapi """
import ctypes as ct
import os
import sys

GPU_PAGE_SHIFT = 16
GPU_PAGE_SIZE = 1 << GPU_PAGE_SHIFT
GPU_PAGE_OFFSET = GPU_PAGE_SIZE - 1
GPU_PAGE_MASK = ~GPU_PAGE_OFFSET

if "libgdrapi.so" not in os.listdir("/usr/local/lib/"):
    sys.exit("ERROR: libgdrapi.so not found! Compile gdrcopy! See README.md.")


class Gdr(ct.Structure):
    """Wrapped gdr struct"""

    _fields_ = [("fd", ct.c_int)]


class GdrMemoryHandle(ct.Structure):
    """Wrapped gdr_mh_s struct"""

    _fields_ = [("h", ct.c_ulong)]


class GdrInfo(ct.Structure):
    """Wrapped gdr_info struct"""

    _fields_ = [
        ("va", ct.c_uint64),
        ("mapped_size", ct.c_uint64),
        ("page_size", ct.c_uint32),
        ("tm_cycles", ct.c_uint64),
        ("cycles_per_ms", ct.c_uint32),
        ("physical", ct.c_uint64),
        ("mapped", ct.c_int),
        ("wc_mapping", ct.c_int),
    ]


class GdrPhybar(ct.Structure):
    """Wrapped gdr_phybar struct"""

    _fields_ = [
        ("idx", ct.c_uint64),
        ("entires", ct.c_uint64),
        ("physical", ct.c_uint64),
    ]


# Shared object from gdrapi
_gdrapi = ct.CDLL("/usr/local/lib/libgdrapi.so")


class GdrPtrVa:
    """Pointers for gdr object"""

    def __init__(self, size):
        self.handles = [ct.c_void_p()] * size


class GdrApi:
    """Main gdrapi calls"""

    def __init__(self, lib=_gdrapi):
        # gdr_t gdr_open()
        lib.gdr_open.restype = ct.POINTER(Gdr)
        # int gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size,
        # uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle)
        lib.gdr_pin_buffer.argtypes = (
            ct.POINTER(Gdr),
            ct.c_ulong,
            ct.c_size_t,
            ct.c_uint64,
            ct.c_uint32,
            ct.POINTER(GdrMemoryHandle),
        )
        lib.gdr_pin_buffer.restype = ct.c_int
        # int gdr_map(gdr_t g, gdr_mh_t handle, void **ptr_va, size_t size)
        lib.gdr_map.argtypes = (
            ct.POINTER(Gdr),
            GdrMemoryHandle,
            ct.POINTER(ct.c_void_p),
            ct.c_size_t,
        )
        lib.gdr_map.restype = ct.c_int
        # int gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info)
        lib.gdr_get_info.argtypes = (
            ct.POINTER(Gdr),
            GdrMemoryHandle,
            ct.POINTER(GdrInfo),
        )
        lib.gdr_get_info.restype = ct.c_int
        # int gdr_get_phybar(gdr_t g, gdr_mh_t handle, uint64_t idx, gdr_phybar_t *phybar)
        lib.gdr_get_phybar.argtypes = (
            ct.POINTER(Gdr),
            GdrMemoryHandle,
            ct.c_uint64,
            ct.POINTER(GdrPhybar),
        )
        lib.gdr_get_phybar.restype = ct.c_int
        # int gdr_validate_phybar(gdr_t g, gdr_mh_t mh)
        lib.gdr_validate_phybar.argtypes = (ct.POINTER(Gdr), GdrMemoryHandle)
        lib.gdr_validate_phybar.restype = ct.c_int
        # int gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size)
        lib.gdr_unmap.argtypes = (ct.POINTER(Gdr), GdrMemoryHandle, ct.c_void_p, ct.c_size_t)
        lib.gdr_unmap.restype = ct.c_int
        # int gdr_unpin_buffer(gdr_t g, gdr_mh_t handle)
        lib.gdr_unpin_buffer.argtypes = (ct.POINTER(Gdr), GdrMemoryHandle)
        lib.gdr_unpin_buffer.restype = ct.c_int
        self.lib = lib

    def gdr_open(self):
        """Open gdr object"""
        return self.lib.gdr_open()

    def gdr_pin_buffer(self, gdr, addr, size, p2p_token, va_space, handle_ptr):
        """Pin buffer"""
        return self.lib.gdr_pin_buffer(gdr, addr, size, p2p_token, va_space, handle_ptr)

    def gdr_map(self, gdr, handle_ptr, ptr_va, size):
        """Map memory"""
        return self.lib.gdr_map(gdr, handle_ptr, ptr_va, size)

    def gdr_get_info(self, gdr, handle_ptr, info):
        """Get general info"""
        return self.lib.gdr_get_info(gdr, handle_ptr, info)

    def gdr_get_phybar(self, gdr, handle_ptr, idx, phybar):
        """Get physical bar info"""
        return self.lib.gdr_get_phybar(gdr, handle_ptr, idx, phybar)

    def gdr_validate_phybar(self, gdr, handle_ptr):
        """Validate physical bar"""
        return self.lib.gdr_validate_phybar(gdr, handle_ptr)

    def gdr_unmap(self, gdr, handle_ptr, va, size):
        """Unmap"""
        return self.lib.gdr_unmap(gdr, handle_ptr, va, size)

    def gdr_unpin_buffer(self, gdr, handle_ptr):
        """Unpin buffer"""
        return self.lib.gdr_unpin_buffer(gdr, handle_ptr)


def gdr_check_error_exit_func(err, string):
    """Check for error, exit if found"""
    if err != 0:
        sys.exit(f"Error {err} from function {string}.")
