import pyadq
import ctypes as ct

cu = pyadq.ADQControlUnit()
adq = cu.SetupDevice(0)

UL1_TARGET  = 1
ZERO_CTRL   = 3
ZERO_OFFSET = 4
ZERO_LENGTH = 5
ALL_BITS    = 0xFFFFFFFF

def write_ul1_reg(regnum, value):
    rc = adq.WriteUserRegister(
        UL1_TARGET,             # ul_target
        ct.c_uint32(regnum),    # regnum
        ct.c_uint32(ALL_BITS),  # mask (all bits affected)
        ct.c_uint32(value)      # data
    )
    if rc != 0:
        raise RuntimeError(f"WriteUserRegister rc={rc}")

def read_ul1_reg(regnum):
    # Try 2-arg form first (your error message indicates this is correct)
    try:
        val = adq.ReadUserRegister(UL1_TARGET, ct.c_uint32(regnum))
        # Some builds return a Python int; some return ctypes c_uint32
        return int(val)
    except TypeError:
        # Fallback to 3-arg form (with pointer) if your build requires it
        v = ct.c_uint32()
        rc = adq.ReadUserRegister(UL1_TARGET, ct.c_uint32(regnum), ct.byref(v))
        if rc != 0:
            raise RuntimeError(f"ReadUserRegister rc={rc}")
        return v.value

# --- example: write and verify ---
write_ul1_reg(ZERO_CTRL,   1)   # enable
write_ul1_reg(ZERO_OFFSET, 5)   # beats
write_ul1_reg(ZERO_LENGTH, 10)  # beats

ctrl   = read_ul1_reg(ZERO_CTRL)
offset = read_ul1_reg(ZERO_OFFSET)
length = read_ul1_reg(ZERO_LENGTH)

print(f"ZERO_CTRL   = 0x{ctrl:08X}")
print(f"ZERO_OFFSET = {offset}")
print(f"ZERO_LENGTH = {length}")
