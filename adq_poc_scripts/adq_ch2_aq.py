#!/usr/bin/env python3
#python3 capture.py --channel 0 --records 10 --record-length 4096 --mode software --enable-ch2 --debug-introspect
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyadq


BANNER = "[capture_adq33] v2 (robust ReturnBuffer shim, introspection)"


def ceil_to_step(val, step):
    return ((val + step - 1) // step) * step


def save_plot(data: np.ndarray, outpath: Path, ch: int, recno: int,
              sample_rate_hz: float | None, max_samples: int | None):
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
    if sample_rate_hz and sample_rate_hz > 0:
        x = np.arange(data.size, dtype=float) / sample_rate_hz
        xlab = "Time (s)"
    else:
        x = np.arange(data.size, dtype=int)
        xlab = "Sample index"
    plt.figure(figsize=(10, 4))
    plt.plot(x, data)
    plt.title(f"ADQ33 CH{ch} – Record {recno}")
    plt.xlabel(xlab)
    plt.ylabel("Amplitude (ADC counts)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def decode_ch2_meta_u16(u16: np.ndarray) -> dict:
    # [0]=AAAA, [1]=BBBB, [2]=CCCC, [3]=DDDD, [4]=EEEE, [5]=FFFF, [6]=CNT16, [7]=ABCD
    if u16.size < 8:
        raise ValueError(f"CH2 meta expects 8x uint16, got {u16.size}")
    return {
        "AAAA": int(u16[0]),
        "BBBB": int(u16[1]),
        "CCCC": int(u16[2]),
        "DDDD": int(u16[3]),
        "EEEE": int(u16[4]),
        "FFFF": int(u16[5]),
        "CNT16": int(u16[6]),
        "ABCD": int(u16[7]),
    }


# ---------- pyadq compatibility (pointer-hunting) ----------
def return_record_buffer(dev, rch, rec, *, debug=False):
    """
    Return the record buffer across pyadq variants.
    Your build exposes dev.ADQ_ReturnRecordBuffer but it expects a C pointer.
    This hunts for the pointer inside the Python wrapper and tries all call shapes.
    """
    import ctypes
    import inspect

    def _call(label, fn, *args):
        try:
            if debug:
                name = getattr(fn, "__name__", None) or getattr(fn, "__qualname__", None) or str(fn)
                print(f"[shim] trying {label}: {name}(" + ", ".join(type(a).__name__ for a in args) + ")")
            return fn(*args)
        except Exception as e:
            if debug:
                print(f"[shim]   -> {type(e).__name__}: {e}")
            return None

    # 1) Build candidate list of "pointer-like" attributes from the wrapper.
    cand = [rec]
    # common names seen in different wheels/bindings
    likely_names = [
        "_as_parameter_", "this", "ptr", "_ptr", "c_ptr", "cptr", "capsule", "_capsule",
        "_cobj", "c_object", "_c_object", "handle", "_handle", "__handle__", "buffer",
        "_buffer", "__buffer__", "raw", "_raw", "ref", "_ref"
    ]
    for n in likely_names:
        if hasattr(rec, n):
            try:
                cand.append(getattr(rec, n))
            except Exception:
                pass

    # scan any attr that *looks* pointery to widen the net
    for n in dir(rec):
        nl = n.lower()
        if any(k in nl for k in ("ptr", "caps", "cobj", "handle", "buffer", "raw", "ref")):
            try:
                cand.append(getattr(rec, n))
            except Exception:
                pass

    # ctypes byref (works only if rec is a ctypes.Structure)
    try:
        cand.append(ctypes.byref(rec))
    except Exception:
        pass

    # dedupe while keeping order
    seen = set()
    cands = []
    for x in cand:
        try:
            k = (type(x), id(x))
        except Exception:
            k = (type(x), object.__hash__(x))
        if k not in seen:
            seen.add(k)
            cands.append(x)

    # 2) Try the C-style device function first (your build)
    c_fn = getattr(dev, "ADQ_ReturnRecordBuffer", None)
    if callable(c_fn):
        for c in cands:
            res = _call("dev.ADQ_ReturnRecordBuffer", c_fn, rch, c)
            if res is not None:
                if debug: print("[shim] ✅ using dev.ADQ_ReturnRecordBuffer(ch, <c-ptr>)")
                return res

    # 3) Try other device-level names seen in other wheels
    fn = getattr(dev, "ReturnRecordBuffer", None)
    if callable(fn):
        for c in cands:
            res = _call("dev.ReturnRecordBuffer", fn, rch, c)
            if res is not None:
                if debug: print("[shim] ✅ using dev.ReturnRecordBuffer(ch, <c-ptr>)")
                return res

    fn = getattr(dev, "ReturnBuffer", None)
    if callable(fn):
        # try wrapper then pointery candidates
        res = _call("dev.ReturnBuffer(rec)", fn, rec)
        if res is not None:
            if debug: print("[shim] ✅ using dev.ReturnBuffer(rec)")
            return res
        for c in cands:
            res = _call("dev.ReturnBuffer(cand)", fn, c)
            if res is not None:
                if debug: print("[shim] ✅ using dev.ReturnBuffer(<c-ptr>)")
                return res

    # 4) Try buffer-object methods (rare bindings)
    for m in ("Return", "release", "Release"):
        if hasattr(rec, m):
            meth = getattr(rec, m)
            if callable(meth):
                res = _call(f"rec.{m}()", meth)
                if res is not None:
                    if debug: print(f"[shim] ✅ using rec.{m}()")
                    return res

    # 5) Module-level fallbacks (some wheels export C API at top-level)
    for name in ("ADQ_ReturnRecordBuffer", "ReturnRecordBuffer"):
        fn = getattr(pyadq, name, None)
        if callable(fn):
            for c in cands:
                res = _call(f"pyadq.{name}", fn, dev, rch, c)
                if res is not None:
                    if debug: print(f"[shim] using pyadq.{name}(dev, ch, <c-ptr>)")
                    return res

    # 6) Nothing worked—print a short diagnostic
    if debug:
        print("[shim]  unable to return buffer; inspected candidates:")
        for x in cands[:20]:
            r = repr(x)
            if len(r) > 120: r = r[:117] + "..."
            print("   -", type(x).__name__, r)
        # Also list any dunder-named attributes that might hide a capsule
        hidden = [n for n in dir(rec) if n.startswith("__") and ("ptr" in n.lower() or "caps" in n.lower())]
        if hidden:
            print("   hidden-like attrs:", hidden)

    raise AttributeError("No working buffer-return path for this pyadq build.")





def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        description="ADQ33 capture of one main channel (+ optional CH2 meta channel)."
    )
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--enable-ch2", action="store_true",
                        help="Also capture Channel 2 meta (single-beat records).")
    parser.add_argument("--record-length", type=int, default=4096)
    parser.add_argument("--records", type=int, default=10)
    parser.add_argument("--mode", choices=["software", "periodic"], default="software")
    parser.add_argument("--periodic-freq", type=float, default=10_000.0)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--png-samples", type=int, default=4000)
    parser.add_argument("--base-dir", type=Path, default=Path("/external_ssd/adq_data"))
    parser.add_argument("--debug-introspect", action="store_true",
                        help="Print available methods and which shim path is used.")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.base_dir / ts
    outdir.mkdir(parents=True, exist_ok=True)

    acu = pyadq.ADQControlUnit()
    acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, str(outdir))
    devlist = acu.ListDevices()
    if not devlist:
        raise RuntimeError("No ADQ devices detected.")

    # Pick device (prefer ADQ33)
    if args.device_index is not None:
        if args.device_index < 0 or args.device_index >= len(devlist):
            raise RuntimeError(f"Invalid --device-index {args.device_index}; {len(devlist)} device(s) found.")
        didx = args.device_index
    else:
        didx = 0
        for i, d in enumerate(devlist):
            try:
                if d.ProductID == pyadq.PID_ADQ33:
                    didx = i
                    break
            except Exception:
                pass

    dev: pyadq.ADQ = acu.SetupDevice(didx)

    if args.debug_introspect:
        def filt(obj, key):
            return [n for n in dir(obj) if key.lower() in n.lower()]
        print("[debug] device methods ~Return/~Buffer:", filt(dev, "Return") + filt(dev, "Buffer"))
        # We’ll also inspect a dummy Wait in a moment to see rec methods.

    params: pyadq.ADQParameters = dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)
    nof_ch = int(params.constant.nof_channels)

    main_ch = int(args.channel)
    if main_ch < 0 or main_ch >= nof_ch:
        raise RuntimeError(f"Main channel {main_ch} not available; device reports {nof_ch} channel(s).")

    meta_ch = 2
    capture_ch2 = bool(args.enable_ch2 and meta_ch < nof_ch)

    main_bps = int(params.constant.channel[main_ch].nof_bytes_per_sample)

    # Acquisition
    params.acquisition.channel[main_ch].record_length = int(args.record_length)
    params.acquisition.channel[main_ch].nof_records = int(args.records)
    params.acquisition.channel[main_ch].trigger_edge = pyadq.ADQ_EDGE_RISING
    params.acquisition.channel[main_ch].trigger_source = (
        pyadq.ADQ_EVENT_SOURCE_SOFTWARE if args.mode == "software"
        else pyadq.ADQ_EVENT_SOURCE_PERIODIC
    )

    if args.mode == "periodic":
        params.event_source.periodic.frequency = float(args.periodic_freq)

    if capture_ch2:
        params.acquisition.channel[meta_ch].record_length = 1
        params.acquisition.channel[meta_ch].nof_records = int(args.records)
        params.acquisition.channel[meta_ch].trigger_edge = pyadq.ADQ_EDGE_RISING
        params.acquisition.channel[meta_ch].trigger_source = pyadq.ADQ_EVENT_SOURCE_SOFTWARE

    # Transfer
    rec_buf_sz_main = args.record_length * main_bps
    rec_buf_sz_main = ceil_to_step(rec_buf_sz_main, params.constant.record_buffer_size_step)
    rec_buf_sz_main = min(rec_buf_sz_main, 512 * 1024)

    ptx_main = params.transfer.channel[main_ch]
    ptx_main.record_size = 0
    ptx_main.infinite_record_length_enabled = 0
    ptx_main.record_buffer_size = rec_buf_sz_main
    ptx_main.dynamic_record_length_enabled = 1
    ptx_main.metadata_enabled = 1
    ptx_main.metadata_buffer_size = pyadq.SIZEOF_ADQ_GEN4_HEADER
    ptx_main.nof_buffers = pyadq.ADQ_MAX_NOF_BUFFERS
    ptx_main.eject_buffer_source = pyadq.ADQ_FUNCTION_RECORD_STOP

    if capture_ch2:
        bytes_per_sample_ch2 = int(params.constant.channel[meta_ch].nof_bytes_per_sample)  # expect 2
        ch2_rec_bytes = max(16, bytes_per_sample_ch2 * 8)
        ch2_rec_buf_sz = ceil_to_step(ch2_rec_bytes, params.constant.record_buffer_size_step)
        ch2_rec_buf_sz = max(ch2_rec_buf_sz, 4096)
        ch2_rec_buf_sz = min(ch2_rec_buf_sz, 128 * 1024)
        ptx_ch2 = params.transfer.channel[meta_ch]
        ptx_ch2.record_size = 0
        ptx_ch2.infinite_record_length_enabled = 0
        ptx_ch2.record_buffer_size = ch2_rec_buf_sz
        ptx_ch2.dynamic_record_length_enabled = 1
        ptx_ch2.metadata_enabled = 1
        ptx_ch2.metadata_buffer_size = pyadq.SIZEOF_ADQ_GEN4_HEADER
        ptx_ch2.nof_buffers = pyadq.ADQ_MAX_NOF_BUFFERS
        ptx_ch2.eject_buffer_source = pyadq.ADQ_FUNCTION_RECORD_STOP

    dev.SetParameters(params)

    # Start
    rc = dev.ADQ_StartDataAcquisition()
    if rc != pyadq.ADQ_EOK:
        raise RuntimeError(f"ADQ_StartDataAcquisition failed with code {rc}")

    if args.mode == "software":
        for _ in range(args.records):
            dev.ADQ_SWTrig()

    print(f"[ADQ] Capturing main CH{main_ch} ({args.records} recs, len={args.record_length})"
          + (f" + CH{meta_ch} meta" if capture_ch2 else ""))

    got_main = 0
    target_main = args.records
    ch2_rows = []

    while got_main < target_main:
        rec = dev.WaitForRecordBuffer(pyadq.ADQ_ANY_CHANNEL, 2000)
        if rec is None:
            continue

        if args.debug_introspect:
            # One-time peek at rec methods
            print("[debug] rec type:", type(rec))
            print("[debug] rec methods ~Return/~release:", [n for n in dir(rec) if ("Return" in n or "release" in n)])
            args.debug_introspect = False  # only print once

        rch = rec.header.channel
        recno = rec.header.record_number
        rlen = rec.header.record_length
        data = np.asarray(rec.data)

        if rch == main_ch:
            base = f"ch{rch}_rec{recno:06d}"
            np.save(outdir / f"{base}.npy", data)
            data.astype(np.int16, copy=False).tofile(outdir / f"{base}.bin")
            save_plot(data=data, outpath=outdir / f"{base}.png",
                      ch=rch, recno=recno, sample_rate_hz=args.sample_rate,
                      max_samples=None if args.png_samples == -1 else int(args.png_samples))
            print(f"[ADQ] MAIN  CH{rch} REC{recno} ({len(data)} samples) saved.")
            got_main += 1

        elif capture_ch2 and rch == meta_ch:
            u16 = data.view(np.int16)
            try:
                fields = decode_ch2_meta_u16(u16)
            except Exception as e:
                fields = {"decode_error": str(e)}
            row = {
                "record_number": int(recno),
                "record_length": int(rlen),
                "n_samples": int(u16.size),
                **fields,
            }
            ch2_rows.append(row)
            base = f"ch{rch}_rec{recno:06d}"
            np.save(outdir / f"{base}.npy", u16.astype(np.uint16, copy=False))
            u16.astype(np.uint16, copy=False).tofile(outdir / f"{base}.bin")
            (outdir / f"{base}.json").write_text(json.dumps(row, indent=2))
            print(f"[ADQ] META  CH{rch} REC{recno} (u16[{u16.size}]) saved.")

        # return buffer using robust shim
        return_record_buffer(dev, rch, rec, debug=True)

    stop_rc = dev.ADQ_StopDataAcquisition()
    if stop_rc not in (pyadq.ADQ_EOK, pyadq.ADQ_EINTERRUPTED):
        raise RuntimeError(f"ADQ_StopDataAcquisition returned {stop_rc}")

    if ch2_rows:
        ch2_rows.sort(key=lambda r: r.get("record_number", 0))
        csv_path = outdir / "ch2_meta.csv"
        keys = ["record_number", "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "CNT16", "ABCD",
                "record_length", "n_samples"]
        for r in ch2_rows:
            for k in keys:
                r.setdefault(k, "")
        with csv_path.open("w") as f:
            f.write(",".join(keys) + "\n")
            for r in ch2_rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"[ADQ] Wrote {csv_path.name} with {len(ch2_rows)} rows.")

    print("✅ Done.")


if __name__ == "__main__":
    main()
