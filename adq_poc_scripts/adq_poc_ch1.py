#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from ctypes import ArgumentError

import numpy as np
import matplotlib.pyplot as plt
import pyadq


# ---------- Helpers ----------
def ceil_to_step(val, step):
    return ((val + step - 1) // step) * step


def save_plot(data: np.ndarray, outpath: Path, ch: int, recno: int,
              sample_rate_hz: float | None, max_samples: int | None):
    """Save a PNG plot of the waveform (CH0)."""
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]

    if sample_rate_hz and sample_rate_hz > 0:
        x = np.arange(data.size, dtype=float) / sample_rate_hz
        xlab = "Time (s)"
    else:
        x = np.arange(data.size, dtype=int)
        xlab = "Sample index"

    plt.figure(figsize=(10, 4))
    plt.plot(x, data)  # no explicit style/colors
    plt.title(f"ADQ33 CH{ch} – Record {recno}")
    plt.xlabel(xlab)
    plt.ylabel("Amplitude (ADC counts)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def ensure_csv(csv_path: Path):
    """Create CSV (shared across run) if missing."""
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "record_number",
                "record_length_ch0",
                "record_length_ch1",
                # CH1 first 128 bits as 8×u16 (big-endian words)
                "ch1_beat_u16_w0", "ch1_beat_u16_w1", "ch1_beat_u16_w2", "ch1_beat_u16_w3",
                "ch1_beat_u16_w4", "ch1_beat_u16_w5", "ch1_beat_u16_w6", "ch1_beat_u16_w7",
                # Same 16 bytes exactly as seen on the wire (hex, little-endian byte order)
                "ch1_beat_hex16",
                # CH0 artifacts for convenience
                "ch0_bin", "ch0_npy", "ch0_png",
            ])


def ch1_first_beat_words_and_hex(data_array):
    """
    Return (words8, hex32) from the first 16 bytes of payload.
    Interpret as **little-endian** 16-bit words (matches what the hex shows).
    """
    byte_view = np.asarray(data_array).view(np.uint8)
    if byte_view.size < 16:
        pad = np.zeros(16, dtype=np.uint8)
        pad[:byte_view.size] = byte_view
        first16 = pad
    else:
        first16 = byte_view[:16].copy()

    # Little-endian 16-bit words
    words_le = np.frombuffer(first16.tobytes(), dtype="<u2", count=8)
    hex32 = first16.tobytes().hex()
    return words_le, hex32


def return_record_buffer(dev, ch, rec):
    """
    Version-safe buffer return for pyadq.
    Tries record-scoped calls first, then device-scoped variants and signatures.
    """
    # 1) Record-local methods
    for name in ("ReturnRecordBuffer", "return_buffer", "ReturnBuffer"):
        m = getattr(rec, name, None)
        if callable(m):
            try:
                m()
                return
            except (TypeError, ArgumentError):
                pass

    # 2) Build candidate ctypes carriers for the buffer
    rec_args = [rec]
    for cand in ("cstruct", "ctypes", "_ctypes", "_cstruct", "_buf", "buffer", "_buffer", "_as_parameter_"):
        val = getattr(rec, cand, None)
        if val is not None:
            rec_args.append(val)

    # 3) Device-scoped (two spellings observed across versions)
    for dev_name in ("ADQ_ReturnRecordBuffer", "ReturnRecordBuffer"):
        dm = getattr(dev, dev_name, None)
        if not callable(dm):
            continue

        # Try (rec) then (ch, rec) for each candidate representation
        for arg in rec_args:
            try:
                dm(arg)
                return
            except (TypeError, ArgumentError):
                pass
            try:
                dm(int(ch), arg)
                return
            except (TypeError, ArgumentError):
                pass

    # 4) Last resort
    try:
        del rec
    except Exception:
        pass
# ---------- /Helpers ----------


def main():
    parser = argparse.ArgumentParser(
        description="ADQ33: capture CH0+CH1; plot/save CH0; CSV CH1 first 128b; records paired by record_number."
    )
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--record-length", type=int, default=1024)
    parser.add_argument("--records", type=int, default=5)
    parser.add_argument("--mode", choices=["software", "periodic"], default="software")
    parser.add_argument("--periodic-freq", type=float, default=10_000.0)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--png-samples", type=int, default=4000)
    parser.add_argument("--base-dir", type=Path, default=Path("/external_ssd/adq_data"))
    args = parser.parse_args()

    # Output folder + run CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.base_dir / ts
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "ch1_first128b_by_record.csv"
    ensure_csv(csv_path)

    # Control unit/device
    acu = pyadq.ADQControlUnit()
    acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, str(outdir))
    devlist = acu.ListDevices()
    if not devlist:
        raise RuntimeError("No ADQ devices detected (check /dev/adq_* and permissions).")

    if args.device_index is not None:
        if not (0 <= args.device_index < len(devlist)):
            raise RuntimeError(f"--device-index {args.device_index} is out of range (found {len(devlist)} device(s)).")
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

    # Params: arm CH0 & CH1 identically so record_number aligns
    params: pyadq.ADQParameters = dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)
    if int(params.constant.nof_channels) < 2:
        raise RuntimeError("This script requires CH0 and CH1.")

    for ch in (0, 1):
        params.acquisition.channel[ch].record_length = int(args.record_length)
        params.acquisition.channel[ch].nof_records = int(args.records)
        params.acquisition.channel[ch].trigger_edge = pyadq.ADQ_EDGE_RISING
        params.acquisition.channel[ch].trigger_source = (
            pyadq.ADQ_EVENT_SOURCE_SOFTWARE if args.mode == "software"
            else pyadq.ADQ_EVENT_SOURCE_PERIODIC
        )

        bps = int(params.constant.channel[ch].nof_bytes_per_sample)
        rec_buf_sz = ceil_to_step(args.record_length * bps, params.constant.record_buffer_size_step)
        rec_buf_sz = min(rec_buf_sz, 512 * 1024)
        ptx = params.transfer.channel[ch]
        ptx.record_size = 0
        ptx.infinite_record_length_enabled = 0
        ptx.record_buffer_size = rec_buf_sz
        ptx.dynamic_record_length_enabled = 1
        ptx.metadata_enabled = 1
        ptx.metadata_buffer_size = pyadq.SIZEOF_ADQ_GEN4_HEADER
        ptx.nof_buffers = pyadq.ADQ_MAX_NOF_BUFFERS
        ptx.eject_buffer_source = pyadq.ADQ_FUNCTION_RECORD_STOP

    if args.mode == "periodic":
        params.event_source.periodic.frequency = float(args.periodic_freq)

    dev.SetParameters(params)

    # Save run metadata
    meta = {
        "timestamp": ts,
        "device_index": didx,
        "channels_active": [0, 1],
        "record_length": int(args.record_length),
        "records": int(args.records),
        "trigger_mode": args.mode,
        "periodic_frequency_hz": float(args.periodic_freq) if args.mode == "periodic" else None,
        "sample_rate_hz": float(args.sample_rate) if args.sample_rate else None,
        "output_dir": str(outdir),
        "csv": str(csv_path),
        "csv_note": "CH1 first 16 bytes per record as 8×uint16 **big-endian words** (matches FPGA packing).",
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Start & trigger
    rc = dev.ADQ_StartDataAcquisition()
    if rc != pyadq.ADQ_EOK:
        raise RuntimeError(f"ADQ_StartDataAcquisition failed with code {rc}")

    if args.mode == "software":
        for _ in range(args.records):
            dev.ADQ_SWTrig()

    print(f"[ADQ] Capturing CH0+CH1: {args.records} record pairs, record_length={args.record_length}")

    # Pairing by record_number
    pending_ch0 = {}  # recno -> (data, rlen)
    pending_ch1 = {}  # recno -> (words8, hex32, rlen)
    committed = 0
    target_pairs = args.records

    while committed < target_pairs:
        rec = dev.WaitForRecordBuffer(pyadq.ADQ_ANY_CHANNEL, 2000)  # 2s
        rch = int(rec.header.channel)
        recno = int(rec.header.record_number)
        rlen = int(rec.header.record_length)
        data = np.asarray(rec.data)

        if rch == 0:
            pending_ch0[recno] = (data, rlen)
        elif rch == 1:
            words8, hex32 = ch1_first_beat_words_and_hex(data)
            pending_ch1[recno] = (words8, hex32, rlen)
        else:
            # ignore other channels (if any)
            pass

        # Try to commit when both sides exist
        if (recno in pending_ch0) and (recno in pending_ch1):
            ch0_data, rlen0 = pending_ch0.pop(recno)
            words8, hex32, rlen1 = pending_ch1.pop(recno)

            # Save CH0 artifacts
            base = f"ch0_rec{recno:06d}"
            np.save(outdir / f"{base}.npy", ch0_data)
            ch0_bin = outdir / f"{base}.bin"
            ch0_png = outdir / f"{base}.png"
            ch0_data.astype(np.int16, copy=False).tofile(ch0_bin)
            save_plot(
                data=ch0_data,
                outpath=ch0_png,
                ch=0,
                recno=recno,
                sample_rate_hz=args.sample_rate,
                max_samples=None if args.png_samples == -1 else int(args.png_samples),
            )

            # Append CSV row (CH1 beat + CH0 filenames)
            with csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    recno, rlen0, rlen1,
                    *[int(x) for x in words8], hex32,
                    ch0_bin.name, f"{base}.npy", ch0_png.name
                ])

            committed += 1
            print(f"[ADQ] Pair REC{recno}: CH0 saved (+plot), CH1 beat→CSV")

        # Always return buffer after processing
        return_record_buffer(dev, rch, rec)

    stop_rc = dev.ADQ_StopDataAcquisition()
    if stop_rc not in (pyadq.ADQ_EOK, pyadq.ADQ_EINTERRUPTED):
        raise RuntimeError(f"ADQ_StopDataAcquisition returned {stop_rc}")

    if pending_ch0 or pending_ch1:
        print(f"[ADQ] Note: unmatched records after stop. CH0-only={len(pending_ch0)}, CH1-only={len(pending_ch1)}")

    print(f"✅ Done. Folder: {outdir}\n   CSV: {csv_path}")


if __name__ == "__main__":
    main()
