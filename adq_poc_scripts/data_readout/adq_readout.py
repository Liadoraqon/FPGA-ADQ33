#!/usr/bin/env python3
import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pyadq


def ceil_to_step(val: int, step: int) -> int:
    return ((val + step - 1) // step) * step


def main():
    parser = argparse.ArgumentParser(
        description="ADQ33 first read and saver (no external trigger needed)."
    )
    parser.add_argument("--device-index", type=int, default=None,
                        help="Device index to open (default: first detected).")
    parser.add_argument("--record-length", type=int, default=1024,
                        help="Samples per record (default: 1024).")
    parser.add_argument("--records", type=int, default=10,
                        help="Number of records per channel (default: 10).")
    parser.add_argument("--mode", choices=["software", "periodic"], default="software",
                        help="Trigger mode. 'software' fires SWTrig(); 'periodic' uses internal generator.")
    parser.add_argument("--periodic-freq", type=float, default=10_000.0,
                        help="Periodic event frequency in Hz (if --mode periodic). Default: 10 kHz.")
    parser.add_argument("--base-dir", type=Path, default=Path("/external_ssd/adq_data"),
                        help="Base directory for output (default: /external_ssd/adq_data).")
    args = parser.parse_args()

    # Prepare timestamped output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.base_dir / ts
    outdir.mkdir(parents=True, exist_ok=True)

    # Open control unit and list devices
    acu = pyadq.ADQControlUnit()
    acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, str(outdir))  # logs will land in outdir
    devlist = acu.ListDevices()
    if not devlist:
        raise RuntimeError("No ADQ devices detected by pyadq. Check permissions (/dev/adq_*) and cabling.")

    # Choose a device
    if args.device_index is not None:
        if args.device_index < 0 or args.device_index >= len(devlist):
            raise RuntimeError(f"Invalid --device-index {args.device_index}; {len(devlist)} device(s) found.")
        didx = args.device_index
    else:
        # Prefer an explicitly recognized ADQ33, else take first device
        didx = 0
        for i, d in enumerate(devlist):
            try:
                if d.ProductID in [pyadq.PID_ADQ33]:
                    didx = i
                    break
            except Exception:
                pass

    dev: pyadq.ADQ = acu.SetupDevice(didx)

    # Initialize parameter tree
    params: pyadq.ADQParameters = dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)

    nof_channels = params.constant.nof_channels
    bytes_per_sample = params.constant.channel[0].nof_bytes_per_sample
    record_len = int(args.record_length)

    # Acquisition for ALL channels (uniform config)
    for ch in range(nof_channels):
        params.acquisition.channel[ch].record_length = record_len
        params.acquisition.channel[ch].nof_records = args.records
        params.acquisition.channel[ch].trigger_edge = pyadq.ADQ_EDGE_RISING
        if args.mode == "software":
            params.acquisition.channel[ch].trigger_source = pyadq.ADQ_EVENT_SOURCE_SOFTWARE
        else:
            params.acquisition.channel[ch].trigger_source = pyadq.ADQ_EVENT_SOURCE_PERIODIC

    # Transfer settings (safe defaults; dynamic record length on)
    for ch in range(nof_channels):
        rec_buf_sz = record_len * params.constant.channel[ch].nof_bytes_per_sample
        rec_buf_sz = ceil_to_step(rec_buf_sz, params.constant.record_buffer_size_step)
        # keep buffer modest for Jetson contig alloc:
        rec_buf_sz = min(rec_buf_sz, 512 * 1024)

        params.transfer.channel[ch].record_size = 0
        params.transfer.channel[ch].infinite_record_length_enabled = 0
        params.transfer.channel[ch].record_buffer_size = rec_buf_sz
        params.transfer.channel[ch].dynamic_record_length_enabled = 1
        params.transfer.channel[ch].metadata_enabled = 1
        params.transfer.channel[ch].metadata_buffer_size = pyadq.SIZEOF_ADQ_GEN4_HEADER
        params.transfer.channel[ch].nof_buffers = pyadq.ADQ_MAX_NOF_BUFFERS
        params.transfer.channel[ch].eject_buffer_source = pyadq.ADQ_FUNCTION_RECORD_STOP

    # Periodic event generator config (if requested)
    if args.mode == "periodic":
        params.event_source.periodic.frequency = float(args.periodic_freq)
        # default sync mode is fine; periodic starts immediately

    # Apply parameters
    dev.SetParameters(params)

    # Save minimal metadata file
    meta = {
        "timestamp": ts,
        "device_index": didx,
        "nof_channels": int(nof_channels),
        "record_length": record_len,
        "records_per_channel": int(args.records),
        "trigger_mode": args.mode,
        "periodic_frequency_hz": float(args.periodic_freq) if args.mode == "periodic" else None,
        "bytes_per_sample": int(bytes_per_sample),
        "output_dir": str(outdir),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Start acquisition
    rc = dev.ADQ_StartDataAcquisition()
    if rc != pyadq.ADQ_EOK:
        raise RuntimeError(f"ADQ_StartDataAcquisition failed with code {rc} (see trace logs in {outdir})")

    # If software mode, fire one SWTrig per expected record batch; simplest is records_per_channel times
    if args.mode == "software":
        # Fire enough software events to produce the requested number of records per channel
        total_events = args.records
        for _ in range(total_events):
            dev.ADQ_SWTrig()

    print(f"[ADQ] Capturing {args.records} record(s)/channel, record_length={record_len} samples, mode={args.mode}")

    # Read loop: collect records until we reach target across all channels
    target = args.records * nof_channels
    got = 0
    while got < target:
        # 1000 ms timeout
        rec = dev.WaitForRecordBuffer(pyadq.ADQ_ANY_CHANNEL, 1000)
        # rec.header has: channel, record_number, record_length, time_unit, timestamp, etc.
        ch = rec.header.channel
        recno = rec.header.record_number
        rlen = rec.header.record_length

        # Copy numpy array
        arr = np.asarray(rec.data)

        # File names
        base = f"ch{ch}_rec{recno:06d}"
        npy_path = outdir / f"{base}.npy"
        bin_path = outdir / f"{base}.bin"

        # Save
        np.save(npy_path, arr)
        # Raw signed 16-bit is standard; if your FW packs differently, adjust dtype
        arr.astype(np.int16, copy=False).tofile(bin_path)

        print(f"[ADQ] Saved CH{ch} REC{recno} -> {bin_path.name} ({arr.size} samples)")

        got += 1

    # Stop acquisition
    stop_rc = dev.ADQ_StopDataAcquisition()
    if stop_rc not in (pyadq.ADQ_EOK, pyadq.ADQ_EINTERRUPTED):
        raise RuntimeError(f"ADQ_StopDataAcquisition returned {stop_rc}")

    print(f"[ADQ] Done. Files saved in: {outdir}")


if __name__ == "__main__":
    main()

