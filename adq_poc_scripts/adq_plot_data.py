#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyadq


def ceil_to_step(val, step):
    return ((val + step - 1) // step) * step


def save_plot(data: np.ndarray, outpath: Path, ch: int, recno: int,
              sample_rate_hz: float | None, max_samples: int | None):
    """Save a PNG plot of the waveform."""
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]

    if sample_rate_hz and sample_rate_hz > 0:
        x = np.arange(data.size, dtype=float) / sample_rate_hz
        xlab = "Time (s)"
    else:
        x = np.arange(data.size, dtype=int)
        xlab = "Sample index"

    plt.figure(figsize=(10, 4))
    plt.plot(x, data)  # no explicit colors/styles
    plt.title(f"ADQ33 CH{ch} – Record {recno}")
    plt.xlabel(xlab)
    plt.ylabel("Amplitude (ADC counts)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="ADQ33 one-channel capture + save + plot (no external trigger needed)."
    )
    parser.add_argument("--device-index", type=int, default=None,
                        help="Device index to open (default: first detected).")
    parser.add_argument("--channel", type=int, default=0,
                        help="Channel index to capture (0=A, 1=B).")
    parser.add_argument("--record-length", type=int, default=1024,
                        help="Samples per record.")
    parser.add_argument("--records", type=int, default=5,
                        help="Number of records to capture for the selected channel.")
    parser.add_argument("--mode", choices=["software", "periodic"], default="software",
                        help="Trigger mode: 'software' (SWTrig) or 'periodic' (internal generator).")
    parser.add_argument("--periodic-freq", type=float, default=10_000.0,
                        help="Periodic event frequency in Hz (used when --mode periodic).")
    parser.add_argument("--sample-rate", type=float, default=None,
                        help="Sample rate in Hz for the plot time axis (optional).")
    parser.add_argument("--png-samples", type=int, default=4000,
                        help="How many samples to plot (per record). Use -1 for full record.")
    parser.add_argument("--base-dir", type=Path, default=Path("/external_ssd/adq_data"),
                        help="Base directory where timestamped runs are created.")
    args = parser.parse_args()

    # Timestamped output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.base_dir / ts
    outdir.mkdir(parents=True, exist_ok=True)

    # Control unit and device selection
    acu = pyadq.ADQControlUnit()
    acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, str(outdir))  # trace logs in the run folder
    devlist = acu.ListDevices()
    if not devlist:
        raise RuntimeError("No ADQ devices detected by pyadq. Check /dev/adq_* and permissions (adq group or sudo).")

    if args.device_index is not None:
        if args.device_index < 0 or args.device_index >= len(devlist):
            raise RuntimeError(f"Invalid --device-index {args.device_index}; {len(devlist)} device(s) found.")
        didx = args.device_index
    else:
        # Prefer an ADQ33 if enumerated, otherwise take first device
        didx = 0
        for i, d in enumerate(devlist):
            try:
                if d.ProductID == pyadq.PID_ADQ33:
                    didx = i
                    break
            except Exception:
                pass

    dev: pyadq.ADQ = acu.SetupDevice(didx)

    # Initialize parameter tree
    params: pyadq.ADQParameters = dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)
    nof_ch = int(params.constant.nof_channels)
    if args.channel < 0 or args.channel >= nof_ch:
        raise RuntimeError(f"Channel {args.channel} not available; device reports {nof_ch} channel(s).")

    active_channels = [args.channel]
    bytes_per_sample = params.constant.channel[args.channel].nof_bytes_per_sample

    # --- Acquisition: configure only the selected channel ---
    ch = args.channel
    params.acquisition.channel[ch].record_length = int(args.record_length)
    params.acquisition.channel[ch].nof_records = int(args.records)
    params.acquisition.channel[ch].trigger_edge = pyadq.ADQ_EDGE_RISING
    params.acquisition.channel[ch].trigger_source = (
        pyadq.ADQ_EVENT_SOURCE_SOFTWARE if args.mode == "software"
        else pyadq.ADQ_EVENT_SOURCE_PERIODIC
    )

    # --- Periodic generator (if used) ---
    if args.mode == "periodic":
        params.event_source.periodic.frequency = float(args.periodic_freq)

    # --- Transfer settings for the selected channel ---
    rec_buf_sz = args.record_length * bytes_per_sample
    rec_buf_sz = ceil_to_step(rec_buf_sz, params.constant.record_buffer_size_step)
    rec_buf_sz = min(rec_buf_sz, 512 * 1024)  # keep buffers modest on Jetson

    ptx = params.transfer.channel[ch]
    ptx.record_size = 0
    ptx.infinite_record_length_enabled = 0
    ptx.record_buffer_size = rec_buf_sz
    ptx.dynamic_record_length_enabled = 1
    ptx.metadata_enabled = 1
    ptx.metadata_buffer_size = pyadq.SIZEOF_ADQ_GEN4_HEADER
    ptx.nof_buffers = pyadq.ADQ_MAX_NOF_BUFFERS
    ptx.eject_buffer_source = pyadq.ADQ_FUNCTION_RECORD_STOP

    # Apply parameters
    dev.SetParameters(params)

    # Save run metadata
    meta = {
        "timestamp": ts,
        "device_index": didx,
        "channel": ch,
        "record_length": int(args.record_length),
        "records": int(args.records),
        "trigger_mode": args.mode,
        "periodic_frequency_hz": float(args.periodic_freq) if args.mode == "periodic" else None,
        "bytes_per_sample": int(bytes_per_sample),
        "sample_rate_hz": float(args.sample_rate) if args.sample_rate else None,
        "output_dir": str(outdir),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Start acquisition
    rc = dev.ADQ_StartDataAcquisition()
    if rc != pyadq.ADQ_EOK:
        raise RuntimeError(f"ADQ_StartDataAcquisition failed with code {rc} (see trace logs in {outdir})")

    # Fire software triggers if requested
    if args.mode == "software":
        for _ in range(args.records):
            dev.ADQ_SWTrig()

    print(f"[ADQ] Capturing CH{ch}: {args.records} record(s), length={args.record_length} samples, mode={args.mode}")

    # Read loop
    got = 0
    target = args.records  # only one channel
    while got < target:
        rec = dev.WaitForRecordBuffer(pyadq.ADQ_ANY_CHANNEL, 2000)  # 2s timeout
        rch = rec.header.channel
        recno = rec.header.record_number
        rlen = rec.header.record_length

        if rch != ch:
            # Shouldn't happen since only one channel is active, but guard anyway.
            dev.ReturnRecordBuffer(rch, rec)
            continue

        data = np.asarray(rec.data)

        base = f"ch{rch}_rec{recno:06d}"
        np.save(outdir / f"{base}.npy", data)
        data.astype(np.int16, copy=False).tofile(outdir / f"{base}.bin")

        png_samples = None if args.png_samples == -1 else int(args.png_samples)
        save_plot(
            data=data,
            outpath=outdir / f"{base}.png",
            ch=rch,
            recno=recno,
            sample_rate_hz=args.sample_rate,
            max_samples=png_samples,
        )

        print(f"[ADQ] Saved CH{rch} REC{recno} ({len(data)} samples): {base}.bin/.npy/.png")
        got += 1

    # Stop acquisition
    stop_rc = dev.ADQ_StopDataAcquisition()
    if stop_rc not in (pyadq.ADQ_EOK, pyadq.ADQ_EINTERRUPTED):
        raise RuntimeError(f"ADQ_StopDataAcquisition returned {stop_rc}")

    print(f"✅ Done. Files saved in: {outdir}")


if __name__ == "__main__":
    main()
