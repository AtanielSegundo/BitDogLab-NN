#!/usr/bin/env python3
"""
model_sender.py — Sends a .bin model file to evaluate.c on the Pico over USB CDC.

The .bin files are raw weight arrays (doubles) produced by metric_listener.py.
Architecture info is parsed from the filename (e.g. reluosig_l2x8_lr0.1.bin).

Usage:
    python model_sender.py --port COM3 --model train/first_models/reluosig_l2x8_lr0.1.bin
    python model_sender.py --port /dev/ttyACM0 --model sigosig_l4x16_lr0.05.bin
    python model_sender.py --list-ports
"""

import argparse
import base64
import re
import sys
import time
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("pyserial not found.  Install it with:  pip install pyserial")
    sys.exit(1)

CHUNK_RAW_BYTES = 45  # must match MODEL_CHUNK_BYTES in evaluate.c

ACT_MAP = {"sig": 0, "sigmoid": 0, "relu": 1, "lin": 2, "linear": 2}


def parse_arch_from_filename(name: str) -> dict:
    """
    Parse architecture from filename like:
        reluosig_l16x16_lr0.100000.bin
        sigosig_l2x8_lr0.1.bin
        linosig_l4x32_lr0.05.bin

    Pattern: <act_hidden>o<act_output>_l<layers>x<neurons>_lr<rate>.bin
    """
    stem = Path(name).stem  # strip .bin

    m = re.match(
        r"^(?P<ah>\w+?)o(?P<ao>\w+?)_l(?P<hl>\d+)x(?P<h>\d+)_lr(?P<lr>[\d.]+)$",
        stem,
    )
    if not m:
        return None

    ah_str = m.group("ah")
    ao_str = m.group("ao")

    if ah_str not in ACT_MAP or ao_str not in ACT_MAP:
        return None

    return {
        "hidden_layers": int(m.group("hl")),
        "hidden": int(m.group("h")),
        "act_hidden": ACT_MAP[ah_str],
        "act_output": ACT_MAP[ao_str],
        "lr": float(m.group("lr")),
    }


def wait_for_line(ser: serial.Serial, prefix: str, timeout: float = 10.0) -> str:
    """Read lines until one starts with `prefix` or timeout."""
    deadline = time.monotonic() + timeout
    buf = b""
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            time.sleep(0.01)
            continue
        buf += chunk
        while b"\n" in buf:
            line_bytes, buf = buf.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line:
                print(f"  << {line}")
            if line.startswith(prefix):
                return line
    return ""


def send_model(ser: serial.Serial, arch: dict, weight_data: bytes) -> bool:
    total_bytes = len(weight_data)

    # Wait for E:READY
    print("Waiting for device E:READY...")
    ready = wait_for_line(ser, "E:READY", timeout=30.0)
    if not ready:
        print("ERROR: Device did not send E:READY within 30s")
        return False

    # Send header
    header = (
        f"E:MODEL,{arch['hidden_layers']},{arch['hidden']},"
        f"{arch['act_hidden']},{arch['act_output']},{total_bytes}\r\n"
    )
    print(f"  >> {header.strip()}")
    ser.write(header.encode("utf-8"))

    ack = wait_for_line(ser, "E:ACK", timeout=5.0)
    if not ack:
        print("ERROR: No ACK for header")
        return False

    # Send chunks
    chunk_idx = 0
    for off in range(0, total_bytes, CHUNK_RAW_BYTES):
        chunk = weight_data[off : off + CHUNK_RAW_BYTES]
        b64 = base64.b64encode(chunk).decode("ascii")
        line = f"E:DATA,{chunk_idx},{b64}\r\n"
        ser.write(line.encode("utf-8"))
        chunk_idx += 1

        # Wait for per-chunk ACK
        ack = wait_for_line(ser, "E:ACK", timeout=5.0)
        if not ack:
            print(f"ERROR: No ACK for chunk {chunk_idx - 1}")
            return False

        if chunk_idx % 50 == 0:
            print(f"  ... sent {chunk_idx} chunks ({off + len(chunk)}/{total_bytes} bytes)")

    # Send END
    end_line = f"E:END,{chunk_idx}\r\n"
    print(f"  >> {end_line.strip()}")
    ser.write(end_line.encode("utf-8"))

    # Wait for LOAD_OK or LOAD_FAIL
    result = wait_for_line(ser, "E:LOAD_", timeout=10.0)
    if result.startswith("E:LOAD_OK"):
        print(f"\nModel uploaded successfully! ({chunk_idx} chunks, {total_bytes} bytes)")
        return True
    else:
        print(f"\nModel upload failed: {result}")
        return False


def list_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print(f"{'Port':<20} {'Description':<40} {'HWID'}")
    print("-" * 80)
    for p in ports:
        print(f"{p.device:<20} {p.description:<40} {p.hwid}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Send a .bin model to evaluate.c on the Pico.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", default=None, help="Serial port (e.g. COM3)")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate")
    p.add_argument("--model", default=None, help="Path to .bin model file")
    p.add_argument("--list-ports", action="store_true", help="List serial ports")
    # Manual arch override (if filename doesn't follow convention)
    p.add_argument("--hidden-layers", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--act-hidden", type=int, default=None,
                   help="0=sigmoid, 1=relu, 2=linear")
    p.add_argument("--act-output", type=int, default=None,
                   help="0=sigmoid, 1=relu, 2=linear")

    args = p.parse_args()

    if args.list_ports:
        list_ports()
        return

    if not args.port or not args.model:
        p.error("--port and --model are required")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        sys.exit(1)

    weight_data = model_path.read_bytes()
    print(f"Model file: {model_path}  ({len(weight_data)} bytes)")

    # Parse architecture from filename or CLI args
    arch = parse_arch_from_filename(model_path.name)

    if args.hidden_layers is not None:
        arch = arch or {}
        arch["hidden_layers"] = args.hidden_layers
    if args.hidden is not None:
        arch = arch or {}
        arch["hidden"] = args.hidden
    if args.act_hidden is not None:
        arch = arch or {}
        arch["act_hidden"] = args.act_hidden
    if args.act_output is not None:
        arch = arch or {}
        arch["act_output"] = args.act_output

    if not arch or not all(k in arch for k in ("hidden_layers", "hidden", "act_hidden", "act_output")):
        print(
            "ERROR: Could not parse architecture from filename.\n"
            "Use --hidden-layers, --hidden, --act-hidden, --act-output to specify manually.\n"
            "Expected filename pattern: <act_h>o<act_o>_l<layers>x<neurons>_lr<rate>.bin\n"
            "Example: reluosig_l2x8_lr0.1.bin"
        )
        sys.exit(1)

    act_names = {0: "sigmoid", 1: "relu", 2: "linear"}
    print(f"Architecture: hl={arch['hidden_layers']} h={arch['hidden']} "
          f"act_hidden={act_names.get(arch['act_hidden'], '?')} "
          f"act_output={act_names.get(arch['act_output'], '?')}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
        print(f"Opened {args.port}\n")
    except serial.SerialException as exc:
        print(f"Cannot open {args.port}: {exc}")
        sys.exit(1)

    try:
        success = send_model(ser, arch, weight_data)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
