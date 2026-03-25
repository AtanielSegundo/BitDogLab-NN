#!/usr/bin/env python3
"""
metric_listener.py — Receives training metrics from experiment.c over UART
and writes them to CSV files for offline analysis.

UART protocol (lines emitted by the firmware):
    M:BOOT
    M:START,<ai>,<n>,<hidden_layers>,<hidden>,<lr>,<passes>,<train_len>,<val_len>
    M:EP,<ai>,<ep>,<train_loss>,<val_loss>,<train_acc>,<val_acc>
    M:DONE,<ai>,<n>,<train_loss>,<val_loss>,<train_acc>,<val_acc>,<passes>
    M:MODEL_START,<ai>,<n>,<total_bytes>
    M:MODEL_DATA,<ai>,<chunk_idx>,<base64_payload>
    M:MODEL_END,<ai>,<n>,<total_chunks>
    M:ALL_DONE

Sync protocol (--synch flag):
    host   → device :  M:SYNCH,<last_arch_idx>,<last_episode>
    device → host   :  M:SYNCH_ACK,<last_completed_arch>

Output files (written to --output-dir):
    episodes_<tag>.csv       — one row per epoch (M:EP lines)
    summary_<tag>.csv        — one row per completed architecture (M:DONE lines)
    <tag>_models/<n>.bin     — raw weight array (doubles) for every finished arch

Usage:
    python metric_listener.py --port COM3
    python metric_listener.py --port /dev/ttyUSB0 --baud 115200 --output-dir run_01/
    python metric_listener.py --port COM3 --custom_tag run_01 --synch
    python metric_listener.py --list-ports
"""

import argparse
import base64
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import colorama
    colorama.init(autoreset=True)
    def _c(code): return f"\033[{code}m"
    RESET  = _c(0)
    BOLD   = _c(1)
    CYAN   = _c(36)
    GREEN  = _c(32)
    YELLOW = _c(33)
    RED    = _c(31)
    DIM    = _c(2)
except ImportError:
    RESET = BOLD = CYAN = GREEN = YELLOW = RED = DIM = ""

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("pyserial not found.  Install it with:  pip install pyserial")
    sys.exit(1)

EPISODE_FIELDS = [
    "timestamp", "arch_idx", "arch_name",
    "hidden_layers", "hidden", "lr",
    "episode", "train_loss", "val_loss", "train_acc", "val_acc",
]

SUMMARY_FIELDS = [
    "timestamp", "arch_idx", "arch_name",
    "hidden_layers", "hidden", "lr", "passes",
    "final_train_loss", "final_val_loss",
    "final_train_acc",  "final_val_acc",
    "saved",
]

EPISODE_KEY_FIELDS = ("arch_idx", "episode")
SUMMARY_KEY_FIELDS = ("arch_idx", "saved")


# ---------------------------------------------------------------------------
# Deduplicating CSV writer
# ---------------------------------------------------------------------------

class DeduplicatingWriter:
    """
    Wraps csv.DictWriter and skips rows whose key-tuple was already seen,
    either loaded from an existing file or written in the current session.
    """

    def __init__(self, writer: csv.DictWriter, key_fields: tuple,
                 existing_keys: set | None = None):
        self._writer     = writer
        self._key_fields = key_fields
        self._seen       = existing_keys if existing_keys is not None else set()
        self.skipped     = 0

    def _key(self, row: dict) -> tuple:
        return tuple(str(row.get(f, "")) for f in self._key_fields)

    def writerow(self, row: dict) -> bool:
        key = self._key(row)
        if key in self._seen:
            label = dict(zip(self._key_fields, key))
            print(f"\n{YELLOW}  ↷ skipped duplicate row {label}{RESET}")
            self.skipped += 1
            return False
        self._seen.add(key)
        self._writer.writerow(row)
        return True

    def writeheader(self) -> None:
        self._writer.writeheader()


def _load_existing_keys(path: Path, key_fields: tuple) -> set:
    keys: set = set()
    if not path.exists() or path.stat().st_size == 0:
        return keys
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = tuple(str(row.get(f, "")) for f in key_fields)
                keys.add(key)
        print(f"{DIM}  Loaded {len(keys)} existing keys from {path.name}{RESET}")
    except Exception as exc:
        print(f"{YELLOW}  Warning: could not read {path}: {exc}{RESET}")
    return keys


def open_csv(path: Path, fields: list[str], key_fields: tuple) -> tuple:
    existing_keys = _load_existing_keys(path, key_fields)
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    raw_writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if is_new:
        raw_writer.writeheader()
        fh.flush()
    writer = DeduplicatingWriter(raw_writer, key_fields, existing_keys)
    return fh, writer


# ---------------------------------------------------------------------------
# Sync helpers
# ---------------------------------------------------------------------------

def _find_last_progress(ep_path: Path) -> tuple[int, int]:
    """
    Scan the episodes CSV and return (last_arch_idx, last_episode_for_that_arch).
    Returns (-1, -1) if the file is empty or missing.

    Strategy: find the highest arch_idx in the file, then find the highest
    episode number recorded for that specific arch.  This is what gets sent
    as M:SYNCH so the firmware knows exactly where to resume.
    """
    if not ep_path.exists() or ep_path.stat().st_size == 0:
        return -1, -1

    last_ai = -1
    last_ep = -1
    try:
        with open(ep_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    ai = int(row["arch_idx"])
                    ep = int(row["episode"])
                except (KeyError, ValueError):
                    continue
                # Track the overall highest arch, and within that arch
                # the highest episode.
                if ai > last_ai or (ai == last_ai and ep > last_ep):
                    if ai > last_ai:
                        last_ep = ep          # reset ep counter for new arch
                    else:
                        last_ep = max(last_ep, ep)
                    last_ai = ai
    except Exception as exc:
        print(f"{YELLOW}  Warning: could not scan {ep_path}: {exc}{RESET}")
        return -1, -1

    return last_ai, last_ep


def send_synch(ser: "serial.Serial", ep_path: Path, ack_timeout: float = 6.0) -> bool:
    """
    Read the last recorded (arch_idx, episode) from ep_path and write
    M:SYNCH,<ai>,<ep> to the serial port.

    Then waits up to ack_timeout seconds for M:SYNCH_ACK from the firmware
    to confirm the message landed.  Returns True on ACK, False on timeout.
    """
    last_ai, last_ep = _find_last_progress(ep_path)

    if last_ai < 0:
        print(f"{YELLOW}  --synch: no existing progress found in {ep_path.name}"
              f" — sending cold-start signal{RESET}")
        # Send a cold-start marker so the firmware still exits the synch window quickly
        last_ai, last_ep = -1, -1

    msg = f"M:SYNCH,{last_ai},{last_ep}\r\n"
    ser.write(msg.encode("utf-8"))
    print(f"{CYAN}  → sent {msg.strip()}  (waiting for ACK…){RESET}")

    # Drain lines until we see M:SYNCH_ACK or timeout
    deadline = time.monotonic() + ack_timeout
    buf = b""
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            time.sleep(0.02)
            continue
        buf += chunk
        while b"\n" in buf:
            line_bytes, buf = buf.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line.startswith("M:SYNCH_ACK"):
                parts = line.split(",", 1)
                acked_arch = int(parts[1]) if len(parts) > 1 else "?"
                print(f"{BOLD}{GREEN}  ✓ SYNCH_ACK  last_completed_arch={acked_arch}{RESET}")
                return True
            # Echo any non-ACK lines so nothing is swallowed
            if line:
                print(f"{DIM}  (pre-synch) {line}{RESET}")

    print(f"{RED}  ✗ No M:SYNCH_ACK received within {ack_timeout:.0f}s{RESET}")
    return False


# ---------------------------------------------------------------------------
# Per-architecture state
# ---------------------------------------------------------------------------

class ArchState:
    def __init__(self):
        self.name          = ""
        self.hidden_layers = 0
        self.hidden        = 0
        self.lr            = 0.0
        self.passes        = 0
        self.train_len     = 0
        self.val_len       = 0
        self.saved         = False
        self.ep_count      = 0
        self.last_tl       = float("nan")
        self.last_vl       = float("nan")
        self.last_ta       = float("nan")
        self.last_va       = float("nan")


# ---------------------------------------------------------------------------
# Main listener
# ---------------------------------------------------------------------------

class Listener:
    def __init__(self, ep_writer: DeduplicatingWriter,
                 sum_writer: DeduplicatingWriter,
                 out_dir: Path,
                 models_dir: Path,
                 default_passes: int = 30):
        self.ep_writer  = ep_writer
        self.sum_writer = sum_writer
        self.out_dir    = out_dir
        self.models_dir = models_dir
        self.archs: dict[int, ArchState] = {}
        self.boot_time  = None
        self.all_done   = False
        self.num_passes = default_passes
        self._model_buf: dict[int, list[str]] = {}

    def handle(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("M:"):
            return

        ts  = datetime.now().isoformat(timespec="milliseconds")
        tag, _, rest = line[2:].partition(",")

        if   tag == "BOOT":         self._on_boot(ts)
        elif tag == "START":        self._on_start(ts, rest)
        elif tag == "EP":           self._on_ep(ts, rest)
        elif tag == "DONE":         self._on_done(ts, rest)
        elif tag == "SAVE":         self._on_save(rest)
        elif tag == "MODEL_START":  self._on_model_start(rest)
        elif tag == "MODEL_DATA":   self._on_model_data(rest)
        elif tag == "MODEL_END":    self._on_model_end(rest)
        elif tag == "SYNCH_ACK":    pass   # already handled in send_synch()
        elif tag == "ALL_DONE":     self._on_all_done(ts)

    def _on_boot(self, ts: str) -> None:
        self.boot_time = ts
        self.all_done  = False
        print(f"\n{BOLD}{CYAN}[{ts}] ── Device booted ──────────────────{RESET}")

    def _on_start(self, ts: str, rest: str) -> None:
        parts = rest.split(",")
        if len(parts) < 8:
            return
        ai   = int(parts[0])
        arch = ArchState()
        arch.name          = parts[1]
        arch.hidden_layers = int(parts[2])
        arch.hidden        = int(parts[3])
        arch.lr            = float(parts[4])
        arch.passes        = int(parts[5])
        arch.train_len     = int(parts[6])
        arch.val_len       = int(parts[7])
        self.archs[ai] = arch
        print(f"\n{BOLD}{YELLOW}[{ts}]  Arch {ai}: {arch.name}"
              f"  hl={arch.hidden_layers}  h={arch.hidden}"
              f"  lr={arch.lr}  passes={arch.passes}"
              f"  train={arch.train_len}  val={arch.val_len}{RESET}")

    def _on_ep(self, ts: str, rest: str) -> None:
        parts = rest.split(",")
        if len(parts) < 6:
            return
        ai = int(parts[0]);  ep = int(parts[1])
        tl = float(parts[2]); vl = float(parts[3])
        ta = float(parts[4]); va = float(parts[5])

        arch = self.archs.get(ai)
        if arch:
            arch.ep_count += 1
            arch.last_tl, arch.last_vl = tl, vl
            arch.last_ta, arch.last_va = ta, va

        row = {
            "timestamp":     ts,
            "arch_idx":      ai,
            "arch_name":     arch.name          if arch else "",
            "hidden_layers": arch.hidden_layers if arch else 0,
            "hidden":        arch.hidden        if arch else 0,
            "lr":            arch.lr            if arch else 0.0,
            "episode":       ep,
            "train_loss":    tl, "val_loss": vl,
            "train_acc":     ta, "val_acc":  va,
        }
        written = self.ep_writer.writerow(row)

        if written:
            bar_len = 20
            passes  = arch.passes if arch else self.num_passes
            filled  = int(bar_len * (ep + 1) / passes)
            bar     = "█" * filled + "░" * (bar_len - filled)
            name    = arch.name if arch else f"arch{ai}"
            sys.stdout.write(
                f"\r{DIM}[{ts}]{RESET}  {CYAN}{name:12s}{RESET}  "
                f"ep {ep+1:>3}/{passes}  [{bar}]  "
                f"tL={tl:.4f}  vL={GREEN}{vl:.4f}{RESET}  "
                f"tA={ta*100:.1f}%  vA={va*100:.1f}%   "
            )
            sys.stdout.flush()

    def _on_done(self, ts: str, rest: str) -> None:
        parts = rest.split(",")
        if len(parts) < 7:
            return
        ai     = int(parts[0]);  name = parts[1]
        tl, vl = float(parts[2]), float(parts[3])
        ta, va = float(parts[4]), float(parts[5])
        passes = int(parts[6])

        arch = self.archs.get(ai, ArchState())
        arch.last_tl, arch.last_vl = tl, vl
        arch.last_ta, arch.last_va = ta, va

        print(f"\n{BOLD}{GREEN}[{ts}]  ✓ DONE  arch={name}"
              f"  finalValLoss={vl:.5f}  finalValAcc={va*100:.1f}%{RESET}")

        self.sum_writer.writerow({
            "timestamp": ts, "arch_idx": ai, "arch_name": name,
            "hidden_layers": arch.hidden_layers, "hidden": arch.hidden, "lr": arch.lr,
            "passes": passes,
            "final_train_loss": tl, "final_val_loss": vl,
            "final_train_acc":  ta, "final_val_acc":  va,
            "saved": False,
        })

    def _on_save(self, rest: str) -> None:
        parts = rest.split(",")
        if len(parts) < 2:
            return
        ai = int(parts[0]);  name = parts[1]
        arch = self.archs.get(ai)
        if arch:
            arch.saved = True
        self.sum_writer.writerow({
            "timestamp":        datetime.now().isoformat(timespec="milliseconds"),
            "arch_idx":         ai, "arch_name": name,
            "hidden_layers":    arch.hidden_layers if arch else 0,
            "hidden":           arch.hidden        if arch else 0,
            "lr":               arch.lr            if arch else 0.0,
            "passes":           arch.passes        if arch else 0,
            "final_train_loss": arch.last_tl       if arch else float("nan"),
            "final_val_loss":   arch.last_vl       if arch else float("nan"),
            "final_train_acc":  arch.last_ta       if arch else float("nan"),
            "final_val_acc":    arch.last_va       if arch else float("nan"),
            "saved": True,
        })
        print(f"\n{BOLD}{GREEN}  ↳ saved to flash  arch={name}{RESET}")

    def _on_model_start(self, rest: str) -> None:
        parts = rest.split(",", 3)
        if len(parts) < 3:
            return
        ai = int(parts[0]);  name = parts[1];  total_bytes = int(parts[2])
        self._model_buf[ai] = []
        print(f"\n{DIM}  ↓ model transfer start  arch={name}"
              f"  expected={total_bytes} bytes{RESET}")

    def _on_model_data(self, rest: str) -> None:
        parts = rest.split(",", 3)
        if len(parts) < 3:
            return
        ai = int(parts[0])
        b64_data = parts[2].strip()
        if ai not in self._model_buf:
            self._model_buf[ai] = []
        self._model_buf[ai].append(b64_data)

    def _on_model_end(self, rest: str) -> None:
        parts = rest.split(",", 3)
        if len(parts) < 2:
            return
        ai   = int(parts[0]);  name = parts[1]
        expected_chunks = int(parts[2]) if len(parts) > 2 else None

        chunks = self._model_buf.pop(ai, [])
        if expected_chunks is not None and len(chunks) != expected_chunks:
            print(f"\n{RED}  ✗ model transfer incomplete  arch={name}"
                  f"  got {len(chunks)}/{expected_chunks} chunks — NOT saved{RESET}")
            return
        try:
            raw = base64.b64decode("".join(chunks))
        except Exception as exc:
            print(f"\n{RED}  ✗ base64 decode error  arch={name}: {exc}{RESET}")
            return

        out_path = self.models_dir / f"{name}.bin"
        if not out_path.exists():
            out_path.write_bytes(raw)
            print(f"\n{BOLD}{GREEN}  ↓ model saved  {out_path}  ({len(raw)} bytes){RESET}")
        else:
            print(f"\n{DIM}  ↷ model already exists  {out_path}{RESET}")

    def _on_all_done(self, ts: str) -> None:
        self.all_done = True
        print(f"\n{BOLD}{CYAN}[{ts}] ── All architectures complete ──{RESET}\n")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def list_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print(f"{'Port':<20} {'Description':<40} {'HWID'}")
    print("-" * 80)
    for p in ports:
        print(f"{p.device:<20} {p.description:<40} {p.hwid}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="metric_listener",
        description="Record NN training metrics from experiment.c over UART.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port",       default=None,             help="Serial port for the metric UART")
    p.add_argument("--baud",       type=int, default=115200, help="Baud rate")
    p.add_argument("--output_dir", "-o", default=".",        help="Directory for output CSV files")
    p.add_argument("--custom_tag", "-ctag", default=None,    help="Custom tag for metric listening")
    p.add_argument("--list_ports", action="store_true",      help="List available serial ports and exit")
    p.add_argument("--timeout",    type=float, default=5.0,  help="Serial read timeout (seconds)")
    p.add_argument("--echo",       action="store_true",      help="Echo all received lines to stdout")
    p.add_argument("--epochs",     type=int, default=30,     help="Epoch passes used on current run")
    p.add_argument("--debug_echo", action="store_true",      help="Also print non-M: lines")
    p.add_argument("--synch",      action="store_true",
                   help="Send M:SYNCH with last recorded progress so the "
                        "firmware resumes instead of restarting from arch 0")
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()

    if args.list_ports:
        list_ports()
        return

    if args.port is None:
        print("Error: --port is required (use --list-ports to discover available ports).")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_ts = args.custom_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    ep_path  = out_dir / f"episodes_{run_ts}.csv"
    sum_path = out_dir / f"summary_{run_ts}.csv"

    models_dir = out_dir / f"{run_ts}_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ep_fh,  ep_writer  = open_csv(ep_path,  EPISODE_FIELDS, EPISODE_KEY_FIELDS)
    sum_fh, sum_writer = open_csv(sum_path, SUMMARY_FIELDS, SUMMARY_KEY_FIELDS)

    print(f"{BOLD}metric_listener{RESET}  port={args.port}  baud={args.baud}")
    print(f"  episodes → {ep_path}")
    print(f"  summary  → {sum_path}")
    print(f"  models   → {models_dir}/")
    if args.synch:
        print(f"  {CYAN}--synch enabled{RESET}")
    print(f"  Ctrl-C to stop.\n")

    listener = Listener(ep_writer, sum_writer, out_dir, models_dir, args.epochs)
    consecutive_errors = 0

    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
        print(f"{GREEN}Opened {args.port}{RESET}\n")
    except serial.SerialException as exc:
        print(f"{RED}Cannot open {args.port}: {exc}{RESET}")
        sys.exit(1)

    # ── Send M:SYNCH before entering the main read loop ─────────────────────
    # The firmware waits SYNCH_WAIT_MS (5 s) for this message right after boot,
    # so we send it immediately after opening the port.
    if args.synch:
        send_synch(ser, ep_path)

    try:
        while True:
            try:
                raw = ser.readline()
                if not raw:
                    continue

                consecutive_errors = 0
                line = raw.decode("utf-8", errors="replace").rstrip()

                if args.echo:
                    print(f"{DIM}>> {line}{RESET}")
                elif getattr(args, "debug_echo", False) and not line.startswith("M:"):
                    print(f"{DIM}   {line}{RESET}")

                listener.handle(line)

                if line.startswith("M:"):
                    ep_fh.flush()
                    sum_fh.flush()

            except serial.SerialException as exc:
                consecutive_errors += 1
                print(f"\n{RED}Serial error: {exc}{RESET}")
                if consecutive_errors >= 5:
                    print(f"{RED}Too many consecutive errors — exiting.{RESET}")
                    break
                time.sleep(0.5)

            except UnicodeDecodeError:
                pass

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted by user.{RESET}")
    finally:
        ser.close()
        ep_fh.close()
        sum_fh.close()
        print(f"\n{BOLD}Saved:{RESET}")
        print(f"  {ep_path}   ({ep_path.stat().st_size} bytes)")
        print(f"  {sum_path}  ({sum_path.stat().st_size} bytes)")
        if ep_writer.skipped or sum_writer.skipped:
            print(f"{YELLOW}  Skipped {ep_writer.skipped} duplicate episode row(s), "
                  f"{sum_writer.skipped} duplicate summary row(s).{RESET}")
        if listener._model_buf:
            pending = [str(ai) for ai in listener._model_buf]
            print(f"{YELLOW}  Warning: incomplete model transfer for arch(es): "
                  f"{', '.join(pending)}{RESET}")


if __name__ == "__main__":
    main()