#!/usr/bin/env python3
"""
metric_listener.py — Receives training metrics from experiment.c over UART
and writes them to CSV files for offline analysis.

UART protocol (lines emitted by the firmware):
    M:BOOT
    M:START,<ai>,<name>,<hidden_layers>,<hidden>,<lr>,<passes>,<train_len>,<val_len>
    M:EP,<ai>,<ep>,<train_loss>,<val_loss>,<train_acc>,<val_acc>
    M:DONE,<ai>,<name>,<train_loss>,<val_loss>,<train_acc>,<val_acc>,<passes>
    M:SAVE,<ai>,<name>
    M:ALL_DONE

Output files (written to --output-dir):
    episodes.csv   — one row per epoch (M:EP lines)
    summary.csv    — one row per completed architecture (M:DONE lines)

Usage:
    python metric_listener.py --port COM3
    python metric_listener.py --port /dev/ttyUSB0 --baud 115200 --output-dir run_01/
    python metric_listener.py --list-ports
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Optional: coloured terminal output ────────────────────────────────────────
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

# ── Serial import ──────────────────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("pyserial not found.  Install it with:  pip install pyserial")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CSV writers
# ══════════════════════════════════════════════════════════════════════════════

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


def open_csv(path: Path, fields: list[str]):
    """Open (or append to) a CSV file, writing the header if the file is new."""
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


# ══════════════════════════════════════════════════════════════════════════════
# Listener state
# ══════════════════════════════════════════════════════════════════════════════

class ArchState:
    """Accumulates per-architecture metadata from M:START / M:DONE."""
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


class Listener:
    def __init__(self, ep_writer, sum_writer, default_passes = 30):
        self.ep_writer  = ep_writer
        self.sum_writer = sum_writer
        self.archs: dict[int, ArchState] = {}
        self.boot_time  = None
        self.all_done   = False
        self.num_passes = default_passes

    def handle(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("M:"):
            return

        ts  = datetime.now().isoformat(timespec="milliseconds")
        tag, _, rest = line[2:].partition(",")

        if   tag == "BOOT":      self._on_boot(ts)
        elif tag == "START":     self._on_start(ts, rest)
        elif tag == "EP":        self._on_ep(ts, rest)
        elif tag == "DONE":      self._on_done(ts, rest)
        elif tag == "SAVE":      self._on_save(rest)
        elif tag == "ALL_DONE":  self._on_all_done(ts)

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
        arch.saved         = False
        arch.ep_count      = 0
        self.archs[ai] = arch
        print(f"\n{BOLD}{YELLOW}[{ts}]  Arch {ai}: {arch.name}"
              f"  hl={arch.hidden_layers}  h={arch.hidden}"
              f"  lr={arch.lr}  passes={arch.passes}"
              f"  train={arch.train_len}  val={arch.val_len}{RESET}")

    def _on_ep(self, ts: str, rest: str) -> None:
        parts = rest.split(",")
        if len(parts) < 6:
            return
        ai = int(parts[0])
        ep = int(parts[1])
        tl = float(parts[2])
        vl = float(parts[3])
        ta = float(parts[4])
        va = float(parts[5])

        arch = self.archs.get(ai)
        if arch:
            arch.ep_count += 1
            arch.last_tl, arch.last_vl = tl, vl
            arch.last_ta, arch.last_va = ta, va

        row = {
            "timestamp":  ts,
            "arch_idx":   ai,
            "arch_name":  arch.name if arch else "",
            "hidden_layers": arch.hidden_layers if arch else 0,
            "hidden":     arch.hidden if arch else 0,
            "lr":         arch.lr if arch else 0.0,
            "episode":    ep,
            "train_loss": tl,
            "val_loss":   vl,
            "train_acc":  ta,
            "val_acc":    va,
        }
        self.ep_writer.writerow(row)

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
        ai     = int(parts[0])
        name   = parts[1]
        tl, vl = float(parts[2]), float(parts[3])
        ta, va = float(parts[4]), float(parts[5])
        passes = int(parts[6])

        arch = self.archs.get(ai, ArchState())
        arch.last_tl, arch.last_vl = tl, vl
        arch.last_ta, arch.last_va = ta, va

        print(f"\n{BOLD}{GREEN}[{ts}]  ✓ DONE  arch={name}"
              f"  finalValLoss={vl:.5f}  finalValAcc={va*100:.1f}%{RESET}")

        row = {
            "timestamp":       ts,
            "arch_idx":        ai,
            "arch_name":       name,
            "hidden_layers":   arch.hidden_layers,
            "hidden":          arch.hidden,
            "lr":              arch.lr,
            "passes":          passes,
            "final_train_loss": tl,
            "final_val_loss":   vl,
            "final_train_acc":  ta,
            "final_val_acc":    va,
            "saved":           False,   # updated in _on_save
        }
        # Store the row index so _on_save can patch it later.
        # Because csv.DictWriter has no random access, we track separately.
        self._pending_summary = (ai, row)
        # Write immediately (saved flag will be False; a second row is written
        # if M:SAVE follows — the analysis script can deduplicate by arch_idx).
        self.sum_writer.writerow(row)

    def _on_save(self, rest: str) -> None:
        parts = rest.split(",")
        if len(parts) < 2:
            return
        ai   = int(parts[0])
        name = parts[1]
        arch = self.archs.get(ai)
        if arch:
            arch.saved = True

        # Write a corrected summary row with saved=True
        row = {
            "timestamp":       datetime.now().isoformat(timespec="milliseconds"),
            "arch_idx":        ai,
            "arch_name":       name,
            "hidden_layers":   arch.hidden_layers if arch else 0,
            "hidden":          arch.hidden        if arch else 0,
            "lr":              arch.lr            if arch else 0.0,
            "passes":          arch.passes        if arch else 0,
            "final_train_loss": arch.last_tl      if arch else float("nan"),
            "final_val_loss":   arch.last_vl      if arch else float("nan"),
            "final_train_acc":  arch.last_ta      if arch else float("nan"),
            "final_val_acc":    arch.last_va      if arch else float("nan"),
            "saved": True,
        }
        self.sum_writer.writerow(row)
        print(f"\n{BOLD}{GREEN}  ↳ saved to flash  arch={name}{RESET}")

    def _on_all_done(self, ts: str) -> None:
        self.all_done = True
        print(f"\n{BOLD}{CYAN}[{ts}] ── All architectures complete ──{RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

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
    p.add_argument("--port",       default=None,     help="Serial port for the metric UART (uart1 / pin 4 adapter, NOT the USB debug port)")
    p.add_argument("--baud",       type=int, default=115200, help="Baud rate")
    p.add_argument("--output-dir", default=".",       help="Directory for output CSV files")
    p.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    p.add_argument("--timeout",    type=float, default=5.0 ,help="Serial read timeout (seconds)")
    p.add_argument("--echo",       action="store_true"     ,help="Echo all received lines to stdout")
    p.add_argument("--epochs",     type=int, default=30    ,help="Epoch Passes Used on current run")
    p.add_argument("--debug-echo", action="store_true",
                   help="Also print non-M: lines (useful if accidentally on the USB/stdio port)")
    p.add_argument("--stdio-only", action="store_true",
                   help="Port carries mixed stdio+metric lines (firmware mirrors M: to printf). "
                        "Non-M: lines are silently discarded. Use when you have only one "
                        "USB-serial adapter connected to the debug UART.")
    return p


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

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep_path  = out_dir / f"episodes_{run_ts}.csv"
    sum_path = out_dir / f"summary_{run_ts}.csv"

    ep_fh,  ep_writer  = open_csv(ep_path,  EPISODE_FIELDS)
    sum_fh, sum_writer = open_csv(sum_path, SUMMARY_FIELDS)

    print(f"{BOLD}metric_listener{RESET}  port={args.port}  baud={args.baud}")
    if args.stdio_only:
        print(f"  {YELLOW}stdio-only mode: filtering M: lines from mixed debug output{RESET}")
    print(f"  episodes → {ep_path}")
    print(f"  summary  → {sum_path}")
    print(f"  Ctrl-C to stop.\n")

    listener = Listener(ep_writer, sum_writer, args.epochs)
    consecutive_errors = 0

    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
        print(f"{GREEN}Opened {args.port}{RESET}\n")
    except serial.SerialException as exc:
        print(f"{RED}Cannot open {args.port}: {exc}{RESET}")
        sys.exit(1)

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
                elif getattr(args, 'debug_echo', False) and not line.startswith('M:'):
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


if __name__ == "__main__":
    main()