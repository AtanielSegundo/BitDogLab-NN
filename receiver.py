#!/usr/bin/env python3
"""
bitdoglab_logger.py
───────────────────
Lê linhas de inferência do BitDogLab via serial USB e salva em CSV.

Protocolo esperado do firmware:
  INF,<joy_x>,<joy_y>,<c0>,<c1>,...,<c24>\n

  • 28 campos por linha: tag "INF" + 2 joystick + 25 células
  • Valores float com 4 casas decimais
  • Demais linhas (debug "[BTN]", "[FLASH]", etc.) são ignoradas

Uso:
  python bitdoglab_logger.py                        # usa defaults abaixo
  python bitdoglab_logger.py --port COM5            # Windows
  python bitdoglab_logger.py --port /dev/ttyACM1   # Linux
  python bitdoglab_logger.py --port /dev/cu.usbmodem14101  # macOS
  python bitdoglab_logger.py --out experimento.csv --baud 115200

Colunas do CSV:
  timestamp   — ISO-8601 com microsegundos
  joy_x       — posição X do joystick normalizada [-1, 1]
  joy_y       — posição Y do joystick normalizada [-1, 1]
  c0 … c24    — saída de cada neurônio da camada de saída [0, 1]
               onde c(row*5 + col) corresponde ao pixel (row, col) da matriz 5×5

Dependências:
  pip install pyserial
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERRO] pyserial não encontrado.  Execute:  pip install pyserial")
    sys.exit(1)


# ── Constantes ────────────────────────────────────────────────────────────────

N_CELLS       = 25        # pixels da matriz 5×5
EXPECTED_COLS = 1 + 2 + N_CELLS   # tag + joy_x + joy_y + células = 28
PREFIX        = "DATA,"

CSV_HEADER = (
    ["timestamp", "joy_x", "joy_y"]
    + [f"p{i}" for i in range(N_CELLS)]
)

# Mapa de índice → (row, col) para referência no CSV de saída
#   c(row*5 + col)  →  pixel (row, col)  na matriz 5×5


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_ports():
    """Lista portas seriais disponíveis."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("[INFO] Nenhuma porta serial encontrada.")
        return
    print("[INFO] Portas disponíveis:")
    for p in ports:
        print(f"  {p.device:20s}  {p.description}")


def parse_line(raw: str):
    """
    Valida e parseia uma linha DATA.
    Retorna lista de strings [joy_x, joy_y, p0, ..., p24] ou None.
    """
    line = raw.strip()
    if not line.startswith(PREFIX):
        return None

    parts = line.split(",")
    if len(parts) != EXPECTED_COLS:
        return None

    try:
        [float(v) for v in parts[1:]]
    except ValueError:
        return None

    return parts[1:]   # descarta a tag "DATA"


def open_csv(path: Path):
    """
    Abre CSV em modo append.
    Escreve cabeçalho apenas se o arquivo estiver vazio ou não existir.
    """
    is_new = not path.exists() or path.stat().st_size == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(CSV_HEADER)
        f.flush()
        print(f"[CSV] Novo arquivo criado: {path}")
    else:
        print(f"[CSV] Appendando em: {path}")
    return f, writer


# ── Logger principal ──────────────────────────────────────────────────────────

def run_logger(port: str, baud: int, csv_path: Path,
               verbose: bool, max_rows: int | None):
    """
    Loop principal: lê serial → filtra → appenda CSV.

    Args:
        port      — porta serial (e.g. "COM3", "/dev/ttyACM0")
        baud      — baudrate (deve coincidir com o firmware)
        csv_path  — arquivo de saída
        verbose   — imprime cada linha recebida
        max_rows  — para após N amostras (None = infinito)
    """
    csv_file, writer = open_csv(csv_path)

    rows_written  = 0
    lines_skipped = 0
    t_start       = time.monotonic()
    last_report   = t_start

    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"[SER] Conectado em {port} @ {baud} baud")
        print(f"[SER] Aguardando linhas 'DATA,...'  (Ctrl+C para parar)\n")

        while True:
            raw = ser.readline()

            # Decodifica tolerando bytes ruins (firmware pode enviar lixo no boot)
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue

            values = parse_line(line)

            if values is None:
                # Linha de debug ou malformada — imprime se verbose
                debug = line.strip()
                if debug and verbose:
                    print(f"  [DBG] {debug}")
                lines_skipped += 1
                continue

            ts  = datetime.now().isoformat(timespec="microseconds")
            row = [ts] + values
            writer.writerow(row)
            csv_file.flush()
            rows_written += 1

            if verbose:
                jx, jy = values[0], values[1]
                active = [i for i, v in enumerate(values[2:]) if float(v) >= 0.5]
                print(f"[{rows_written:6d}] x={float(jx):+.3f} y={float(jy):+.3f}  "
                      f"pixels_ativos={active}")

            # Relatório periódico (a cada 5 s) mesmo sem verbose
            now = time.monotonic()
            if not verbose and (now - last_report) >= 5.0:
                rate = rows_written / (now - t_start)
                print(f"[STAT] {rows_written} amostras salvas  ({rate:.1f} Hz)")
                last_report = now

            if max_rows and rows_written >= max_rows:
                print(f"\n[INFO] Limite de {max_rows} amostras atingido.")
                break

    except serial.SerialException as e:
        print(f"\n[ERRO] Serial: {e}")
        print("[DICA] Verifique se a porta está correta e o cabo USB conectado.")
        sys.exit(1)
    except KeyboardInterrupt:
        elapsed = time.monotonic() - t_start
        rate    = rows_written / elapsed if elapsed > 0 else 0
        print(f"\n[INFO] Encerrado pelo usuário.")
        print(f"[STAT] {rows_written} amostras em {elapsed:.1f}s  ({rate:.1f} Hz)")
        print(f"[STAT] {lines_skipped} linhas ignoradas (debug / malformadas)")
    finally:
        csv_file.close()
        print(f"[CSV] Arquivo fechado: {csv_path}")
        try:
            ser.close()
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="BitDogLab — logger de inferência para CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--port",    default=None,
                    help="Porta serial (ex: COM3, /dev/ttyACM0). "
                         "Omita para listar portas disponíveis.")
    ap.add_argument("--baud",    type=int, default=115200,
                    help="Baudrate (padrão: 115200)")
    ap.add_argument("--out",     default="data.csv",
                    help="Arquivo CSV de saída (padrão: data.csv)")
    ap.add_argument("--verbose", action="store_true",
                    help="Imprime cada amostra no terminal")
    ap.add_argument("--max",     type=int, default=None, metavar="N",
                    help="Para após N amostras")
    ap.add_argument("--list",    action="store_true",
                    help="Lista portas disponíveis e sai")

    args = ap.parse_args()

    if args.list or args.port is None:
        list_ports()
        if args.port is None:
            print("\nUse --port <PORTA> para iniciar a captura.")
        sys.exit(0)

    run_logger(
        port     = args.port,
        baud     = args.baud,
        csv_path = Path(args.out),
        verbose  = args.verbose,
        max_rows = args.max,
    )


if __name__ == "__main__":
    main()