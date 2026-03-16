import os
import csv
import random
import argparse
import sys

'''

csv2header.py goal:

Take N entrys from an CSV with the following structure:

timestamp,joy_x,joy_y,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24
<str>,<f32>,<f32>,<f32>,<f32>,<f32>,<f32>,<f32>,.......................................................,<f32>
.
.
.
<str>,<f32>,<f32>,<f32>,<f32>,<f32>,<f32>,<f32>,.......................................................,<f32>

Where the last Row is the Nth Row 


The rows are randomly swapped using a seed and an offset is calculated following the validation percent:
    
    e.g => val = 20% => train = (1.0 - val)

Process the entrys and output the following header file for C:

#ifndef GENANN_NN_DATASET

#define GENANN_NN_DATASET

#define DATASET_TRAIN_LEN  (<UINT>)
#define DATASET_VAL_LEN    (<UINT>)

// Offset To Apply in DatasetInput and DatasetOutput To Get The Validation Pairs
#define DATASET_VAL_OFFSET (<UINT>) 


typedef double[2]  InputVec;
typedef double[25] OutputVec;

const InputVec dataset_input[] = {
    (InputVec){<[0]joy_x>,<[0]joy_y>},
    .
    .
    .
    (InputVec){<[N]joy_x>,<[N]joy_y>},
}

const OutputVec dataset_output[] = {
    (OutputVec){<[0]c0>,...,<[0]c24>},
    .
    .
    .
    (OutputVec){<[N]c0>,...,<[N]c24>},
}

'''

OUTPUT_COLS = [f"c{i}" for i in range(25)]
INPUT_COLS  = ["joy_x", "joy_y"]


def parse_args():
    parser = argparse.ArgumentParser(
        prog="csv2header",
        description="Convert a joystick/channel CSV dataset into a C header file for GENANN."
    )
    parser.add_argument(
        "input",
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path for the output .h file. Defaults to <input_basename>.h"
    )
    parser.add_argument(
        "-n", "--num-rows",
        type=int,
        default=None,
        help="Number of rows to take from the CSV (default: all rows)."
    )
    parser.add_argument(
        "-v", "--val",
        type=float,
        default=0.2,
        help="Validation split fraction, e.g. 0.2 means 20%% validation (default: 0.2)."
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle rows (default: 42)."
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable row shuffling (keep original order)."
    )
    return parser.parse_args()


def load_csv(path: str, num_rows: int | None) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        # Validate required columns
        required = {"timestamp"} | set(INPUT_COLS) | set(OUTPUT_COLS)
        if reader.fieldnames is None:
            print("ERROR: CSV file is empty or has no header.", file=sys.stderr)
            sys.exit(1)
        missing = required - set(reader.fieldnames)
        if missing:
            print(f"ERROR: CSV is missing required columns: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)

        for i, row in enumerate(reader):
            if num_rows is not None and i >= num_rows:
                break
            rows.append(row)

    if not rows:
        print("ERROR: No data rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    return rows


def shuffle_rows(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled


def format_double(value: str) -> str:
    """Format a string float value as a C double literal."""
    try:
        return repr(float(value))
    except ValueError:
        print(f"ERROR: Could not parse float value: '{value}'", file=sys.stderr)
        sys.exit(1)


def generate_header(rows: list[dict], val_fraction: float, guard: str = "GENANN_NN_DATASET") -> str:
    total     = len(rows)
    val_len   = max(1, round(total * val_fraction)) if val_fraction > 0 else 0
    train_len = total - val_len
    # Training entries come first [0 .. train_len-1], validation entries follow [train_len .. total-1]
    val_offset = train_len

    lines = []

    # ── Header guard ──────────────────────────────────────────────────────────
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")

    # ── Defines ───────────────────────────────────────────────────────────────
    lines.append(f"#define DATASET_TRAIN_LEN  ({train_len}u)")
    lines.append(f"#define DATASET_VAL_LEN    ({val_len}u)")
    lines.append("")
    lines.append("// Offset To Apply in DatasetInput and DatasetOutput To Get The Validation Pairs")
    lines.append(f"#define DATASET_VAL_OFFSET ({val_offset}u)")
    lines.append("")

    # ── Typedefs ──────────────────────────────────────────────────────────────
    lines.append("typedef double InputVec[2];")
    lines.append("typedef double OutputVec[25];")
    lines.append("")

    # ── dataset_input ─────────────────────────────────────────────────────────
    lines.append("const InputVec dataset_input[] = {")
    for i, row in enumerate(rows):
        x = format_double(row["joy_x"])
        y = format_double(row["joy_y"])
        comma = "," if i < total - 1 else ""
        lines.append(f"    {{  {x}, {y}  }}{comma}")
    lines.append("};")
    lines.append("")

    # ── dataset_output ────────────────────────────────────────────────────────
    lines.append("const OutputVec dataset_output[] = {")
    for i, row in enumerate(rows):
        vals  = ", ".join(format_double(row[col]) for col in OUTPUT_COLS)
        comma = "," if i < total - 1 else ""
        lines.append(f"    {{ {vals} }}{comma}")
    lines.append("};")
    lines.append("")

    # ── DataPair ──────────────────────────────────────────────────────────────
    lines.append("typedef struct {")
    lines.append("    InputVec*  in;")
    lines.append("    OutputVec* out;")
    lines.append("} DataPair;")
    lines.append("")
    lines.append("#define GET_DATA_PAIR(IDX)     ((DataPair){&dataset_input[(IDX)],                &dataset_output[(IDX)]})")
    lines.append("#define GET_VAL_DATA_PAIR(IDX) ((DataPair){&dataset_input[(IDX)+DATASET_VAL_OFFSET], &dataset_output[(IDX)+DATASET_VAL_OFFSET]})")
    lines.append("")

    # ── End guard ─────────────────────────────────────────────────────────────
    lines.append(f"#endif /* {guard} */")
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()

    # Resolve output path
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = base + ".h"

    # Validate val fraction
    if not (0.0 <= args.val < 1.0):
        print("ERROR: --val must be in [0.0, 1.0).", file=sys.stderr)
        sys.exit(1)

    print(f"[csv2header] Reading  : {args.input}")
    rows = load_csv(args.input, args.num_rows)
    print(f"[csv2header] Loaded   : {len(rows)} rows")

    if not args.no_shuffle:
        rows = shuffle_rows(rows, args.seed)
        print(f"[csv2header] Shuffled : seed={args.seed}")

    total     = len(rows)
    val_len   = max(1, round(total * args.val)) if args.val > 0 else 0
    train_len = total - val_len
    print(f"[csv2header] Split    : train={train_len}  val={val_len}  (val={args.val*100:.1f}%)")

    header = generate_header(rows, args.val)

    with open(args.output, "w") as f:
        f.write(header)

    print(f"[csv2header] Written  : {args.output}")


if __name__ == "__main__":
    main()