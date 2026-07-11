import csv, sys, pathlib, glob

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "results"

def compute(row):
    return round(float(row["mult_max"]), 3)

def convert(tokenizer: str):
    result = {}
    pattern = str(DATA / f"multipliers_{tokenizer}_vs_*.csv")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for row in csv.DictReader(f):
                script = row["script"]
                val = compute(row)
                if script not in result or val > result[script]:
                    result[script] = val
    pairs = [f"{s} = {v}" for s, v in result.items()]
    out = DATA / f"multipliers_{tokenizer}_worst.toml"
    out.write_text("m_prior = {" + ", ".join(pairs) + "}\n")
    print(f"Wrote {out}")

if __name__ == "__main__":
    tokenizer = sys.argv[1] if len(sys.argv) > 1 else "deepseek_v3"
    convert(tokenizer)
