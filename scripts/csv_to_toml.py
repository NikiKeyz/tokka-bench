import csv, sys, pathlib

def convert(raw: str):
    csv_path = pathlib.Path(raw)
    toml_path = csv_path.with_suffix(".toml")
    pairs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            script = row["script"]
            spread = float(row["mult_spread"])
            if spread > 1.1:
                val = round(float(row["mult_max"]) / 1.1, 3)
            else:
                val = round(float(row["mult_median"]), 3)
            pairs.append(f"{script} = {val}")
    toml_path.write_text("m_prior = {" + ", ".join(pairs) + "}\n")
    print(f"Wrote {toml_path}")

if __name__ == "__main__":
    convert(sys.argv[1])
