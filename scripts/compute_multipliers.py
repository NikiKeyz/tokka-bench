"""
Compute per-script token-count multipliers between a base tokenizer (available
in your app) and a target tokenizer (the real model's tokenizer).

The multiplier for a script is:

    mult = base_bytes_per_token / target_bytes_per_token

so that, in your app:

    target_tokens ~= base_tokens * overall_multiplier
    overall_multiplier = sum(share_i * mult_i)

where ``share_i`` is the fraction of the text in script ``i`` (detected e.g. via
regex) and ``base_tokens`` is the count produced by running the text through the
base tokenizer.

Multipliers can be < 1 (target more efficient) or > 1 (target less efficient).
For each script the script reports min / median / max across its languages:

  * ``mult_median`` — central estimate.
  * ``mult_max``    — worst case (largest ratio); use this for a conservative,
                      overflow-proof estimate (never under-counts tokens).

Example (ratio mode, with a base tokenizer):
    uv run python scripts/compute_multipliers.py \
        --base Qwen/Qwen3-8B --target tencent/Hy3

Batch mode (one target, every tiktoken_* base tokenizer)
--------------------------------------------------------
With --batch, a single --target is compared against every ``tiktoken_*.json``
result file found in the results directory. One per-base multiplier CSV is
written for each base, plus a combined cross-base table:

    uv run python scripts/compute_multipliers.py --batch --target tencent/Hy3

The per-base multiplier rows include ``mult_spread = mult_max / mult_median`` -- a
measure of how variable the per-language multipliers are within a script
(1.0 = uniform, larger = more uneven). The cross-base spread output has two
tables, both restricted to scripts with ``n_langs > 1`` (single-language
scripts trivially have a spread of 1.0 and are excluded):

  * Per tokenizer: one row per base reports ``spread_min`` / ``spread_median`` /
    ``spread_max`` of its per-script spreads, sorted by ``spread_max`` ascending
    -- the tokenizer with the smallest worst-case (maximum) per-script spread
    appears first.
  * Per script: one row per script lists the ``mult_spread`` seen under every
    base tokenizer, plus per-script aggregates (min / median / max across bases).

Both are written to CSV (``spread_<target>_vs_tiktoken_bases.csv`` and
``..._by_script.csv``).

Target-only (no base tokenizer) safety mode
-------------------------------------------
If you do NOT run a base tokenizer, you only know script shares (e.g. via
regex) and must assume the worst language in each script. The per-script
quantity you then need is the TARGET's own worst-case bytes-per-token: the
minimum across that script's languages (rarest/least-supported languages tend
to land here, but the benchmark measures it directly -- do not assume by
rarity). Then:

    target_tokens ~= total_bytes * sum(share_i * (1 / target_min_bpt_i))

where total_bytes = len(text.encode()). Run with only --target:

    uv run python scripts/compute_multipliers.py --target tencent/Hy3
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "results"


def normalize(name: str) -> str:
    """Turn a tokenizer tag into the result-file stem (e.g. Qwen/Qwen3-8B)."""
    return name.strip().replace("/", "_").replace("-", "_")


def load_result(tokenizer: str, results_dir: Path) -> dict:
    """Load a benchmark result JSON by tag or by exact filename stem."""
    stem = normalize(tokenizer)
    candidates = [
        results_dir / f"{stem}.json",
        results_dir / f"{tokenizer}.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No result file for '{tokenizer}'. Looked for: "
        + ", ".join(str(c) for c in candidates)
    )


def script_language_bpt(result: dict) -> dict:
    """Map (script, lang_key) -> bytes_per_token for one result file."""
    out = {}
    for lang_key, lang_data in result.get("languages", {}).items():
        script = lang_data.get("language_info", {}).get("script")
        bpt = lang_data.get("metrics", {}).get("bytes_per_token")
        if script is None or bpt is None:
            continue
        out[(script, lang_key)] = bpt
    return out


def compute_multipliers(base_result: dict, target_result: dict) -> list:
    """Return per-script multiplier rows comparing base vs target.

    Each row: script, n_langs, mult_min, mult_median, mult_max,
    mult_spread (mult_max / mult_median, a measure of how variable the per-language
    multipliers are within the script), worst_lang (lang with the max ratio),
    worst_ratio, base_median_bpt, target_median_bpt.
    """
    base_map = script_language_bpt(base_result)
    target_map = script_language_bpt(target_result)

    # Group languages by script, keeping only langs present in BOTH tokenizers
    by_script: dict = {}
    for (script, lang_key), bpt in base_map.items():
        if (script, lang_key) in target_map:
            by_script.setdefault(script, []).append(lang_key)

    rows = []
    for script in sorted(by_script.keys()):
        ratios = []
        base_vals = []
        target_vals = []
        worst_lang = None
        worst_ratio = float("-inf")
        for lang_key in by_script[script]:
            b = base_map[(script, lang_key)]
            t = target_map[(script, lang_key)]
            if t <= 0:
                continue
            r = b / t
            ratios.append(r)
            base_vals.append(b)
            target_vals.append(t)
            if r > worst_ratio:
                worst_ratio = r
                worst_lang = lang_key
        if not ratios:
            continue
        mmax = max(ratios)
        rows.append(
            {
                "script": script,
                "n_langs": len(ratios),
                "mult_min": min(ratios),
                "mult_median": statistics.median(ratios),
                "mult_max": mmax,
                "mult_spread": (mmax / statistics.median(ratios)) if statistics.median(ratios) > 0 else float("inf"),
                "worst_lang": worst_lang,
                "worst_ratio": worst_ratio,
                "base_median_bpt": statistics.median(base_vals),
                "target_median_bpt": statistics.median(target_vals),
            }
        )
    return rows


def compute_target_safety(target_result: dict) -> list:
    """Per-script worst-case table for the TARGET tokenizer alone.

    Used when no base tokenizer is run: only script shares (regex) are known,
    so the worst language in each script must be assumed. For each script the
    row reports the minimum bytes-per-token (worst language), the corresponding
    tokens-per-byte (1 / min_bpt, the conservative tokens/byte to use), plus the
    median for reference.

    Plug into: target_tokens ~= total_bytes * sum(share_i * tokens_per_byte_worst_i)
    """
    tmap = script_language_bpt(target_result)
    by_script: dict = {}
    for (script, lang_key), bpt in tmap.items():
        by_script.setdefault(script, []).append(lang_key)

    rows = []
    for script in sorted(by_script.keys()):
        vals = [(lang_key, tmap[(script, lang_key)]) for lang_key in by_script[script]]
        if not vals:
            continue
        bpts = [v for _, v in vals]
        worst_lang, min_bpt = min(vals, key=lambda x: x[1])
        rows.append(
            {
                "script": script,
                "n_langs": len(vals),
                "worst_lang": worst_lang,
                "worst_bytes_per_token": min_bpt,
                "tokens_per_byte_worst": 1.0 / min_bpt if min_bpt > 0 else float("inf"),
                "median_bytes_per_token": statistics.median(bpts),
            }
        )
    return rows


def find_tiktoken_bases(results_dir: Path) -> list:
    """Return sorted list of (file_path, tokenizer_name) for tiktoken_* result files."""
    bases = []
    for path in sorted(results_dir.glob("tiktoken_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("tokenizer", path.stem)
        except (json.JSONDecodeError, OSError):
            name = path.stem
        bases.append((path, name))
    return bases


def write_multiplier_csv(rows: list, out_path: Path) -> None:
    """Write ratio-mode per-script rows to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "script",
                "n_langs",
                "mult_min",
                "mult_median",
                "mult_max",
                "mult_spread",
                "worst_lang",
                "worst_ratio",
                "base_median_bpt",
                "target_median_bpt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_cross_base_table(all_rows: list) -> tuple:
    """Build cross-base spread tables from per-base multiplier rows.

    ``all_rows`` is a list of ``(base_name, rows)`` where each ``rows`` is the
    output of :func:`compute_multipliers`. For each base tokenizer we collect the
    per-script ``mult_spread`` (mult_max / mult_median), but ONLY for scripts with
    ``n_langs > 1`` (a single-language script has a trivial spread of 1 and is
    meaningless to compare).

    Two tables are returned:

    * ``by_tokenizer`` -- one row per base tokenizer, aggregating the per-script
      spreads across all qualifying scripts::

          base, n_scripts, spread_min, spread_median, spread_max

      sorted by ``spread_max`` ascending, so the tokenizer with the smallest
      worst-case (maximum) spread across scripts appears first.
    * ``by_script`` -- one row per script, listing the ``mult_spread`` seen under
      every base tokenizer plus per-script aggregates (min / median / max across
      bases)::

          script, n_bases, <one column per base>, spread_min, spread_median, spread_max
    """
    base_names = [name for name, _ in all_rows]

    # Per-tokenizer aggregation across qualifying scripts.
    by_tokenizer = []
    # Per-script spread per base: script -> {base_name: spread}
    by_script: dict = {}
    for base_name, rows in all_rows:
        spreads = []
        for r in rows:
            if r["n_langs"] <= 1 or r["mult_spread"] == float("inf"):
                continue
            spread = r["mult_spread"]
            spreads.append(spread)
            by_script.setdefault(r["script"], {})[base_name] = spread
        if spreads:
            by_tokenizer.append(
                {
                    "base": base_name,
                    "n_scripts": len(spreads),
                    "spread_min": min(spreads),
                    "spread_median": statistics.median(spreads),
                    "spread_max": max(spreads),
                }
            )
    by_tokenizer.sort(key=lambda r: r["spread_max"])

    by_script_rows = []
    for script in sorted(by_script.keys()):
        per_base = by_script[script]
        vals = list(per_base.values())
        by_script_rows.append(
            {
                "script": script,
                "n_bases": len(per_base),
                "per_base": per_base,
                "spread_min": min(vals),
                "spread_median": statistics.median(vals),
                "spread_max": max(vals),
            }
        )

    return by_tokenizer, by_script_rows, base_names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-script token multipliers (base / target) "
        "or a target-only worst-case safety table."
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base tokenizer tag or file stem (omit for target-only mode)",
    )
    parser.add_argument(
        "--target", required=True, help="Target tokenizer tag or file stem"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: one --target vs every tiktoken_* base tokenizer. "
        "Produces one per-base multiplier table plus a cross-base spread table "
        "(spread = mult_max/mult_min) restricted to scripts with n_langs > 1.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory with benchmark result JSONs",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV path to write the per-script table. If omitted, an auto-named "
        "file is written into the results directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print the per-language ratio table",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Target-only safety mode (no base tokenizer): worst-case per-script table.
    # (Skipped when --batch is requested; that path is handled below.)
    if not args.base and not args.batch:
        try:
            target_result = load_result(args.target, results_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        target_name = target_result.get("tokenizer", args.target)
        rows = compute_target_safety(target_result)

        print(f"Target-only worst-case safety table (target: {target_name})")
        print("  Use: target_tokens ~= total_bytes * sum(share_i * tokens_per_byte_worst_i)")
        print()
        header = (
            f"{'script':<8}{'#':>4}{'worst_lang':>16}"
            f"{'worst_bpt':>11}{'tokens/byte_worst':>19}{'median_bpt':>12}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            tpb = r["tokens_per_byte_worst"]
            tpb_s = f"{tpb:.4f}" if tpb != float("inf") else "inf"
            print(
                f"{r['script']:<8}{r['n_langs']:>4}{r['worst_lang']:>16}"
                f"{r['worst_bytes_per_token']:>11.2f}{tpb_s:>19}{r['median_bytes_per_token']:>12.2f}"
            )

        out_path = (
            Path(args.out)
            if args.out
            else DEFAULT_RESULTS_DIR / f"safety_{normalize(target_name)}.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "script",
                    "n_langs",
                    "worst_lang",
                    "worst_bytes_per_token",
                    "tokens_per_byte_worst",
                    "median_bytes_per_token",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {out_path}")
        return 0

    # Batch mode: one target vs every tiktoken_* base tokenizer.
    if args.batch:
        try:
            target_result = load_result(args.target, results_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        target_name = target_result.get("tokenizer", args.target)

        bases = find_tiktoken_bases(results_dir)
        if not bases:
            print("Error: no tiktoken_* base files found.", file=sys.stderr)
            return 1

        all_rows = []
        for base_path, base_name in bases:
            with open(base_path, "r", encoding="utf-8") as f:
                base_result = json.load(f)
            rows = compute_multipliers(base_result, target_result)
            all_rows.append((base_name, rows))

            out_path = (
                DEFAULT_RESULTS_DIR
                / f"multipliers_{normalize(base_name)}_vs_{normalize(target_name)}.csv"
            )
            write_multiplier_csv(rows, out_path)
            print(f"Wrote per-base CSV: {out_path}")

        # Cross-base spread table (spread = mult_max / mult_median), only scripts
        # with n_langs > 1. One row per base tokenizer, aggregated across scripts.
        # Cross-base spread tables (spread = mult_max / mult_median), only scripts
        # with n_langs > 1.
        by_tokenizer, by_script, base_names = build_cross_base_table(all_rows)

        # Table 1: per-tokenizer aggregation across scripts.
        print()
        print(f"Cross-base spread table -- per tokenizer (target: {target_name})")
        print("  spread = mult_max / mult_median, restricted to scripts with n_langs > 1")
        print("  sorted by spread_max ascending (top = smallest worst-case spread)")
        print()
        header = (
            f"{'base':<26}{'#scr':>5}"
            f"{'spread_min':>12}{'spread_med':>12}{'spread_max':>12}"
        )
        print(header)
        print("-" * len(header))
        for r in by_tokenizer:
            print(
                f"{r['base']:<26}{r['n_scripts']:>5}"
                f"{r['spread_min']:>12.3f}{r['spread_median']:>12.3f}"
                f"{r['spread_max']:>12.3f}"
            )

        # Table 2: per-script spread across tokenizers.
        print()
        print(f"Cross-base spread table -- per script (target: {target_name})")
        base_cols = [normalize(b)[:10] for b in base_names]
        header = f"{'script':<8}{'#b':>4}" + "".join(
            f"{c:>11}" for c in base_cols
        ) + f"{'spread_min':>12}{'spread_med':>12}{'spread_max':>12}"
        print(header)
        print("-" * len(header))
        for r in by_script:
            line = f"{r['script']:<8}{r['n_bases']:>4}"
            for b, c in zip(base_names, base_cols):
                v = r["per_base"].get(b)
                line += f"{(f'{v:.3f}' if v is not None else '-'):>11}"
            line += (
                f"{r['spread_min']:>12.3f}{r['spread_median']:>12.3f}"
                f"{r['spread_max']:>12.3f}"
            )
            print(line)

        # CSV: per-tokenizer table.
        tok_path = (
            DEFAULT_RESULTS_DIR
            / f"spread_{normalize(target_name)}_vs_tiktoken_bases.csv"
        )
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tok_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "base",
                    "n_scripts",
                    "spread_min",
                    "spread_median",
                    "spread_max",
                ],
            )
            writer.writeheader()
            writer.writerows(by_tokenizer)
        print(f"\nWrote per-tokenizer spread CSV: {tok_path}")

        # CSV: per-script table.
        scr_path = (
            DEFAULT_RESULTS_DIR
            / f"spread_{normalize(target_name)}_vs_tiktoken_bases_by_script.csv"
        )
        fields = ["script", "n_bases"] + [normalize(b) for b in base_names] + [
            "spread_min",
            "spread_median",
            "spread_max",
        ]
        with open(scr_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in by_script:
                row = {"script": r["script"], "n_bases": r["n_bases"]}
                for b in base_names:
                    row[normalize(b)] = r["per_base"].get(b)
                row["spread_min"] = r["spread_min"]
                row["spread_median"] = r["spread_median"]
                row["spread_max"] = r["spread_max"]
                writer.writerow(row)
        print(f"Wrote per-script spread CSV: {scr_path}")
        return 0

    # Ratio mode (base vs target)
    try:
        base_result = load_result(args.base, results_dir)
        target_result = load_result(args.target, results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    base_name = base_result.get("tokenizer", args.base)
    target_name = target_result.get("tokenizer", args.target)

    rows = compute_multipliers(base_result, target_result)

    print(f"Multipliers (base_bytes_per_token / target_bytes_per_token)")
    print(f"  base:   {base_name}")
    print(f"  target: {target_name}")
    print()
    header = (
        f"{'script':<8}{'#':>4}{'min':>9}{'median':>9}{'max':>9}"
        f"{'spread':>8}{'worst_lang':>16}{'base_med':>10}{'tgt_med':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        spread = r["mult_spread"]
        spread_s = f"{spread:.3f}" if spread != float("inf") else "inf"
        print(
            f"{r['script']:<8}{r['n_langs']:>4}"
            f"{r['mult_min']:>9.3f}{r['mult_median']:>9.3f}{r['mult_max']:>9.3f}"
            f"{spread_s:>8}"
            f"{r['worst_lang']:>16}"
            f"{r['base_median_bpt']:>10.2f}{r['target_median_bpt']:>9.2f}"
        )
    print()
    print(
        "Use 'max' for a conservative (overflow-proof) estimate; "
        "'median' for a central estimate."
    )

    if args.verbose:
        print("\nPer-language ratios (base_bpt / target_bpt):")
        base_map = script_language_bpt(base_result)
        target_map = script_language_bpt(target_result)
        langs = sorted(
            set(base_map) & set(target_map), key=lambda k: (k[0], k[1])
        )
        for (script, lang_key) in langs:
            b = base_map[(script, lang_key)]
            t = target_map[(script, lang_key)]
            if t > 0:
                print(f"  {lang_key:<16} {script:<7} {b / t:>7.3f}")

    out_path = (
        Path(args.out)
        if args.out
        else DEFAULT_RESULTS_DIR
        / f"multipliers_{normalize(base_name)}_vs_{normalize(target_name)}.csv"
    )
    write_multiplier_csv(rows, out_path)
    print(f"\nWrote CSV: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
