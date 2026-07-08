"""
Language categorization and filtering utilities.
"""

from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd


def detect_language_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build the exact category sets requested, with careful ordering and labeling.

    Categories returned (in order):
    - All Languages
    - Top 30 Natural (English first)
    - 31–60 Natural
    - 61–100 Natural
    - Coding
    - European
    - Non-European
    - One preset per Unicode script present in the data (e.g. "Script: Cyrl")

    All category lists contain unique ``lang_key`` values (iso_script), NOT
    display names, so a script preset can never accidentally include a different
    script variant that happens to share a language name (e.g. Serbian Cyrl vs
    Serbian Latn).
    """

    # Unique language entries (one per iso_script variant)
    all_keys: List[str] = list(df["lang_key"].unique())

    def info(lk: str) -> pd.Series:
        return df[df["lang_key"] == lk].iloc[0]

    # Popularity order if available (lower rank = more popular)
    rank_map: Dict[str, Optional[float]] = (
        df.groupby("lang_key")["language_rank"].min().to_dict()
        if "language_rank" in df.columns
        else {}
    )

    def rank_key(lk: str) -> float:
        rank = rank_map.get(lk)
        return rank if rank is not None else 1e9

    # Identify programming vs natural and English
    programming_languages: List[str] = []
    natural_languages: List[str] = []
    english_languages: List[str] = []
    for lk in all_keys:
        r = info(lk)
        script_value = str(r.get("script", "")).lower()
        name_value = str(r.get("language", ""))
        if "(code)" in name_value.lower() or script_value == "code":
            programming_languages.append(lk)
        elif "english" in name_value.lower():
            english_languages.append(lk)
        else:
            natural_languages.append(lk)

    # Ordering
    natural_by_rank = sorted(natural_languages, key=rank_key)
    programming_by_rank = sorted(programming_languages, key=rank_key)

    # All Languages overall order: English (if present) → natural by rank → coding by rank
    ordered_all_languages = english_languages + natural_by_rank + programming_by_rank

    # Natural ranges with English at the front of Top 30
    top_natural_with_english: List[str] = english_languages[:1] + [
        l for l in natural_by_rank if l not in set(english_languages)
    ]
    top_30_natural = top_natural_with_english[:30]
    natural_31_60 = top_natural_with_english[30:60]
    natural_61_100 = top_natural_with_english[60:100]

    # Script mapping per lang_key
    script_map: Dict[str, str] = {
        lk: str(info(lk).get("script", "")).lower() for lk in all_keys
    }
    latin_script = [lk for lk, s in script_map.items() if "latn" in s]
    cyrillic_script = [lk for lk, s in script_map.items() if "cyrl" in s]
    arabic_script = [lk for lk, s in script_map.items() if "arab" in s]
    cjk_scripts = [
        lk
        for lk, s in script_map.items()
        if any(tag in s for tag in ["hani", "jpan", "hang"])
    ]

    # European vs Non-European using CSV families plus script as a guard
    # Load FineWeb-2 CSV to access the Language Family column
    try:
        repo_root = Path(__file__).resolve().parents[1]
        fineweb_csv = repo_root / "fineweb-2-languages.csv"
        name_to_family: Dict[str, str] = {}
        if fineweb_csv.exists():
            fw = pd.read_csv(fineweb_csv)
            for _, row in fw.iterrows():
                name_to_family[str(row.get("Name", ""))] = str(
                    row.get("Language Family", "")
                )
        european_families = {"Indo-European", "Uralic", "Turkic", "Kartvelian"}
        european_name_exceptions = {"Basque", "Maltese"}

        def is_european(lk: str) -> bool:
            # Ignore coding
            if lk in programming_languages:
                return False
            r = info(lk)
            fam = name_to_family.get(str(r.get("language", "")), "")
            scr = str(r.get("script", "")).lower()
            name_val = str(r.get("language", ""))
            if name_val in european_name_exceptions:
                return True
            # Require European-associated script and qualifying family
            if any(tag in scr for tag in ["latn", "cyrl", "grek"]) and (
                fam in european_families
            ):
                return True
            return False

        european = [lk for lk in natural_by_rank if is_european(lk)]
    except Exception:
        # Fallback: script-only heuristic
        european = [
            lk
            for lk, s in script_map.items()
            if any(tag in s for tag in ["latn", "cyrl", "grek"])
            and lk in natural_languages
        ]

    non_european = [lk for lk in natural_by_rank if lk not in set(european)]

    # One preset per script actually present in the data, so every script is
    # selectable (not just the four hard-coded ones).
    script_codes = sorted({s for s in script_map.values() if s and s != "code"})
    per_script: Dict[str, List[str]] = {}
    for code in script_codes:
        per_script[f"Script: {code.upper()}"] = [
            lk for lk, s in script_map.items() if s == code
        ]

    # Script families: linguistically related scripts grouped together (the
    # same idea as the existing "CJK Scripts" bucket). Each family collects the
    # lang_keys whose script code belongs to that family.
    SCRIPT_FAMILIES: Dict[str, List[str]] = {
        "CJK Scripts": ["hani", "jpan", "hang"],
        "Indic Scripts": [
            "deva",
            "beng",
            "gujr",
            "guru",
            "knda",
            "mlym",
            "mymr",
            "orya",
            "sinh",
            "taml",
            "telu",
            "tibt",
        ],
        "Southeast Asian Scripts": ["thai", "khmr", "laoo"],
    }
    family_presets: Dict[str, List[str]] = {}
    for family, codes in SCRIPT_FAMILIES.items():
        family_presets[family] = [
            lk for lk, s in script_map.items() if s in codes
        ]

    # Coding/script buckets preserved as convenient groupings
    categories: Dict[str, List[str]] = {
        "All Languages": ordered_all_languages,
        "Top 30 Natural": top_30_natural,
        "31–60 Natural": natural_31_60,
        "61–100 Natural": natural_61_100,
        "Coding": programming_by_rank,
        "European": european,
        "Non-European": non_european,
        "Latin Script": latin_script,
        "Cyrillic Script": cyrillic_script,
        "Arabic Script": arabic_script,
        "CJK Scripts": cjk_scripts,
    }
    # Append per-script presets (sorted for stable ordering)
    for key in sorted(per_script.keys()):
        categories[key] = per_script[key]
    # Append family presets (sorted for stable ordering)
    for key in sorted(family_presets.keys()):
        # Don't clobber an existing single-script preset of the same name
        if key not in categories:
            categories[key] = family_presets[key]

    return categories
