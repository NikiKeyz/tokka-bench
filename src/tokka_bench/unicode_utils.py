"""
Unicode script detection and text analysis utilities.

This module provides functions for analyzing Unicode scripts and text properties,
which are used throughout the tokenizer benchmarking process.
"""

import functools
import unicodedata
from typing import Dict, FrozenSet, List, Set

# Unicode category constants
PUNCTUATION_CATEGORY_PREFIX = "P"
SYMBOL_CATEGORY_PREFIX = "S"
NUMBER_CATEGORY_PREFIX = "N"

# Special symbol keywords
MATHEMATICAL_KEYWORDS = {"MATHEMATICAL", "DOUBLE-STRUCK"}


# Unicode script mappings for major writing systems
# Order matters: more specific scripts should come first
UNICODE_SCRIPTS: Dict[str, List[str]] = {
    "Korean": ["HANGUL"],  # Must come before Chinese to avoid HAN matching in HANGUL
    "Japanese": ["HIRAGANA", "KATAKANA"],
    "Chinese": ["CJK"],  # CJK covers all CJK ideographs without false matches
    "Latin": ["LATIN"],
    "Cyrillic": ["CYRILLIC"],
    "Arabic": ["ARABIC"],
    "Devanagari": ["DEVANAGARI"],
    "Thai": ["THAI"],
    "Hebrew": ["HEBREW"],
    "Greek": ["GREEK"],
    # Extended coverage for all scripts present in fineweb-2
    "Armenian": ["ARMENIAN"],
    "Bengali": ["BENGALI"],
    "Canadian_Aboriginal": ["CANADIAN SYLLABICS"],
    "Cherokee": ["CHEROKEE"],
    "Coptic": ["COPTIC"],
    "Ethiopic": ["ETHIOPIC"],
    "Georgian": ["GEORGIAN"],
    "Gothic": ["GOTHIC"],
    "Gujarati": ["GUJARATI"],
    "Gurmukhi": ["GURMUKHI"],
    "Kayah_Li": ["KAYAH LI"],
    "Khmer": ["KHMER"],
    "Kannada": ["KANNADA"],
    "Lao": ["LAO"],
    "Limbu": ["LIMBU"],
    "Lisu": ["LISU"],
    "Malayalam": ["MALAYALAM"],
    "Mongolian": ["MONGOLIAN"],
    "Meetei_Mayek": ["MEETEI"],
    "Myanmar": ["MYANMAR"],
    "Nko": ["NKO"],
    "Ol_Chiki": ["OL CHIKI"],
    "Oriya": ["ORIYA"],
    "Sinhala": ["SINHALA"],
    "Syriac": ["SYRIAC"],
    "Tamil": ["TAMIL"],
    "Telugu": ["TELUGU"],
    "Tifinagh": ["TIFINAGH"],
    "Thaana": ["THAANA"],
    "Tibetan": ["TIBETAN"],
}

# Maps ISO 15924 script codes to the family name(s) in UNICODE_SCRIPTS.
# Characters matching any family for the target script code are considered
# "native" for purity purposes.
SCRIPT_CODE_TO_FAMILIES: Dict[str, FrozenSet[str]] = {
    "Jpan": frozenset({"Japanese", "Chinese"}),
    "Hani": frozenset({"Chinese"}),
    "Kore": frozenset({"Korean"}),
    "Hang": frozenset({"Korean"}),
    "Latn": frozenset({"Latin"}),
    "Cyrl": frozenset({"Cyrillic"}),
    "Arab": frozenset({"Arabic"}),
    "Deva": frozenset({"Devanagari"}),
    "Thai": frozenset({"Thai"}),
    "Hebr": frozenset({"Hebrew"}),
    "Grek": frozenset({"Greek"}),
    "Armn": frozenset({"Armenian"}),
    "Beng": frozenset({"Bengali"}),
    "Cans": frozenset({"Canadian_Aboriginal"}),
    "Cher": frozenset({"Cherokee"}),
    "Copt": frozenset({"Coptic"}),
    "Ethi": frozenset({"Ethiopic"}),
    "Geor": frozenset({"Georgian"}),
    "Goth": frozenset({"Gothic"}),
    "Gujr": frozenset({"Gujarati"}),
    "Guru": frozenset({"Gurmukhi"}),
    "Kali": frozenset({"Kayah_Li"}),
    "Khmr": frozenset({"Khmer"}),
    "Knda": frozenset({"Kannada"}),
    "Laoo": frozenset({"Lao"}),
    "Limb": frozenset({"Limbu"}),
    "Lisu": frozenset({"Lisu"}),
    "Mlym": frozenset({"Malayalam"}),
    "Mong": frozenset({"Mongolian"}),
    "Mtei": frozenset({"Meetei_Mayek"}),
    "Mymr": frozenset({"Myanmar"}),
    "Nkoo": frozenset({"Nko"}),
    "Olck": frozenset({"Ol_Chiki"}),
    "Orya": frozenset({"Oriya"}),
    "Sinh": frozenset({"Sinhala"}),
    "Syrc": frozenset({"Syriac"}),
    "Taml": frozenset({"Tamil"}),
    "Telu": frozenset({"Telugu"}),
    "Tfng": frozenset({"Tifinagh"}),
    "Thaa": frozenset({"Thaana"}),
    "Tibt": frozenset({"Tibetan"}),
}

# Character categories always counted as native regardless of target script.
ALWAYS_NATIVE: FrozenSet[str] = frozenset({
    "Punctuation", "Symbols", "Numbers", "Other",
})


def get_unicode_scripts(text: str) -> Set[str]:
    """Get the set of Unicode scripts present in the text."""
    if not text:
        return set()

    scripts: Set[str] = set()
    for char in text:
        if char.isspace():
            continue

        # Get character name and category with error handling
        try:
            char_name: str = unicodedata.name(char, "")
            category: str = unicodedata.category(char)
        except (ValueError, TypeError):
            # Skip characters that can't be analyzed
            continue

        # Check for mathematical symbols first (they're classified as Lu but should be Symbols)
        if any(keyword in char_name for keyword in MATHEMATICAL_KEYWORDS):
            scripts.add("Symbols")
            continue

        # Check for punctuation, symbols, numbers using category prefixes
        if category.startswith(
            PUNCTUATION_CATEGORY_PREFIX
        ):  # Punctuation (Po, Pc, Pd, Ps, Pe, Pi, Pf)
            scripts.add("Punctuation")
            continue
        elif category.startswith(SYMBOL_CATEGORY_PREFIX):  # Symbols (Sm, Sc, Sk, So)
            scripts.add("Symbols")
            continue
        elif category.startswith(NUMBER_CATEGORY_PREFIX):  # Numbers (Nd, Nl, No)
            scripts.add("Numbers")
            continue

        # Check for script families using the full character name
        for script_name, script_codes in UNICODE_SCRIPTS.items():
            if any(script_code in char_name for script_code in script_codes):
                scripts.add(script_name)
                break
    return scripts


@functools.lru_cache(maxsize=None)
def _classify_char(char: str) -> str:
    """Classify a single Unicode character into a script family or category label.

    Returns one of:
    - A script family name from ``UNICODE_SCRIPTS`` (e.g. ``"Latin"``, ``"Chinese"``)
    - ``"Punctuation"``, ``"Symbols"``, ``"Numbers"`` (by Unicode category)
    - ``"Other"`` for Common/Inherited/Unknown scripts not covered by UNICODE_SCRIPTS
    """
    try:
        char_name = unicodedata.name(char, "")
        category = unicodedata.category(char)
    except (ValueError, TypeError):
        return "Other"

    if not char_name:
        return "Other"

    # Mathematical symbols (classified as Lu/Ll but semantically symbols)
    if any(keyword in char_name for keyword in MATHEMATICAL_KEYWORDS):
        return "Symbols"

    # Category-based classification
    if category.startswith(PUNCTUATION_CATEGORY_PREFIX):
        return "Punctuation"
    if category.startswith(SYMBOL_CATEGORY_PREFIX):
        return "Symbols"
    if category.startswith(NUMBER_CATEGORY_PREFIX):
        return "Numbers"

    # Script family matching (first match wins — order in UNICODE_SCRIPTS is intentional)
    for script_name, script_codes in UNICODE_SCRIPTS.items():
        if any(code in char_name for code in script_codes):
            return script_name

    return "Other"


def check_script_purity(text: str, script_code: str) -> float:
    """Return fraction of non-whitespace characters that are native to ``script_code``.

    "Native" means the character belongs to the target script family (as defined
    by ``SCRIPT_CODE_TO_FAMILIES``), or is one of the always-native categories
    (Punctuation, Symbols, Numbers, Common/Inherited/Unknown).

    Returns ``1.0`` for empty text or when ``script_code`` has no known mapping
    (fallback: no filtering).
    """
    if not text:
        return 1.0

    allowed_families = SCRIPT_CODE_TO_FAMILIES.get(script_code)
    if allowed_families is None:
        return 1.0

    total = 0
    native = 0

    for char in text:
        if char.isspace():
            continue
        total += 1
        family = _classify_char(char)
        if family in allowed_families or family in ALWAYS_NATIVE:
            native += 1

    return native / total if total > 0 else 1.0


def has_whitespace_in_middle(text: str) -> bool:
    """Check if text has whitespace characters in the middle (not at start/end)."""
    if not text:
        return False

    stripped = text.strip()
    return len(stripped) > 0 and any(char.isspace() for char in stripped)


def starts_with_space(text: str) -> bool:
    """Check if text starts with a whitespace character."""
    return bool(text and text[0].isspace())
