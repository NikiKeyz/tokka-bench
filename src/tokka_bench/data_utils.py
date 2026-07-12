"""
Data loading utilities for tokenizer benchmarking.

This module provides functions for loading language metadata and text samples
from various datasets (FineWeb-2, StarCoder, FineWeb).
"""

import gc
import os
import time
from typing import Dict, List

import pandas as pd

from .unicode_utils import check_script_purity

# Local on-disk cache for per-language sample text. Only the ~sample_size_mb
# that is actually consumed is stored (never the full dataset), so 100 languages
# at 2 MB each is roughly 200 MB on disk.
SAMPLE_CACHE_DIR = os.path.join("data", "cache", "samples")

# Constants for data processing
TB_TO_GB_FACTOR = 1000
MB_TO_GB_FACTOR = 1000
MB_TO_BYTES_FACTOR = 1024 * 1024
DEFAULT_CODING_LANGUAGES = 10
DEFAULT_TOP_LANGUAGES = 5

# Dataset constants
FINEWEB2_DATASET = "HuggingFaceFW/fineweb-2"
FINEWEB_DATASET = "HuggingFaceFW/fineweb"
STARCODER_DATASET = "bigcode/starcoderdata"
FINEWEB_SAMPLE_NAME = "sample-10BT"
TRAIN_SPLIT = "train"

# Try to disable datasets/tqdm progress bars globally to avoid noisy logs and
# potential tqdm issues in some environments.
try:  # pragma: no cover - best-effort safety
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()
except (ImportError, AttributeError, ModuleNotFoundError):
    # Datasets library not available or progress bar control failed
    pass


def load_language_data() -> pd.DataFrame:
    """Load natural language data from CSV file."""
    # Get the path to the CSV file in the src directory
    current_dir: str = os.path.dirname(__file__)
    csv_path: str = os.path.join(current_dir, "..", "fineweb-2-languages.csv")

    df: pd.DataFrame = pd.read_csv(csv_path)
    # Clean up the column names and data
    df.columns = df.columns.str.strip()
    return df


def load_coding_languages(n: int = 10) -> List[Dict[str, str]]:
    """Load coding language data from CSV file."""
    # Get the path to the CSV file in the src directory
    current_dir: str = os.path.dirname(__file__)
    csv_path: str = os.path.join(current_dir, "..", "starcoderdata-dirs.csv")

    df: pd.DataFrame = pd.read_csv(csv_path)

    # Convert to list of language info dictionaries (first n languages only)
    coding_langs: List[Dict[str, str]] = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= n:  # Stop after n languages
            break
        lang: str = row["Language"].strip()
        coding_langs.append(
            {
                "iso_code": lang,  # Use language name as identifier
                "script": "code",  # Mark as coding language
                "name": f"{lang.title()} (code)",
                "source": "starcoder",
                "data_dir": lang,
            }
        )

    return coding_langs

def get_natural_languages(df: pd.DataFrame, lang_list: List[str], n: int = 5) -> List[Dict[str, str]]:
    """Identify which natural language loader to use."""
    if len(lang_list) > 0:
        return get_listed_languages(df, lang_list)
    else:
        return get_top_languages(df, n)


def get_top_languages(df: pd.DataFrame, n: int = 5) -> List[Dict[str, str]]:
    """Get the top N natural languages by size."""
    # Filter out invalid rows (like Total row)
    df = df.dropna(subset=["Name", "Script"])
    df = df[df["ISO 639-3 code"] != "Total"]

    # Convert disk size to numeric for sorting
    def parse_size(size_str: str) -> float:
        size_str = str(size_str).strip()
        if "TB" in size_str:
            return float(size_str.replace("TB", "")) * 1000
        elif "GB" in size_str:
            return float(size_str.replace("GB", ""))
        elif "MB" in size_str:
            return float(size_str.replace("MB", "")) / 1000
        return 0.0

    df["size_gb"] = df["Disk size"].apply(parse_size)
    top_langs: pd.DataFrame = df.nlargest(n, "size_gb")

    return [
        {
            "iso_code": row["ISO 639-3 code"],
            "script": row["Script"],
            "name": row["Name"],
            "source": "fineweb2",
        }
        for _, row in top_langs.iterrows()
    ]


def get_listed_languages(df: pd.DataFrame, lang_list: List[str]) -> List[Dict[str, str]]:
    """Get the mentioned languages."""
    df = df.dropna(subset=["Name", "Script"])
    # Only keep languages that are mentioned by the user.
    df = df[df["Subset"].isin(lang_list)]

    return [
        {
            "iso_code": row["ISO 639-3 code"],
            "script": row["Script"],
            "name": row["Name"],
            "source": "fineweb2",
        }
        for _, row in df.iterrows()
    ]


def get_english_fineweb() -> Dict[str, str]:
    """Get English from FineWeb sample-10BT."""
    return {
        "iso_code": "eng",
        "script": "Latn",
        "name": "English (FineWeb)",
        "source": "fineweb",
    }


def _sample_cache_path(
    language_info: Dict[str, str],
    sample_size_mb: float,
    cache_dir: str,
    purity_threshold: float = 0.0,
) -> str:
    """Human-readable cache file path for a given language + parameters.

    Pattern: ``{source}_{iso}_{script}[_{data_dir}]_{sample_size_mb}mb_p{purity_threshold}.txt``
    Uniquely identifies the language, source, sample size, and purity threshold
    so cached files can be found by name for external reuse (e.g. testing in Veska).
    """
    source = language_info.get("source", "fineweb2")
    iso = language_info.get("iso_code", "")
    script = language_info.get("script", "")
    data_dir = language_info.get("data_dir", "")
    purity_part = f"_p{purity_threshold}" if purity_threshold > 0.0 else ""
    data_dir_part = f"_{data_dir}" if data_dir else ""
    name = f"{source}_{iso}_{script}{data_dir_part}_{sample_size_mb}mb{purity_part}.txt"
    return os.path.join(cache_dir, name)


def load_real_sample_text(
    language_info: Dict[str, str],
    sample_size_mb: float = 2.0,
    verbose: bool = False,
    max_retries: int = 3,
    use_cache: bool = True,
    cache_dir: str = SAMPLE_CACHE_DIR,
    purity_threshold: float = 0.0,
) -> str:
    """Load real sample text from appropriate dataset based on source.

    Caches the consumed bytes (~``sample_size_mb``) to a local file so repeated
    runs do not re-stream from HuggingFace. With a warm cache, HF is never
    contacted. Retries transient failures (e.g. ``datasets`` parquet
    ``CastError`` or ``tqdm`` lock races under high concurrency), and
    re-attempts when a stream yields zero bytes. Never falls back to synthetic
    text — failures raise after retries are exhausted.

    When ``purity_threshold > 0.0``, each document from the stream is checked
    for script purity: characters belonging to the target script (as defined by
    the ``language_info["script"]`` code) plus always-native categories
    (Punctuation, Symbols, Numbers, Common/Inherited/Unknown) are counted as
    native. Documents with a native fraction below the threshold are skipped.
    This ensures the sample contains minimal cross-script contamination.
    """
    from datasets import load_dataset

    # Use the module constant for bytes-per-MB conversion
    target_bytes: int = int(sample_size_mb * MB_TO_BYTES_FACTOR)
    source: str = language_info.get("source", "fineweb2")

    cache_path = ""
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = _sample_cache_path(language_info, sample_size_mb, cache_dir, purity_threshold)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = f.read()
                if verbose:
                    print(
                        f"    Loaded {len(cached.encode('utf-8')):,} bytes from cache "
                        f"({os.path.relpath(cache_path)})"
                    )
                return cached
            except OSError:
                # Corrupt/unreadable cache — fall through to re-download.
                pass

    if verbose:
        print(f"    Loading real data from {source}...")

    def _open_stream():
        if source == "fineweb2":
            dataset_name: str = f"{language_info['iso_code']}_{language_info['script']}"
            return (
                load_dataset(
                    "HuggingFaceFW/fineweb-2",
                    name=dataset_name,
                    split="train",
                    streaming=True,
                    # Some shards carry extra columns (e.g. `wordlist_ratio`)
                    # absent from the subset schema, which makes the strict
                    # schema cast fail with a CastError. Requesting only the
                    # column we need sidesteps that entirely.
                    columns=["text"],
                ),
                "text",
            )
        elif source == "fineweb":
            return (
                load_dataset(
                    "HuggingFaceFW/fineweb",
                    name="sample-10BT",
                    split="train",
                    streaming=True,
                    columns=["text"],
                ),
                "text",
            )
        elif source == "starcoder":
            data_dir: str = language_info.get("data_dir", language_info["iso_code"])
            return (
                load_dataset(
                    "bigcode/starcoderdata",
                    data_dir=data_dir,
                    split="train",
                    streaming=True,
                    columns=["content"],
                ),
                "content",
            )
        else:
            raise ValueError(f"Unknown source: {source}")

    last_err: Exception = RuntimeError("no attempts made")
    for attempt in range(1, max_retries + 1):
        fw = None
        dataset_iter = None
        try:
            fw, content_key = _open_stream()

            # Accumulate text until we reach target size
            accumulated_text: List[str] = []
            total_bytes: int = 0
            docs_skipped: int = 0

            dataset_iter = iter(fw)
            try:
                while total_bytes < target_bytes:
                    sample = next(dataset_iter)
                    text: str = sample.get(content_key, "")
                    if not text:
                        continue
                    if purity_threshold > 0.0:
                        purity = check_script_purity(text, language_info["script"])
                        if purity < purity_threshold:
                            docs_skipped += 1
                            continue
                    accumulated_text.append(text)
                    total_bytes += len(text.encode("utf-8"))
            except StopIteration:
                # End of dataset reached
                pass

            # A stream that ends immediately (0 bytes) is a classic symptom of a
            # concurrent parquet cast failure; treat it as a retryable error.
            if total_bytes == 0:
                last_err = RuntimeError("stream yielded 0 bytes")
                if verbose:
                    print(
                        f"    Warning: 0 bytes loaded (attempt {attempt}/{max_retries}), retrying..."
                    )
                time.sleep(0.5 * attempt)
                continue

            # Join all accumulated text
            full_text: str = "\n".join(accumulated_text)

            # Truncate to exact size if needed
            text_bytes: bytes = full_text.encode("utf-8")
            if len(text_bytes) > target_bytes:
                full_text = text_bytes[:target_bytes].decode("utf-8", errors="ignore")

            if verbose:
                print(
                    f"    Loaded {len(full_text.encode('utf-8')):,} bytes of real data"
                )
                if docs_skipped > 0:
                    print(
                        f"    Skipped {docs_skipped} docs below {purity_threshold:.0%} purity"
                    )

            # Persist to local cache (atomic write) for subsequent runs.
            if use_cache:
                tmp_path = cache_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                os.replace(tmp_path, cache_path)

            return full_text

        except (ValueError, KeyError, ImportError, ConnectionError, OSError) as e:
            last_err = e
            if verbose:
                print(
                    f"    Warning: load failed (attempt {attempt}/{max_retries}): {e}"
                )
            time.sleep(0.5 * attempt)
            continue
        finally:
            # Simple cleanup
            del fw
            del dataset_iter
            gc.collect()

    lang_name = language_info.get("name", language_info.get("iso_code", "unknown"))
    raise RuntimeError(
        f"Failed to load REAL data for '{lang_name}' after {max_retries} attempts "
        f"(last error: {last_err!r}). Refusing to use synthetic fallback text — "
        f"rerun with fewer workers or check network/HF access."
    )
