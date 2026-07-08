"""
Fast, clean multi-tokenizer benchmark.

Key principles:
- Load each language's text once (e.g., 2MB), concatenate into one string.
- Tokenize that text once per tokenizer; reuse token IDs everywhere.
- Compute per-language metrics and global metrics efficiently.
- Preserve classic output shape and quiet logging.
"""

from __future__ import annotations

import gc
import hashlib
import random
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from .tokenizer import _load_raw_tokenizer

from .data_utils import (
    get_english_fineweb,
    get_natural_languages,
    load_coding_languages,
    load_language_data,
    load_real_sample_text,
)
from .metrics import GlobalMetricsTracker, calculate_word_metrics
from .unicode_utils import (
    get_unicode_scripts,
    has_whitespace_in_middle,
    starts_with_space,
)

# Module constants
DEFAULT_SAMPLE_BYTES = 300
MIN_SKIP_BYTES = 4096
MAX_GLOBAL_SAMPLES = 2000
VOCAB_SAMPLE_SIZE = 10000
PROGRESS_BAR_WIDTH = 30
SAMPLE_TOKEN_LIMIT = 10


@dataclass
class FastTokenizer:
    """Lightweight tokenizer wrapper without verbose analysis on init."""

    name: str

    def __post_init__(self) -> None:
        # trust_remote_code=True allows custom tokenizers (e.g., tiktoken-wrapped)
        try:
            self._tok = AutoTokenizer.from_pretrained(
                self.name, trust_remote_code=True
            )
        except Exception as e:
            # Fall back to loading the raw tokenizer.json directly. This handles
            # models whose tokenizer_config.json uses a format the installed
            # transformers version cannot parse (e.g. Gemma 4).
            print(
                f"    ⚠️  AutoTokenizer failed ({e}); falling back to raw tokenizer.json"
            )
            self._tok = _load_raw_tokenizer(self.name)
        self.vocab_size: int = len(self._tok)

    # Expose a minimal protocol expected by metrics functions
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def tokenizer(self):  # for compatibility with GlobalMetricsTracker helpers
        return self._tok


def _compute_vocab_metrics_quiet(tokenizer: FastTokenizer) -> Dict[str, Any]:
    """Sample-based vocab analysis without printing (compatible keys)."""
    vocab_size: int = len(tokenizer.tokenizer)
    sample_size: int = min(vocab_size, VOCAB_SAMPLE_SIZE)
    step: int = max(1, vocab_size // max(1, sample_size))

    tokens_without_leading_space = 0
    analyzed_count = 0
    sample_non_space_tokens: List[str] = []
    sample_space_tokens: List[str] = []

    for token_id in range(0, vocab_size, step):
        try:
            token_text: str = tokenizer.tokenizer.decode(
                [token_id], skip_special_tokens=True
            )
        except (ValueError, KeyError, RuntimeError, AttributeError):
            # Skip problematic tokens but log in debug mode
            continue

        analyzed_count += 1
        if token_text and not token_text.startswith(" "):
            tokens_without_leading_space += 1
            if len(sample_non_space_tokens) < SAMPLE_TOKEN_LIMIT:
                sample_non_space_tokens.append(token_text)
        else:
            if len(sample_space_tokens) < SAMPLE_TOKEN_LIMIT:
                sample_space_tokens.append(token_text)

    if analyzed_count > 0:
        pct = (tokens_without_leading_space / analyzed_count) * 100
        est_count = int((tokens_without_leading_space / analyzed_count) * vocab_size)
    else:
        pct = 0.0
        est_count = 0

    return {
        "tokens_without_leading_space_count": est_count,
        "tokens_without_leading_space_pct": pct,
        "analyzed_sample_size": analyzed_count,
        "sample_non_space_tokens": sample_non_space_tokens,
        "sample_space_tokens": sample_space_tokens,
    }


def _compute_language_metrics_from_ids(
    tokenizer: FastTokenizer,
    text: str,
    lang_info: Dict[str, str],
    token_ids: List[int],
) -> Dict[str, Any]:
    text_bytes: int = len(text.encode("utf-8"))
    num_tokens: int = len(token_ids)
    unique_tokens: int = len(set(token_ids))

    # Reuse precomputed tokenization for fertility; still encode small samples per-word inside
    word_metrics: Dict[str, Any] = calculate_word_metrics(
        tokenizer,
        text,
        lang_info,
        pretokenized_text_token_ids=token_ids,
    )
    debug_info = word_metrics.get("debug_info", {})

    return {
        "bytes_per_token": (text_bytes / num_tokens) if num_tokens > 0 else 0.0,
        "total_bytes": text_bytes,
        "total_tokens": num_tokens,
        "unique_tokens": unique_tokens,
        "subword_fertility": word_metrics.get("subword_fertility", 0.0),
        # Backward-compat: keep field if present, but prefer new ones downstream
        "continued_word_rate": word_metrics.get("continued_word_rate", 0.0),
        "word_split_pct": word_metrics.get("word_split_pct", 0.0),
        "continuation_token_pct": word_metrics.get(
            "continuation_token_pct", word_metrics.get("continued_word_rate", 0.0)
        ),
        # Promoted debug fields for downstream analysis
        "segmentation_method": debug_info.get("segmentation_method"),
        "total_units": debug_info.get("total_words"),
        "estimated_units_split": debug_info.get("words_split"),
    }


def _random_snippet(
    text: str,
    sample_bytes: int = DEFAULT_SAMPLE_BYTES,
    min_skip_bytes: int = MIN_SKIP_BYTES,
    seed: Optional[int] = None,
) -> Tuple[str, int]:
    """Return a UTF-8 safe snippet and its byte offset.

    - Skips at least min_skip_bytes (when possible) to avoid the very beginning.
    - Uses a deterministic RNG seed (if provided) for reproducibility per-language.
    """
    data = text.encode("utf-8")
    total = len(data)
    if total <= sample_bytes:
        return text, 0

    rng = random.Random(seed)
    start_min = 0 if total <= min_skip_bytes + sample_bytes else min_skip_bytes
    start_max = max(0, total - sample_bytes)
    if start_min >= start_max:
        start = start_max
    else:
        start = rng.randint(start_min, start_max)

    snippet = data[start : start + sample_bytes].decode("utf-8", errors="ignore")
    return snippet, start


def _sample_token_info_for_global(
    tokenizer: FastTokenizer,
    token_ids: List[int],
    max_samples: int = MAX_GLOBAL_SAMPLES,
) -> List[Dict[str, Any]]:
    """Decode a sample of token IDs to reduce cost of global metrics."""
    if not token_ids:
        return []

    if len(token_ids) > max_samples:
        step = max(1, len(token_ids) // max_samples)
        sampled_ids = token_ids[::step][:max_samples]
    else:
        sampled_ids = token_ids

    tokens_info: List[Dict[str, Any]] = []
    for tid in sampled_ids:
        try:
            token_text = tokenizer.decode([tid], skip_special_tokens=True)
        except (ValueError, KeyError, RuntimeError, AttributeError):
            # Skip tokens that can't be decoded
            continue

        scripts = get_unicode_scripts(token_text)
        tokens_info.append(
            {
                "id": tid,
                "text": token_text,
                "starts_with_space": starts_with_space(token_text),
                "has_whitespace_in_middle": has_whitespace_in_middle(token_text),
                "scripts": list(scripts),
                "script_overlap": len(scripts) > 1,
            }
        )

    return tokens_info


def _process_single_language_with_retry(
    tokenizers: List[FastTokenizer],
    lang_info: Dict[str, str],
    sample_size_mb: float,
    max_retries: int = 3,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Run ``_process_single_language`` with retries for transient failures.

    Transient errors (HF dataset races, ``tqdm`` lock corruption, network
    blips) are common under high ``max_workers``; retrying usually succeeds.
    """
    import time

    name = lang_info.get("name", "Unknown")
    last_err: Exception = RuntimeError("no attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            return _process_single_language(tokenizers, lang_info, sample_size_mb)
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            last_err = e
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
                continue
    raise RuntimeError(f"{name}: failed after {max_retries} attempts: {last_err}")


def _process_single_language(
    tokenizers: List[FastTokenizer],
    lang_info: Dict[str, str],
    sample_size_mb: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Return per-tokenizer metrics and sampled token infos for global metrics.

    Steps:
    - Load and concatenate ~sample_size_mb of text for the language once.
    - For each tokenizer: encode once; reuse token IDs for metrics and global sampling.
    """
    text: str = load_real_sample_text(lang_info, sample_size_mb, verbose=False)
    # Deterministic per-language seed for reproducible sampling
    iso = lang_info.get("iso_code", "")
    script = lang_info.get("script", "")
    seed = int.from_bytes(
        hashlib.sha1(f"{iso}:{script}".encode("utf-8")).digest()[:8], "big"
    )
    snippet, byte_offset = _random_snippet(
        text,
        sample_bytes=DEFAULT_SAMPLE_BYTES,
        min_skip_bytes=MIN_SKIP_BYTES,
        seed=seed,
    )

    per_tokenizer_metrics: Dict[str, Dict[str, Any]] = {}
    per_tokenizer_sampled_tokens: Dict[str, List[Dict[str, Any]]] = {}

    for tok in tokenizers:
        # Encode once per tokenizer per language
        token_ids: List[int] = tok.encode(text)

        # Compute per-language metrics using pretokenized IDs
        metrics = _compute_language_metrics_from_ids(tok, text, lang_info, token_ids)
        # Build sample tokenization for the 300-byte snippet
        sample_token_ids: List[int] = tok.encode(snippet)
        sample_tokens: List[str] = []
        for tid in sample_token_ids:
            try:
                token_text = tok.decode([tid], skip_special_tokens=True)
            except (ValueError, KeyError, RuntimeError, AttributeError):
                continue
            if token_text:
                sample_tokens.append(token_text)

        per_tokenizer_metrics[tok.name] = {
            "language_info": lang_info,
            "metrics": metrics,
            "sample": {
                "byte_offset": byte_offset,
                "text_bytes": len(snippet.encode("utf-8")),
                "text": snippet,
                "token_ids": sample_token_ids,
                "tokens": sample_tokens,
            },
        }

        # Global metrics sampling from precomputed IDs
        tokens_info = _sample_token_info_for_global(
            tok, token_ids, max_samples=MAX_GLOBAL_SAMPLES
        )
        per_tokenizer_sampled_tokens[tok.name] = tokens_info

    return per_tokenizer_metrics, per_tokenizer_sampled_tokens


def _script_stat_block(values: List[float], keys: List[str]) -> Dict[str, Any]:
    """Return min/median/max plus the language holding the min/max value.

    ``values`` and ``keys`` are parallel lists (one entry per language in the
    script group). For bytes/token the *min* is the worst case for context
    overflow (fewest bytes per token -> more tokens per byte).
    """
    if not values:
        return {}

    paired = sorted(zip(values, keys))
    min_v, min_k = paired[0]
    max_v, max_k = paired[-1]
    return {
        "min": min_v,
        "median": statistics.median(values),
        "max": max_v,
        "min_language": min_k,
        "max_language": max_k,
    }


def aggregate_by_script(
    languages: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Group per-language results by Unicode script and summarize metrics.

    Produces ``min``/``median``/``max`` (with the holding language) for the
    metrics that matter for coverage and overflow safety: ``bytes_per_token``,
    ``unique_tokens`` and ``subword_fertility``. Rare languages are already
    excluded by the top-N language selection upstream, so the min reflects a
    realistic worst case for that script.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for lang_key, lang_result in languages.items():
        script = lang_result.get("language_info", {}).get("script", "Unknown")
        g = groups.setdefault(
            script,
            {"keys": [], "bpt": [], "ut": [], "fert": []},
        )
        g["keys"].append(lang_key)
        g["bpt"].append(lang_result.get("metrics", {}).get("bytes_per_token", 0.0))
        g["ut"].append(lang_result.get("metrics", {}).get("unique_tokens", 0))
        g["fert"].append(lang_result.get("metrics", {}).get("subword_fertility", 0.0))

    out: Dict[str, Dict[str, Any]] = {}
    for script, g in groups.items():
        out[script] = {
            "num_languages": len(g["keys"]),
            "languages": g["keys"],
            "bytes_per_token": _script_stat_block(g["bpt"], g["keys"]),
            "unique_tokens": _script_stat_block(g["ut"], g["keys"]),
            "subword_fertility": _script_stat_block(g["fert"], g["keys"]),
        }
    return out


def run_benchmark(
    tokenizer_names: List[str],
    output_names: Optional[List[str]] = None,
    sample_size_mb: float = 2.0,
    max_workers: int = 8,
    natural_n: int = 99,
    natural_lang_list: List[str] = [],
    code_n: int = 20,
) -> Dict[str, Any]:
    """Benchmark multiple tokenizers across many languages quickly.

    Returns a dict keyed by tokenizer name with the same shape as the classic benchmark.
    """
    if not tokenizer_names:
        raise ValueError("tokenizer_names must be a non-empty list")

    # Quiet header; a single compact line
    print(
        f"🚀 Fast benchmark | tokenizers={len(tokenizer_names)} | sample={sample_size_mb}MB | workers={max_workers}"
    )

    # Load tokenizers (quiet)
    print("🔧 Loading tokenizers...")
    tokenizers: List[FastTokenizer] = []
    for name in tokenizer_names:
        # concise per-tokenizer load message
        print(f"  • {name}")
        tokenizers.append(FastTokenizer(name))
    print(f"✅ Loaded: {len(tokenizers)}")

    # Pre-compute quiet vocab metrics once per tokenizer (parallelized)
    vocab_metrics_by_tok: Dict[str, Dict[str, Any]] = {}
    try:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(tokenizers) or 1)
        ) as ex:
            metrics_list = list(ex.map(_compute_vocab_metrics_quiet, tokenizers))
        for tok, vm in zip(tokenizers, metrics_list):
            vocab_metrics_by_tok[tok.name] = vm
    except Exception:
        # Fallback to sequential if any issue arises
        for tok in tokenizers:
            vocab_metrics_by_tok[tok.name] = _compute_vocab_metrics_quiet(tok)

    # Languages: English + configurable natural/code language counts (defaults match classic)
    english = get_english_fineweb()
    df = load_language_data()
    natural_languages = get_natural_languages(df, lang_list=natural_lang_list, n=natural_n)
    coding_languages = load_coding_languages(n=code_n)

    all_languages: List[Dict[str, str]] = (
        [english] + natural_languages + coding_languages
    )

    print(
        f"📊 Languages: {len(all_languages)} (1 Eng, {len(natural_languages)} nat, {len(coding_languages)} code)"
    )

    # Prepare per-tokenizer result containers
    all_results: Dict[str, Dict[str, Any]] = {
        tok.name: {
            "tokenizer": tok.name,
            "vocab_size": tok.vocab_size,
            "vocab_metrics": vocab_metrics_by_tok[tok.name],
            "benchmark_size_mb": sample_size_mb,
            "timestamp": time.time(),
            "languages": {},
            "global_metrics": {},
        }
        for tok in tokenizers
    }

    # Global trackers per tokenizer
    global_trackers: Dict[str, GlobalMetricsTracker] = {
        tok.name: GlobalMetricsTracker() for tok in tokenizers
    }

    # Process languages in parallel (one unit of work per language)
    print("🏃 Running...")
    effective_workers = max(1, min(max_workers, len(all_languages)))
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                _process_single_language_with_retry,
                tokenizers,
                lang,
                sample_size_mb,
            ): lang
            for lang in all_languages
        }

        completed = 0
        total = len(all_languages)
        for future in as_completed(futures):
            lang_info = futures[future]
            try:
                per_tok_metrics, per_tok_tokens = future.result()
            except Exception as e:
                print(f"  ❌ {lang_info.get('name', 'Unknown')}: {e}")
                completed += 1
                continue

            # Unique key per language
            if lang_info["source"] == "starcoder":
                key = f"{lang_info['iso_code']}-code"
            else:
                key = f"{lang_info['iso_code']}-{lang_info['script']}"

            # Store per-tokenizer language results and update trackers
            for tok_name, lang_result in per_tok_metrics.items():
                all_results[tok_name]["languages"][key] = lang_result
            for tok_name, tokens_info in per_tok_tokens.items():
                global_trackers[tok_name].add_tokens(tokens_info)

            completed += 1
            # Simple inline progress bar - update every language or every 5% of progress
            progress_interval = max(
                1, total // 20
            )  # Update every 5% or at least every language
            if completed % progress_interval == 0 or completed == total:
                filled = int(PROGRESS_BAR_WIDTH * completed / total)
                bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
                print(f"\r📈 [{bar}] {completed}/{total}", end="", flush=True)
        print()  # newline after progress bar

    # Finalize global metrics
    print("🔄 Finalizing global metrics...")
    for tok_name, tracker in global_trackers.items():
        all_results[tok_name]["global_metrics"] = tracker.get_global_metrics()
        # Group per-language metrics by script (worst-case aware summaries)
        all_results[tok_name]["script_aggregations"] = aggregate_by_script(
            all_results[tok_name]["languages"]
        )

    # Save results per tokenizer
    saved_files: List[str] = []
    for i, tok_name in enumerate(tokenizer_names):
        results = all_results[tok_name]

        # Determine output filename
        if output_names and i < len(output_names):
            output_filename = f"{output_names[i]}.json"
        else:
            safe_name = tok_name.replace("/", "_").replace("-", "_")
            output_filename = f"{safe_name}.json"

        output_path = os.path.join("data", "results", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, sort_keys=True)
        saved_files.append(output_path)
        print(f"💾 {output_path}")

    # Cleanup
    try:
        gc.collect()
    except (RuntimeError, MemoryError) as e:
        # Log but don't fail on cleanup issues
        print(f"Warning: Cleanup failed: {e}")

    print("✅ Done")
    if len(saved_files) > 1:
        print(f"📁 Files: {', '.join(saved_files)}")

    return all_results
