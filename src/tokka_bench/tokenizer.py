"""
Universal tokenizer wrapper for HuggingFace models.

This module provides the UniversalTokenizer class that can load any HuggingFace
tokenizer and calculate various metrics including vocabulary analysis and
word-level statistics.
"""

from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from .metrics import analyze_vocabulary, calculate_word_metrics, get_token_analysis

# Constants
DEFAULT_SAMPLE_TOKENIZATIONS_LIMIT = 5
DEFAULT_TRUST_REMOTE_CODE = False


class RawTokenizerAdapter:
    """Thin adapter exposing the minimal interface tokka-bench needs.

    Wraps a ``tokenizers.Tokenizer`` (loaded directly from a model's
    ``tokenizer.json``) so the rest of the codebase can use it like a
    HuggingFace tokenizer. Used as a fallback when ``AutoTokenizer`` cannot
    load a model (e.g. when a tokenizer config uses a format the installed
    ``transformers`` version does not understand, as with Gemma 4).
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._tokenizer.get_vocab())

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = True
    ) -> str:
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )


def _load_raw_tokenizer(model_name: str) -> RawTokenizerAdapter:
    """Load a tokenizer directly from a model's ``tokenizer.json`` file.

    Uses the ``tokenizers`` library, bypassing ``transformers`` special-token
    handling. Raises ``RuntimeError`` if the raw tokenizer cannot be loaded.
    """
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as e:  # pragma: no cover - tokenizers always present
        raise RuntimeError(
            f"Cannot load raw tokenizer for '{model_name}': {e}"
        ) from e

    try:
        tokenizer_path = hf_hub_download(
            repo_id=model_name, filename="tokenizer.json"
        )
        raw_tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load raw tokenizer for '{model_name}': {e}"
        ) from e

    return RawTokenizerAdapter(raw_tokenizer)


class UniversalTokenizer:
    """Universal tokenizer that can load any HuggingFace model."""

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE,
        verbose: bool = True,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty or None")

        self.name: str = model_name
        self.verbose: bool = verbose

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
        except (OSError, ValueError, ConnectionError) as e:
            raise RuntimeError(f"Failed to load tokenizer '{model_name}': {e}") from e
        except Exception as e:
            # Fall back to loading the raw `tokenizers` tokenizer from
            # tokenizer.json. This handles models (e.g. Gemma 4) whose
            # tokenizer_config.json uses a format the installed transformers
            # version cannot parse, even though the underlying tokenizer is valid.
            if self.verbose:
                print(
                    f"    ⚠️  AutoTokenizer failed ({e}); falling back to raw tokenizer.json"
                )
            self.tokenizer = _load_raw_tokenizer(model_name)

        # Get vocabulary size
        try:
            self.vocab_size: int = len(self.tokenizer)
        except (AttributeError, TypeError) as e:
            raise RuntimeError(
                f"Invalid tokenizer vocabulary for '{model_name}': {e}"
            ) from e

        # Calculate vocabulary-level metrics once
        try:
            self.vocab_metrics: Dict[str, Any] = analyze_vocabulary(self)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not analyze vocabulary: {e}")
            self.vocab_metrics = {}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if text is None:
            raise ValueError("text cannot be None")

        try:
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except (ValueError, RuntimeError, AttributeError) as e:
            raise RuntimeError(f"Failed to encode text: {e}") from e

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if token_ids is None:
            raise ValueError("token_ids cannot be None")

        try:
            return self.tokenizer.decode(
                token_ids, skip_special_tokens=skip_special_tokens
            )
        except (ValueError, RuntimeError, KeyError, AttributeError) as e:
            raise RuntimeError(f"Failed to decode token IDs: {e}") from e

    def get_metrics(
        self,
        text: str,
        language_info: Optional[Dict[str, str]] = None,
        debug: bool = True,
    ) -> Dict[str, Any]:
        """Calculate metrics for given text."""
        if not text:
            raise ValueError("text cannot be empty or None")

        # Encode the text
        # Exclude special tokens (e.g., BOS/EOS) so metrics reflect only content
        token_ids: List[int] = self.encode(text, add_special_tokens=False)

        # Basic metrics from the actual text we're tokenizing
        text_bytes: int = len(text.encode("utf-8"))
        num_tokens: int = len(token_ids)
        unique_tokens: int = len(set(token_ids))  # Count unique token IDs

        # Calculate word-level metrics with language information, reusing tokenization
        word_metrics: Dict[str, Any] = calculate_word_metrics(
            self,
            text,
            language_info,
            pretokenized_text_token_ids=token_ids,
        )

        # Debug logging (only if enabled)
        debug_info: Dict[str, Any] = word_metrics.get("debug_info", {})

        if debug and self.verbose:
            segmentation_method = debug_info.get("segmentation_method", "whitespace")

            print("    🔍 Debug Info:")
            print(
                f"      Text bytes: {text_bytes:,}, Tokens: {num_tokens:,}, Unique tokens: {unique_tokens:,}"
            )
            print(
                f"      Segmentation: {segmentation_method.title()} | Units: {debug_info.get('total_words', 0):,}, Units split: {debug_info.get('words_split', 0):,}"
            )
            print(
                f"      Continued word rate: {word_metrics['continued_word_rate']:.1f}%"
            )

        return {
            "bytes_per_token": text_bytes / num_tokens if num_tokens > 0 else 0,
            "total_bytes": text_bytes,
            "total_tokens": num_tokens,
            "unique_tokens": unique_tokens,
            "subword_fertility": word_metrics["subword_fertility"],
            "continued_word_rate": word_metrics["continued_word_rate"],
            "debug_info": debug_info,  # Include debug info in output
        }

    def get_token_analysis(self, text: str) -> Dict[str, Any]:
        """Get detailed token analysis for global metrics."""
        if not text:
            raise ValueError("text cannot be empty or None")

        try:
            return get_token_analysis(self, text)
        except Exception as e:
            raise RuntimeError(f"Failed to analyze tokens: {e}") from e
