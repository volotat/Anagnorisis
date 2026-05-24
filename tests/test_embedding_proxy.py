"""
Tests for src/embedding_proxy.py — quantize_embedding()

Covers:
  - Zero/near-zero embedding returns empty string
  - Output length equals input dimensionality
  - All output characters belong to the 32-char alphabet
  - Each character appears roughly equally often (histogram-equalisation guarantee)
  - Two similar embeddings produce strings with high character overlap
  - Two very different (orthogonal) embeddings produce different strings
  - Deterministic output for the same input
  - Works for typical model dimensions (512 CLAP, 768 SigLIP)
"""
import numpy as np
import pytest
from collections import Counter
from src.embedding_proxy import quantize_embedding, _CHARS, _LEVELS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rand_unit(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _char_overlap_ratio(s1: str, s2: str) -> float:
    """Fraction of positions where both strings have the same character."""
    tokens1 = s1.split()
    tokens2 = s2.split()
    if len(tokens1) != len(tokens2) or not tokens1:
        return 0.0
    matches = sum(a == b for a, b in zip(tokens1, tokens2))
    return matches / len(tokens1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQuantizeEmbedding:
    def test_zero_embedding_returns_empty(self):
        emb = np.zeros(512, dtype=np.float32)
        assert quantize_embedding(emb) == ''

    def test_near_zero_embedding_returns_empty(self):
        emb = np.full(512, 1e-8, dtype=np.float32)
        assert quantize_embedding(emb) == ''

    def test_output_has_correct_number_of_tokens(self):
        for dim in (512, 768):
            emb = _rand_unit(dim)
            result = quantize_embedding(emb)
            tokens = result.split()
            assert len(tokens) == dim, f"Expected {dim} tokens for dim={dim}, got {len(tokens)}"

    def test_all_chars_in_alphabet(self):
        emb = _rand_unit(768)
        result = quantize_embedding(emb)
        alphabet = set(_CHARS)
        for ch in result.split():
            assert ch in alphabet, f"Character '{ch}' not in alphabet"

    def test_histogram_equalization(self):
        """Each of the 32 characters should appear roughly dim/32 times."""
        dim = 512
        emb = _rand_unit(dim)
        tokens = quantize_embedding(emb).split()
        counts = Counter(tokens)
        expected = dim / _LEVELS
        for ch in _CHARS:
            count = counts.get(ch, 0)
            # Allow generous tolerance: ±50% of expected count
            assert count >= expected * 0.5, (
                f"Character '{ch}' appears only {count} times, expected ~{expected:.1f}"
            )
            assert count <= expected * 1.5, (
                f"Character '{ch}' appears {count} times, expected ~{expected:.1f}"
            )

    def test_similar_embeddings_high_char_overlap(self):
        """A very small perturbation should preserve most character positions.

        Rank-based quantization is sensitive to rank inversions: individual
        component std ≈ 1/sqrt(512) ≈ 0.044, so noise must be far smaller
        than that to keep most ranks intact.  We use noise_scale=1e-4 which
        gives cosine similarity > 0.9999 and preserves the majority of ranks.
        """
        dim = 512
        base = _rand_unit(dim)
        # Tiny noise: std=1e-4 << component std ≈ 0.044
        noisy = base + np.random.default_rng(99).standard_normal(dim).astype(np.float32) * 1e-4
        noisy = noisy / np.linalg.norm(noisy)

        s_base  = quantize_embedding(base)
        s_noisy = quantize_embedding(noisy)

        overlap = _char_overlap_ratio(s_base, s_noisy)
        assert overlap > 0.5, f"Expected high overlap for nearly-identical embeddings, got {overlap:.2f}"

    def test_orthogonal_embeddings_low_char_overlap(self):
        """Two orthogonal embeddings should produce largely different strings."""
        dim = 512
        rng = np.random.default_rng(7)
        v1 = rng.standard_normal(dim).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        # Build orthogonal vector via Gram-Schmidt
        v2 = rng.standard_normal(dim).astype(np.float32)
        v2 -= np.dot(v2, v1) * v1
        v2 /= np.linalg.norm(v2)

        s1 = quantize_embedding(v1)
        s2 = quantize_embedding(v2)

        overlap = _char_overlap_ratio(s1, s2)
        # For orthogonal embeddings, random chance ≈ 1/32 = 0.03; allow up to 0.15
        assert overlap < 0.15, f"Orthogonal embeddings should differ more, overlap={overlap:.2f}"

    def test_deterministic(self):
        emb = _rand_unit(512, seed=42)
        assert quantize_embedding(emb) == quantize_embedding(emb)

    def test_clap_dimension(self):
        """CLAP embeddings are 512-dim — verify no error and correct length."""
        emb = _rand_unit(512)
        tokens = quantize_embedding(emb).split()
        assert len(tokens) == 512

    def test_siglip_dimension(self):
        """SigLIP embeddings are 768-dim — verify no error and correct length."""
        emb = _rand_unit(768)
        tokens = quantize_embedding(emb).split()
        assert len(tokens) == 768


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
