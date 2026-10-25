"""Tests for weighted_shuffle: the ordering behind every search result page.

The sampling used to be a loop that re-normalised the whole remaining array on
every pick, which is O(n^2) and cost seconds inside a search request. It is now
one sort via the Gumbel top-k identity, so what these tests mostly guard is that
the *distribution* did not change along with the speed.
"""
import numpy as np
import pytest

from anagnorisis_core.search.ranking import weighted_shuffle


DESCENDING = np.array([5.0, 4.0, 3.0, 2.0, 1.0])


def _reference(scores, temperature):
    """The original loop, kept here as the thing the new code must agree with."""
    remaining = list(range(len(scores)))
    indices = []
    scores = np.array(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=-np.inf)
    while remaining:
        current = np.clip(scores[remaining], 1e-9, None)
        adjusted = current ** (1 / temperature)
        total = adjusted.sum()
        if total == 0 or not np.isfinite(total):
            probs = np.ones(len(remaining)) / len(remaining)
        else:
            probs = adjusted / total
        pick = np.random.choice(len(remaining), p=probs)
        indices.append(remaining.pop(pick))
    return indices


class TestOrdering:
    def test_temperature_zero_is_strictly_descending(self):
        order = weighted_shuffle(DESCENDING, temperature=0)
        assert [DESCENDING[i] for i in order] == [5.0, 4.0, 3.0, 2.0, 1.0]

    @pytest.mark.parametrize('temperature', [0, 0.2, 1.0, 3.0])
    def test_always_a_permutation(self, temperature):
        """Every file has to appear exactly once: this is the page order, so a
        dropped index loses a result and a repeated one shows a duplicate."""
        scores = np.random.default_rng(0).random(500)
        order = weighted_shuffle(scores, temperature=temperature)
        assert sorted(order) == list(range(len(scores)))

    def test_unscored_files_sort_last(self):
        """NaN means the file could not be embedded. It must not poison the
        ordering of the files that were scored."""
        scores = np.array([1.0, np.nan, 5.0, np.nan, 3.0])
        order = weighted_shuffle(scores, temperature=0)
        assert order[:3] == [2, 4, 0], order
        assert set(order[3:]) == {1, 3}, order

    def test_empty_input(self):
        assert weighted_shuffle(np.array([]), temperature=0) == []
        assert weighted_shuffle(np.array([]), temperature=0.5) == []


class TestReproducibility:
    """The app seeds the *global* numpy RNG so a given seed keeps paging stable
    while the user clicks through results."""

    def test_same_seed_gives_the_same_order(self):
        np.random.seed(42)
        first = weighted_shuffle(DESCENDING, temperature=0.5)
        np.random.seed(42)
        second = weighted_shuffle(DESCENDING, temperature=0.5)
        assert first == second

    def test_different_seeds_differ(self):
        np.random.seed(1)
        first = weighted_shuffle(np.random.default_rng(0).random(50), temperature=0.5)
        np.random.seed(2)
        second = weighted_shuffle(np.random.default_rng(0).random(50), temperature=0.5)
        assert first != second


class TestDistributionIsUnchanged:
    """The speed-up must not have changed which files come first."""

    @pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0])
    def test_first_pick_matches_the_analytic_probability(self, temperature):
        trials = 20000
        counts = np.zeros(len(DESCENDING))
        np.random.seed(7)
        for _ in range(trials):
            counts[weighted_shuffle(DESCENDING, temperature)[0]] += 1
        observed = counts / trials

        weights = DESCENDING ** (1 / temperature)
        expected = weights / weights.sum()

        # Sampling error at 20k trials is well under 0.01 per bucket.
        assert np.abs(observed - expected).max() < 0.015, (
            f'observed {observed} vs expected {expected}')

    def test_agrees_with_the_original_loop(self):
        """Compared against the implementation this replaced, not just theory."""
        trials = 8000
        temperature = 1.0
        new = np.zeros(len(DESCENDING))
        old = np.zeros(len(DESCENDING))
        np.random.seed(11)
        for _ in range(trials):
            new[weighted_shuffle(DESCENDING, temperature)[0]] += 1
        np.random.seed(11)
        for _ in range(trials):
            old[_reference(DESCENDING, temperature)[0]] += 1
        assert np.abs(new / trials - old / trials).max() < 0.02
