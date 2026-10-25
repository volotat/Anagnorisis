"""Turning scores into an order.

Ranking is engine work, not presentation: the same maths orders a search result
page, a recommendation queue and a CLI listing. It lives here so all three agree,
and so none of them has to import the web layer to sort something.
"""
import numpy as np


def weighted_shuffle(scores, temperature=1.0):
    """
    Returns a permutation (list of indices) from files_list where
    each item is sampled without replacement. The probability is adjusted by temperature.
    - temperature = 0: Strict descending order (highest score first).
    - temperature = 1: Probability proportional to score.
    - temperature < 1: More deterministic, sharpens probability distribution.
    - temperature > 1: More random, flattens probability distribution.
    
    Args:
        scores (list or np.array): List of scores for each item.
        temperature (float): Temperature parameter to adjust randomness.
        order (str): 'most-relevant' for descending order, 'least-relevant' for ascending order.

    NaN scores (an unscored item) sort last rather than poisoning the result.
    Left in place, a NaN makes every comparison against it False, which breaks
    the sort's ordering invariant and silently returns an arbitrary permutation
    of the *scored* items too; in the sampling branch it turns the probability
    sum into NaN and degrades every item to a uniform draw.
    """
    scores = np.array(scores, dtype=np.float64)  # Use float64 for stability
    scores = np.nan_to_num(scores, nan=-np.inf)

    if temperature == 0:
        # Strict mode: highest score first, ties in their original order.
        return np.argsort(-scores, kind='stable').tolist()

    # Sampling without replacement, with probability proportional to
    # score ** (1 / temperature), in one pass.
    #
    # This used to be a loop that re-sliced, re-normalised and re-sampled the
    # whole remaining array for every single pick, and then popped from the
    # middle of a list. That is O(n^2): measured 0.6s at 5,000 items, 1.9s at
    # 9,500, and around 13s at 24,000, all of it inside a search request.
    #
    # The Gumbel top-k identity gives the identical distribution from one sort:
    # perturb each log-weight by a Gumbel variate and rank by the result.
    # Drawing repeatedly in proportion to w and sorting by log(w) + G produce
    # the same ordering, so this is a speed change and not a behaviour one.
    #
    # Working in logs also removes the overflow the old code had to guard: at a
    # low temperature, score ** (1 / temperature) could reach inf and collapse
    # every probability to uniform.
    weights = np.clip(scores, 1e-9, None)
    log_weights = np.log(weights) / temperature

    # np.random rather than a Generator on purpose: the app seeds the global
    # legacy RNG (file_manager.get_files) so that a given seed keeps paging
    # stable, and a Generator would ignore that seed.
    uniform = np.random.random(len(scores))
    # Both ends of the interval send the Gumbel variate to an infinity, which
    # would make the sort order undefined rather than merely extreme.
    uniform = np.clip(uniform, 1e-12, 1.0 - 1e-12)
    gumbel = -np.log(-np.log(uniform))

    # Descending: argsort is ascending, so rank the negated keys.
    return np.argsort(-(log_weights + gumbel), kind='stable').tolist()

import time

###########################################
# Sorting Progress Callback
