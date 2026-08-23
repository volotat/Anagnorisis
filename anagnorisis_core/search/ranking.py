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
    remaining = list(range(len(scores)))
    indices = []
    scores = np.array(scores, dtype=np.float64)  # Use float64 for stability
    scores = np.nan_to_num(scores, nan=-np.inf)

    if temperature == 0:
        # Strict mode: sort remaining indices by their scores in descending order
        # and append them all at once.
        sorted_remaining_indices = sorted(remaining, key=lambda i: scores[i], reverse=True)
        return sorted_remaining_indices

    while remaining:
        # Apply temperature to the scores of the remaining items
        current_scores = scores[remaining]
        
        # Prevent overflow/underflow with very high/low scores
        current_scores = np.clip(current_scores, 1e-9, None)
        
        # The core of the temperature logic
        adjusted_scores = current_scores ** (1 / temperature)

        total = adjusted_scores.sum()
        if total == 0 or not np.isfinite(total):
            # If total is 0 or inf, assign uniform probabilities.
            probs = np.ones(len(remaining)) / len(remaining)
        else:
            probs = adjusted_scores / total
        
        pick = np.random.choice(len(remaining), p=probs)
        indices.append(remaining.pop(pick))
    return indices

import time

###########################################
# Sorting Progress Callback
