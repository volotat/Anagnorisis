import datetime
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def weighted_shuffle(scores):
    """
    Returns a permutation (list of indices) from files_list where
    each item is sampled without replacement with a probability proportional to its score.
    """
    remaining = list(range(len(scores)))
    order = []
    scores = np.array(scores)  # ensure it is a NumPy array
    while remaining:
        total = scores[remaining].sum()
        if total == 0:
            # If total weight is 0, assign uniform probabilities.
            probs = np.ones(len(remaining)) / len(remaining)
        else:
            probs = scores[remaining] / total
        pick = np.random.choice(len(remaining), p=probs)
        order.append(remaining.pop(pick))
    return order

def sort_files_by_recommendation(files_list, files_data, mode='random'):
    """
    Expects each files dict to have:
      - user_rating (may be None)
      - model_rating (may be None)
      - full_play_count (int)
      - skip_count (int)
      - last_played (datetime or None)
    The 'mode' parameter can be:
      - 'random': perform a weighted shuffle based on the recommendation score.
      - 'strict': sort strictly in descending order of recommendation score.
    Returns the files list and corresponding scores in the chosen order.
    """
    # Fallback: when no user rating is provided use median/mean from those available.
    not_none_ratings = np.array([m['user_rating'] for m in files_data if m['user_rating'] is not None])
    if len(not_none_ratings) > 0:
        median_value = np.median(not_none_ratings)
        mean_value = np.mean(not_none_ratings)
    else:
        median_value = 5.0
        mean_value = 5.0

    # Base scores: use the user rating if available; else model_rating; else fallback.
    base_scores = []
    for m in files_data:
        if m['user_rating'] is not None:
            base_scores.append(m['user_rating'])
        elif m['model_rating'] is not None:
            base_scores.append(m['model_rating'])
        else:
            base_scores.append(mean_value)
    base_scores = np.array(base_scores)
    base_scores = np.maximum(0.1, base_scores)  # ensure a minimum value
    base_scores = (base_scores / 10) ** 2       # enhance differences

    # Skip adjustment: encourage songs with lower skip counts relative to full plays.
    full_play_count = np.array([m['full_play_count'] for m in files_data])
    skip_count = np.array([m['skip_count'] for m in files_data])
    skip_score = sigmoid((5 + full_play_count - skip_count) / 5)

    # Last played adjustment: songs not played recently get a higher weight.
    def lp_value(m):
        return m['last_played'].timestamp() if m['last_played'] is not None else 0
    sorted_lp = sorted(files_data, key=lp_value, reverse=True)
    lp_indices = { id(m): index for index, m in enumerate(sorted_lp) }
    last_played_score = np.array([ lp_indices.get(id(m), 0) for m in files_data ])
    if len(files_data) > 1:
        last_played_score = last_played_score / (len(files_data) - 1)
    
    # Compute final recommendation score.
    final_scores = base_scores * skip_score * last_played_score
    if not np.isfinite(final_scores).all():
        print("Warning: Final scores contain non-finite values.")
        final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1.0, neginf=0.0)

    if mode == 'random':
        order = weighted_shuffle(final_scores)
    elif mode == 'strict':
        order = np.argsort(final_scores)[::-1]
    else:
        raise ValueError("mode must be either 'random' or 'strict'")

    sorted_files = [files_list[i] for i in order]
    sorted_scores = final_scores[order]
    return sorted_files, sorted_scores

# ------------------- TESTS -------------------
def test_weighted_shuffle_zero():
    # Test that weighted_shuffle works correctly when all scores are 0.
    scores = np.array([0, 0, 0, 0])
    order = weighted_shuffle(scores)
    assert sorted(order) == list(range(4)), f"Expected permutation of indices, got {order}"
    print("test_weighted_shuffle_zero PASSED")

def test_strict_order():
    # Prepare sample data with datetime objects for last_played.
    music_list = [
        {'user_rating': 8.0, 'model_rating': None, 'full_play_count': 10, 'skip_count': 2, 'last_played': datetime.datetime.fromtimestamp(100)},
        {'user_rating': None, 'model_rating': 6.0, 'full_play_count': 5, 'skip_count': 1, 'last_played': datetime.datetime.fromtimestamp(50)},
        {'user_rating': 9.0, 'model_rating': None, 'full_play_count': 12, 'skip_count': 0, 'last_played': datetime.datetime.fromtimestamp(120)},
        {'user_rating': None, 'model_rating': None, 'full_play_count': 3, 'skip_count': 3, 'last_played': None},  # Not played yet
    ]
    sorted_files, sorted_scores = sort_files_by_recommendation(music_list, music_list, mode='strict')
    # Check that scores are in descending order.
    for i in range(len(sorted_scores) - 1):
        assert sorted_scores[i] >= sorted_scores[i+1], "Strict order not descending"
    print("test_strict_order PASSED")

def test_random_order():
    # Using the same sample data, test that random order is a permutation of strict order.
    music_list = [
        {'user_rating': 8.0, 'model_rating': None, 'full_play_count': 10, 'skip_count': 2, 'last_played': datetime.datetime.fromtimestamp(100)},
        {'user_rating': None, 'model_rating': 6.0, 'full_play_count': 5, 'skip_count': 1, 'last_played': datetime.datetime.fromtimestamp(50)},
        {'user_rating': 9.0, 'model_rating': None, 'full_play_count': 12, 'skip_count': 0, 'last_played': datetime.datetime.fromtimestamp(120)},
        {'user_rating': None, 'model_rating': None, 'full_play_count': 3, 'skip_count': 3, 'last_played': None},
    ]
    # To help with reproducibility in tests.
    np.random.seed(42)
    sorted_files_random, sorted_scores_random = sort_files_by_recommendation(music_list, music_list, mode='random')
    np.random.seed(42)
    # Running strict sort for comparison.
    sorted_files_strict, sorted_scores_strict = sort_files_by_recommendation(music_list, music_list, mode='strict')
    # They are not expected to be equal overall; however, the set of files should match.
    assert set(tuple(m.items()) for m in sorted_files_random) == set(tuple(m.items()) for m in sorted_files_strict), "Random mode did not produce a valid permutation"
    print("test_random_order PASSED")

if __name__ == "__main__":
    # Run tests.
    print("Running tests...")
    test_weighted_shuffle_zero()
    test_strict_order()
    test_random_order()
    
    # Example usage.
    _music_list = lambda: [
        {'user_rating': 8.0, 'model_rating': None, 'full_play_count': 10, 'skip_count': 2, 'last_played': datetime.datetime.fromtimestamp(100)},
        {'user_rating': None, 'model_rating': 6.0, 'full_play_count': 5, 'skip_count': 1, 'last_played': datetime.datetime.fromtimestamp(50)},
        {'user_rating': 9.0, 'model_rating': None, 'full_play_count': 12, 'skip_count': 0, 'last_played': datetime.datetime.fromtimestamp(120)},
        {'user_rating': None, 'model_rating': None, 'full_play_count': 3, 'skip_count': 3, 'last_played': None},  # Not played yet
    ]
    print("\nExample usage with strict mode:")
    sorted_music, scores = sort_files_by_recommendation(_music_list(), _music_list(), mode='strict')
    for m, s in zip(sorted_music, scores):
        print(f"{m} => recommendation score: {s}")

    print("\nExample usage with random mode:")
    sorted_music, scores = sort_files_by_recommendation(_music_list(), _music_list(), mode='random')
    for m, s in zip(sorted_music, scores):
        print(f"{m} => recommendation score: {s}")