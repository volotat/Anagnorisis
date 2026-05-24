import datetime
import numpy as np
from src.utils import weighted_shuffle
from typing import List

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sort_files_by_recommendation(files_list:List[str], files_data:List[dict]) -> List[float]:
    """
    Expects each files dict to have:
      - hash (str)
      - user_rating (may be None)
      - model_rating (may be None)
      - full_play_count (int)
      - skip_count (int)
      - last_played (datetime or None)

    Returns the files list and corresponding scores in the chosen order.

    Important! The files_data must correspond to files_list in order.
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

    # order = weighted_shuffle(final_scores, temperature=temperature)

    # sorted_files = [files_list[i] for i in order]
    # sorted_scores = final_scores[order]

    return final_scores

# ------------------- TESTS -------------------
def _make_music_data():
    return [
        {'user_rating': 8.0, 'model_rating': None, 'full_play_count': 10, 'skip_count': 2, 'last_played': datetime.datetime.fromtimestamp(100)},
        {'user_rating': None, 'model_rating': 6.0, 'full_play_count': 5,  'skip_count': 1, 'last_played': datetime.datetime.fromtimestamp(50)},
        {'user_rating': 9.0, 'model_rating': None, 'full_play_count': 12, 'skip_count': 0, 'last_played': datetime.datetime.fromtimestamp(120)},
        {'user_rating': None, 'model_rating': None, 'full_play_count': 3,  'skip_count': 3, 'last_played': None},
    ]

def test_weighted_shuffle_zero():
    # Test that weighted_shuffle works correctly when all scores are 0.
    scores = np.array([0, 0, 0, 0])
    order = weighted_shuffle(scores)
    assert sorted(order) == list(range(4)), f"Expected permutation of indices, got {order}"
    print("test_weighted_shuffle_zero PASSED")

def test_scores_shape_and_finite():
    # sort_files_by_recommendation returns a 1-D array with one score per file.
    files_list = ['a.mp3', 'b.mp3', 'c.mp3', 'd.mp3']
    music_data = _make_music_data()
    scores = sort_files_by_recommendation(files_list, music_data)
    assert len(scores) == len(files_list), f"Expected {len(files_list)} scores, got {len(scores)}"
    assert np.isfinite(scores).all(), f"Scores contain non-finite values: {scores}"
    assert (scores >= 0).all(), f"Scores must be non-negative: {scores}"
    print("test_scores_shape_and_finite PASSED")

def test_higher_rating_higher_score():
    # File with user_rating=9 should outrank file with user_rating=4 when other
    # factors are equal (same play / skip counts, same last_played).
    files_list = ['low.mp3', 'high.mp3']
    music_data = [
        {'user_rating': 4.0, 'model_rating': None, 'full_play_count': 5, 'skip_count': 0, 'last_played': datetime.datetime.fromtimestamp(50)},
        {'user_rating': 9.0, 'model_rating': None, 'full_play_count': 5, 'skip_count': 0, 'last_played': datetime.datetime.fromtimestamp(50)},
    ]
    scores = sort_files_by_recommendation(files_list, music_data)
    assert scores[1] > scores[0], f"Higher rating should produce higher score: {scores}"
    print("test_higher_rating_higher_score PASSED")

def test_all_none_ratings_no_crash():
    # When every file has no rating at all, function must not crash and must return
    # valid (finite, non-negative) scores.
    files_list = ['x.mp3', 'y.mp3']
    music_data = [
        {'user_rating': None, 'model_rating': None, 'full_play_count': 0, 'skip_count': 0, 'last_played': None},
        {'user_rating': None, 'model_rating': None, 'full_play_count': 0, 'skip_count': 0, 'last_played': None},
    ]
    scores = sort_files_by_recommendation(files_list, music_data)
    assert len(scores) == 2
    assert np.isfinite(scores).all(), f"All-None case produced non-finite scores: {scores}"
    print("test_all_none_ratings_no_crash PASSED")

def test_single_file_no_crash():
    # Edge case: single file list must not raise (e.g. division in last_played normalisation).
    files_list = ['solo.mp3']
    music_data = [
        {'user_rating': 7.0, 'model_rating': None, 'full_play_count': 3, 'skip_count': 1, 'last_played': datetime.datetime.fromtimestamp(200)},
    ]
    scores = sort_files_by_recommendation(files_list, music_data)
    assert len(scores) == 1
    assert np.isfinite(scores[0]), f"Single-file score not finite: {scores[0]}"
    print("test_single_file_no_crash PASSED")

if __name__ == "__main__":
    print("Running recommendation_engine tests...")
    test_weighted_shuffle_zero()
    test_scores_shape_and_finite()
    test_higher_rating_higher_score()
    test_all_none_ratings_no_crash()
    test_single_file_no_crash()
    print("\nAll tests PASSED.")

    # Example usage.
    files = ['a.mp3', 'b.mp3', 'c.mp3', 'd.mp3']
    data  = _make_music_data()
    scores = sort_files_by_recommendation(files, data)
    print("\nExample scores (unsorted, aligned with input):")
    for f, s in zip(files, scores):
        print(f"  {f} => {s:.4f}")