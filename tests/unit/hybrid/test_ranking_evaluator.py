import pandas as pd
import pytest

from src.hybrid.ranking_evaluator import HybridRankingEvaluator


def test_perfect_matches_have_precision_and_recall_one():
    evaluator = HybridRankingEvaluator()
    held_out = pd.DataFrame(
        {"userId": [1, 1], "movieId": [10, 11], "rating": [5.0, 4.0]}
    )

    result = evaluator.evaluate(
        {1: [10, 11, 12]},
        held_out,
        k=2,
    )

    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0


def test_partial_overlap_uses_k_as_precision_denominator():
    evaluator = HybridRankingEvaluator()
    held_out = pd.DataFrame(
        {"userId": [1, 1], "movieId": [10, 11], "rating": [5.0, 4.0]}
    )

    result = evaluator.evaluate(
        {1: [10, 99]},
        held_out,
        k=2,
    )

    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 0.5


def test_no_relevant_recommendations_scores_zero():
    evaluator = HybridRankingEvaluator()
    held_out = pd.DataFrame(
        {"userId": [1], "movieId": [10], "rating": [5.0]}
    )

    result = evaluator.evaluate({1: [20, 30]}, held_out, k=2)

    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0


def test_users_without_relevant_held_out_movies_are_excluded_and_counted():
    evaluator = HybridRankingEvaluator()
    held_out = pd.DataFrame(
        {"userId": [1, 2], "movieId": [10, 20], "rating": [5.0, 2.0]}
    )

    result = evaluator.evaluate(
        {1: [10], 2: [20]},
        held_out,
        k=1,
        user_ids=[1, 2],
    )

    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0
    assert result["evaluated_users"] == 1
    assert result["users_without_relevant_held_out"] == 1


def test_short_recommendation_list_still_uses_k_denominator():
    evaluator = HybridRankingEvaluator()
    held_out = pd.DataFrame(
        {"userId": [1, 1], "movieId": [10, 11], "rating": [5.0, 5.0]}
    )

    result = evaluator.evaluate({1: [10]}, held_out, k=3)

    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["recall_at_k"] == 0.5


def test_split_is_reproducible_and_keeps_training_history_per_user():
    evaluator = HybridRankingEvaluator()
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3],
            "movieId": [10, 11, 12, 20, 21, 22, 30],
            "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 3.0, 4.0],
        }
    )

    train, held_out = evaluator.split_ratings(ratings, random_state=7)
    repeated_train, repeated_held_out = evaluator.split_ratings(
        ratings,
        random_state=7,
    )

    pd.testing.assert_frame_equal(train, repeated_train)
    pd.testing.assert_frame_equal(held_out, repeated_held_out)
    assert set(train["userId"]) == {1, 2, 3}
    assert not set(
        zip(train["userId"], train["movieId"])
    ).intersection(zip(held_out["userId"], held_out["movieId"]))
    assert train.groupby("userId").size().loc[1] >= 1
    assert train.groupby("userId").size().loc[2] >= 1
    assert train.groupby("userId").size().loc[3] == 1


def test_invalid_k_is_rejected():
    evaluator = HybridRankingEvaluator()
    held_out = pd.DataFrame(
        {"userId": [1], "movieId": [10], "rating": [5.0]}
    )

    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluator.evaluate({1: [10]}, held_out, k=0)
