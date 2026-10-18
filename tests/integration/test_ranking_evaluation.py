import pandas as pd
import pytest

from scripts.hybrid import evaluate_ranking


pytestmark = pytest.mark.integration


def test_held_out_interactions_are_not_training_or_seen_filter_data(monkeypatch):
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [10, 11, 12, 20, 21, 22],
            "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 3.0],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 11, 12, 20, 21, 22],
            "title": ["A", "B", "C", "D", "E", "F"],
            "genres": ["Drama", "Comedy", "Action", "Drama", "Comedy", "Action"],
        }
    )
    evaluator = evaluate_ranking.HybridRankingEvaluator()
    expected_train, expected_held_out = evaluator.split_ratings(ratings)

    captured = {"train": None, "recommended": []}

    monkeypatch.setattr(
        evaluate_ranking,
        "_build_model",
        lambda train_ratings, random_state: (
            object(),
            object(),
        ),
    )

    class FakePreprocessor:
        def preprocess_movies(self, frame):
            return frame.copy()

    class FakeFeatureEngineer:
        @staticmethod
        def build_genre_matrix(frame):
            return frame[["movieId"]].copy()

    class FakeCollaborativeRecommender:
        def __init__(self, **kwargs):
            pass

    class FakeHybridRecommender:
        def __init__(self, **kwargs):
            self.ratings = kwargs["ratings"].copy()
            captured["train"] = self.ratings

        def recommend(self, user_id, **kwargs):
            train_seen = set(
                self.ratings.loc[
                    self.ratings["userId"] == user_id,
                    "movieId",
                ]
            )
            held_out_for_user = expected_held_out.loc[
                expected_held_out["userId"] == user_id,
                "movieId",
            ].tolist()
            candidate = next(
                movie_id for movie_id in held_out_for_user
                if movie_id not in train_seen
            )
            captured["recommended"].append((user_id, candidate))
            return {
                "special": pd.DataFrame(columns=["movieId"]),
                "content_based": pd.DataFrame({"movieId": [candidate]}),
                "collaborative": pd.DataFrame(columns=["movieId"]),
            }

    monkeypatch.setattr(evaluate_ranking, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(evaluate_ranking, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(
        evaluate_ranking,
        "ContentBasedRecommender",
        lambda: object(),
    )
    monkeypatch.setattr(
        evaluate_ranking,
        "CollaborativeRecommender",
        FakeCollaborativeRecommender,
    )
    monkeypatch.setattr(
        evaluate_ranking,
        "HybridRecommender",
        FakeHybridRecommender,
    )

    results = evaluate_ranking.run_evaluation(
        ratings=ratings,
        movies=movies,
        k=1,
        relevance_threshold=4.0,
    )

    pd.testing.assert_frame_equal(
        captured["train"].sort_values(["userId", "movieId"]).reset_index(drop=True),
        expected_train.sort_values(["userId", "movieId"]).reset_index(drop=True),
    )
    assert not set(
        zip(captured["train"]["userId"], captured["train"]["movieId"])
    ).intersection(
        zip(expected_held_out["userId"], expected_held_out["movieId"])
    )
    assert captured["recommended"]
    assert results["held_out_interactions"] == len(expected_held_out)
    assert results["k"] == 1
