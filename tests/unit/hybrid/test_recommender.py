import pandas as pd
import pytest

from src.hybrid.recommender import HybridRecommender


class StubContentRecommender:
    def __init__(self, recommendations):
        self.recommendations = recommendations
        self.calls = []

    def recommend(self, **kwargs):
        self.calls.append(kwargs)
        return self.recommendations.copy()


class StubCollaborativeRecommender:
    def __init__(self, recommendations):
        self.recommendations = recommendations
        self.calls = []

    def recommend(self, **kwargs):
        self.calls.append(kwargs)
        return self.recommendations.copy()


@pytest.fixture
def movies():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["Seed", "Content pick", "Collaborative pick", "Shared pick"],
            "genres": ["Drama", "Comedy", "Action", "Drama|Comedy"],
        }
    )


@pytest.fixture
def genre_matrix():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "Drama": [1, 0, 0, 1],
            "Comedy": [0, 1, 0, 1],
        }
    )


def make_recommender(movies, ratings, content_results=None, collaborative_results=None):
    content = StubContentRecommender(
        pd.DataFrame(
            content_results
            if content_results is not None
            else {"movieId": [], "similarity": []}
        )
    )
    collaborative = StubCollaborativeRecommender(
        pd.DataFrame(
            collaborative_results
            if collaborative_results is not None
            else {"movieId": [], "estimated_rating": []}
        )
    )
    return HybridRecommender(content, collaborative, movies, ratings), content, collaborative


def test_content_only_request_keeps_three_sections_and_schema(movies, genre_matrix):
    ratings = pd.DataFrame(columns=["userId", "movieId", "rating"])
    recommender, content, collaborative = make_recommender(
        movies,
        ratings,
        content_results={"movieId": [2, 3], "similarity": [0.9, 0.8]},
    )

    result = recommender.recommend(seed_movie_id=1, genre_matrix=genre_matrix)

    assert list(result) == ["special", "content_based", "collaborative"]
    assert result["content_based"]["movieId"].tolist() == [2, 3]
    assert result["special"].empty
    assert result["collaborative"].empty
    assert all(list(section.columns) == HybridRecommender.OUTPUT_COLUMNS for section in result.values())
    assert len(content.calls) == 1
    assert collaborative.calls == []


def test_collaborative_only_request_keeps_three_sections_and_schema(movies, genre_matrix):
    ratings = pd.DataFrame(
        {"userId": [7], "movieId": [1], "rating": [4.0]}
    )
    recommender, content, collaborative = make_recommender(
        movies,
        ratings,
        collaborative_results={"movieId": [2, 3], "estimated_rating": [4.8, 4.2]},
    )

    result = recommender.recommend(user_id=7)

    assert list(result) == ["special", "content_based", "collaborative"]
    assert result["collaborative"]["movieId"].tolist() == [2, 3]
    assert result["special"].empty
    assert result["content_based"].empty
    assert all(list(section.columns) == HybridRecommender.OUTPUT_COLUMNS for section in result.values())
    assert content.calls == []
    assert len(collaborative.calls) == 1


def test_two_signal_request_runs_both_recommenders(movies, genre_matrix):
    ratings = pd.DataFrame(
        {"userId": [7], "movieId": [1], "rating": [4.0]}
    )
    recommender, content, collaborative = make_recommender(
        movies,
        ratings,
        content_results={"movieId": [2, 4], "similarity": [0.9, 0.8]},
        collaborative_results={"movieId": [3, 4], "estimated_rating": [4.7, 4.5]},
    )

    result = recommender.recommend(
        user_id=7,
        seed_movie_id=1,
        genre_matrix=genre_matrix,
        special_candidate_pool_size=4,
    )

    assert len(content.calls) == 1
    assert len(collaborative.calls) == 1
    assert result["special"]["movieId"].tolist() == [4]
    assert all(list(section.columns) == HybridRecommender.OUTPUT_COLUMNS for section in result.values())


def test_request_without_usable_signal_fails_clearly(movies, genre_matrix):
    ratings = pd.DataFrame(columns=["userId", "movieId", "rating"])
    recommender, content, collaborative = make_recommender(movies, ratings)

    with pytest.raises(ValueError, match="At least one valid recommendation signal"):
        recommender.recommend()

    assert content.calls == []
    assert collaborative.calls == []


def test_unknown_user_with_seed_uses_content_only(movies, genre_matrix):
    ratings = pd.DataFrame(
        {"userId": [7], "movieId": [1], "rating": [4.0]}
    )
    recommender, content, collaborative = make_recommender(
        movies,
        ratings,
        content_results={"movieId": [2], "similarity": [0.9]},
    )

    result = recommender.recommend(
        user_id=99,
        seed_movie_id=1,
        genre_matrix=genre_matrix,
    )

    assert result["content_based"]["movieId"].tolist() == [2]
    assert result["collaborative"].empty
    assert collaborative.calls == []
    assert len(content.calls) == 1
