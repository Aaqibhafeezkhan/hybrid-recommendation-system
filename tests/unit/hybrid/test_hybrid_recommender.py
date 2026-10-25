import pandas as pd
import pytest

from src.hybrid.recommender import HybridRecommender


pytestmark = pytest.mark.unit


class StubContentRecommender:
    def __init__(self, recommendations):
        self.recommendations = recommendations
        self.calls = []

    def recommend(self, movie_id, genre_matrix, top_n):
        self.calls.append(
            {
                "movie_id": movie_id,
                "genre_matrix": genre_matrix,
                "top_n": top_n,
            }
        )
        return self.recommendations.copy()


class StubCollaborativeRecommender:
    def __init__(self, recommendations):
        self.recommendations = recommendations
        self.calls = []

    def recommend(self, user_id, n_recommendations):
        self.calls.append(
            {
                "user_id": user_id,
                "n_recommendations": n_recommendations,
            }
        )
        return self.recommendations.copy()


@pytest.fixture
def hybrid_movies():
    return pd.DataFrame(
        {
            "movieId": list(range(1, 9)),
            "title": [f"Movie {movie_id}" for movie_id in range(1, 9)],
            "genres": ["Comedy|Drama"] * 8,
        }
    )


@pytest.fixture
def hybrid_ratings():
    return pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [1, 2, 6],
            "rating": [4.0, 3.5, 5.0],
        }
    )


@pytest.fixture
def hybrid_recommender(hybrid_movies, hybrid_ratings, genre_matrix):
    content = StubContentRecommender(
        pd.DataFrame(
            {
                "movieId": [3, 4, 5, 6, 7],
                "similarity": [0.95, 0.90, 0.85, 0.80, 0.75],
            }
        )
    )
    collaborative = StubCollaborativeRecommender(
        pd.DataFrame(
            {
                "movieId": [4, 5, 7, 8, 2],
                "estimated_rating": [4.9, 4.8, 4.7, 4.6, 4.5],
            }
        )
    )
    recommender = HybridRecommender(
        content_recommender=content,
        collaborative_recommender=collaborative,
        movies=hybrid_movies,
        ratings=hybrid_ratings,
    )
    return recommender, content, collaborative


def test_hybrid_moves_shared_candidates_to_special_and_removes_seen_movies(
    hybrid_recommender,
    genre_matrix,
):
    recommender, content, collaborative = hybrid_recommender

    sections = recommender.recommend(
        user_id=1,
        seed_movie_id=[1, 3],
        genre_matrix=genre_matrix,
        top_n_per_model=3,
        special_candidate_pool_size=5,
        special_top_n=2,
    )

    assert set(sections) == {"special", "content_based", "collaborative"}
    assert sections["special"]["movieId"].tolist() == [4, 5]
    assert sections["content_based"]["movieId"].tolist() == [3]
    assert sections["collaborative"]["movieId"].tolist() == [7]

    displayed_ids = pd.concat(sections.values())["movieId"]
    assert displayed_ids.is_unique
    assert not {1, 2, 6}.intersection(displayed_ids)
    assert sections["special"]["category"].eq("special").all()
    assert sections["content_based"]["category"].eq("content_based").all()
    assert sections["collaborative"]["category"].eq("collaborative").all()

    assert content.calls[0]["movie_id"] == [1, 3]
    assert content.calls[0]["top_n"] == 5
    assert collaborative.calls[0] == {
        "user_id": 1,
        "n_recommendations": 5,
    }


@pytest.mark.parametrize(
    "parameter",
    [
        "top_n_per_model",
        "special_candidate_pool_size",
        "special_top_n",
    ],
)
def test_hybrid_rejects_non_positive_limits(
    hybrid_recommender,
    genre_matrix,
    parameter,
):
    recommender, _, _ = hybrid_recommender
    kwargs = {
        "user_id": 1,
        "seed_movie_id": 1,
        "genre_matrix": genre_matrix,
        "top_n_per_model": 1,
        "special_candidate_pool_size": 1,
        "special_top_n": 1,
    }
    kwargs[parameter] = 0

    with pytest.raises(ValueError, match="positive integer"):
        recommender.recommend(**kwargs)


def test_hybrid_allows_unknown_user_with_seed_but_rejects_unknown_seed_movie(
    hybrid_recommender,
    genre_matrix,
):
    recommender, content, collaborative = hybrid_recommender

    sections = recommender.recommend(99, 1, genre_matrix)

    assert sections["content_based"]["movieId"].tolist() == [3, 4, 5, 6, 7]
    assert sections["collaborative"].empty
    assert collaborative.calls == []
    assert len(content.calls) == 1

    with pytest.raises(ValueError, match="not found in movies"):
        recommender.recommend(1, 999, genre_matrix)


@pytest.mark.parametrize(
    ("movies_change", "ratings_change", "expected_error"),
    [
        ({"drop": "title"}, None, "movies is missing required columns"),
        ({"duplicate_movie_id": True}, None, "duplicate movieId"),
        (None, {"missing_rating": True}, "ratings contains missing"),
    ],
)
def test_hybrid_validates_input_data_contract(
    hybrid_movies,
    hybrid_ratings,
    movies_change,
    ratings_change,
    expected_error,
):
    movies = hybrid_movies.copy()
    ratings = hybrid_ratings.copy()

    if movies_change and "drop" in movies_change:
        movies = movies.drop(columns=movies_change["drop"])
    if movies_change and movies_change.get("duplicate_movie_id"):
        movies.loc[1, "movieId"] = movies.loc[0, "movieId"]
    if ratings_change and ratings_change.get("missing_rating"):
        ratings.loc[0, "rating"] = pd.NA

    with pytest.raises(ValueError, match=expected_error):
        HybridRecommender(
            StubContentRecommender(pd.DataFrame()),
            StubCollaborativeRecommender(pd.DataFrame()),
            movies,
            ratings,
        )


def test_hybrid_rejects_invalid_source_output_contracts(
    hybrid_movies,
    hybrid_ratings,
    genre_matrix,
):
    content = StubContentRecommender(
        pd.DataFrame({"movieId": [3], "similarity": [1.2]})
    )
    collaborative = StubCollaborativeRecommender(
        pd.DataFrame({"movieId": [4], "estimated_rating": [4.0]})
    )
    recommender = HybridRecommender(
        content,
        collaborative,
        hybrid_movies,
        hybrid_ratings,
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        recommender.recommend(1, 1, genre_matrix, special_candidate_pool_size=1)

    content.recommendations = pd.DataFrame({"movieId": [3, 3], "similarity": [0.8, 0.7]})
    with pytest.raises(ValueError, match="duplicate movieId"):
        recommender.recommend(1, 1, genre_matrix, special_candidate_pool_size=2)
