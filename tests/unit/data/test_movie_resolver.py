import pandas as pd
import pytest

from src.data.movie_resolver import MovieTitleResolver


@pytest.fixture
def movies():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": [
                "Jumanji (1995)",
                "Toy Story (1995)",
                "Alien (1979)",
                "Alien (1992)",
            ],
        }
    )


def test_resolves_exact_title(movies):
    assert MovieTitleResolver.resolve_titles(
        movies=movies,
        movie_titles=["Jumanji (1995)"],
    ) == [1]


def test_resolves_title_without_year(movies):
    assert MovieTitleResolver.resolve_titles(
        movies=movies,
        movie_titles=["Jumanji"],
    ) == [1]


def test_resolves_leading_trailing_whitespace(movies):
    assert MovieTitleResolver.resolve_titles(
        movies=movies,
        movie_titles=["  Jumanji (1995)  "],
    ) == [1]


def test_resolves_different_letter_casing(movies):
    assert MovieTitleResolver.resolve_titles(
        movies=movies,
        movie_titles=["jUmAnJi (1995)"],
    ) == [1]


def test_rejects_unknown_title(movies):
    with pytest.raises(
        ValueError,
        match=r"Movie not found: Unknown Movie",
    ):
        MovieTitleResolver.resolve_titles(
            movies=movies,
            movie_titles=["Unknown Movie"],
        )


def test_rejects_ambiguous_title(movies):
    with pytest.raises(
        ValueError,
        match=r"Multiple movies found for title: Alien",
    ):
        MovieTitleResolver.resolve_titles(
            movies=movies,
            movie_titles=["Alien"],
        )
