import pandas as pd
import pytest

from src.hybrid.evaluator import HybridEvaluator


pytestmark = pytest.mark.unit


def make_section(movie_ids, categories, genres, **scores):
    data = {
        "movieId": movie_ids,
        "title": [f"Movie {movie_id}" for movie_id in movie_ids],
        "genres": genres,
        "category": categories,
    }
    data.update(scores)
    return pd.DataFrame(data)


@pytest.fixture
def valid_sections():
    return {
        "special": make_section(
            [1], ["special"], ["Comedy|Drama"], similarity=[0.9], estimated_rating=[4.5]
        ),
        "content_based": make_section(
            [2], ["content_based"], ["Comedy"], similarity=[0.7]
        ),
        "collaborative": make_section(
            [3], ["collaborative"], ["Drama"], estimated_rating=[4.0]
        ),
    }


def test_evaluator_reports_counts_integrity_diversity_scores_and_overlap(
    valid_sections,
):
    results = HybridEvaluator().evaluate(valid_sections)

    assert results["counts"] == {
        "special": 1,
        "content_based": 1,
        "collaborative": 1,
        "total_rows": 3,
        "total_unique": 3,
    }
    assert results["integrity"]["passed"] is True
    assert results["integrity"]["section_category_mismatches"] == []
    assert results["diversity"] == {
        "unique_movies": 3,
        "unique_genres": 2,
        "genre_distribution": {"Comedy": 2, "Drama": 2},
    }
    assert results["scores"]["average_similarity"] == pytest.approx(0.8)
    assert results["scores"]["average_estimated_rating"] == pytest.approx(4.25)
    assert results["overlap"] == {
        "shared_movies": 0,
        "content_candidates": 1,
        "collaborative_candidates": 1,
        "content_overlap_rate": 0.0,
        "collaborative_overlap_rate": 0.0,
    }


def test_evaluator_flags_duplicate_and_special_overlap_integrity_failures(
    valid_sections,
):
    valid_sections["content_based"] = make_section(
        [1, 4], ["content_based", "content_based"], ["Comedy", "Action"]
    )
    valid_sections["collaborative"] = make_section(
        [5], ["collaborative"], ["Action"]
    )

    integrity = HybridEvaluator().evaluate(valid_sections)["integrity"]

    assert integrity["duplicate_movie_ids"] == 1
    assert integrity["special_overlap_violation"] is True
    assert integrity["passed"] is False


def test_evaluator_flags_recognized_category_in_wrong_section(valid_sections):
    valid_sections["content_based"] = make_section(
        [2], ["collaborative"], ["Comedy"]
    )

    integrity = HybridEvaluator().evaluate(valid_sections)["integrity"]

    assert integrity["section_category_mismatches"] == [
        {
            "section": "content_based",
            "movieId": 2,
            "category": "collaborative",
        }
    ]
    assert integrity["invalid_categories"] == []
    assert integrity["passed"] is False


def test_evaluator_flags_unknown_category(valid_sections):
    valid_sections["special"] = make_section(
        [1], ["unknown"], ["Comedy"]
    )

    integrity = HybridEvaluator().evaluate(valid_sections)["integrity"]

    assert integrity["invalid_categories"] == ["unknown"]
    assert integrity["section_category_mismatches"] == [
        {
            "section": "special",
            "movieId": 1,
            "category": "unknown",
        }
    ]
    assert integrity["passed"] is False


def test_evaluator_accepts_empty_sections_and_returns_empty_score_summaries():
    empty = pd.DataFrame(columns=["movieId", "title", "genres", "category"])
    results = HybridEvaluator().evaluate(
        {"special": empty, "content_based": empty, "collaborative": empty}
    )

    assert results["counts"]["total_rows"] == 0
    assert results["integrity"]["passed"] is True
    assert results["integrity"]["section_category_mismatches"] == []
    assert results["scores"] == {
        "average_similarity": None,
        "average_estimated_rating": None,
    }
    assert results["overlap"]["content_overlap_rate"] == 0.0


@pytest.mark.parametrize(
    "sections, expected_error",
    [
        ({"special": pd.DataFrame()}, "Missing Hybrid sections"),
        (
            {
                "special": pd.DataFrame(),
                "content_based": pd.DataFrame(),
                "collaborative": pd.DataFrame(),
            },
            "missing",
        ),
    ],
)
def test_evaluator_rejects_invalid_section_structure(sections, expected_error):
    with pytest.raises((ValueError, TypeError), match=expected_error):
        HybridEvaluator().evaluate(sections)
