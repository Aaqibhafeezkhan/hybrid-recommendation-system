import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd
from surprise import Dataset, Reader, SVD

from src.collaborative.recommender import CollaborativeRecommender
from src.config.settings import HYBRID_REPORTS_DIR
from src.content_based.feature_engineering import FeatureEngineer
from src.content_based.recommender import ContentBasedRecommender
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.hybrid.ranking_evaluator import HybridRankingEvaluator
from src.hybrid.recommender import HybridRecommender


OUTPUT_FILE = HYBRID_REPORTS_DIR / "hybrid_ranking_evaluation_report.txt"
TOP_N_PER_MODEL = 10
SPECIAL_CANDIDATE_POOL_SIZE = 500
SPECIAL_TOP_N = 10


def _build_model(train_ratings: pd.DataFrame, random_state: int):
    reader = Reader(
        rating_scale=(
            float(train_ratings["rating"].min()),
            float(train_ratings["rating"].max()),
        )
    )
    dataset = Dataset.load_from_df(
        train_ratings[["userId", "movieId", "rating"]],
        reader,
    )
    trainset = dataset.build_full_trainset()
    model = SVD(random_state=random_state)
    model.fit(trainset)
    return model, trainset


def _ranked_movie_ids(sections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ordered = pd.concat(
        [
            sections["special"],
            sections["content_based"],
            sections["collaborative"],
        ],
        ignore_index=True,
    )
    if ordered.empty:
        return pd.DataFrame(columns=["movieId"])
    return ordered[["movieId"]].drop_duplicates(ignore_index=True)


def run_evaluation(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> dict:
    evaluator = HybridRankingEvaluator()
    train_ratings, held_out_ratings = evaluator.split_ratings(
        ratings,
        test_fraction=test_fraction,
        random_state=random_state,
    )

    if held_out_ratings.empty:
        raise ValueError(
            "Not enough user rating history to create a held-out evaluation set."
        )

    model, trainset = _build_model(train_ratings, random_state)
    processed_movies = DataPreprocessor().preprocess_movies(movies)
    genre_matrix = FeatureEngineer.build_genre_matrix(processed_movies)
    content_recommender = ContentBasedRecommender()
    collaborative_recommender = CollaborativeRecommender(
        model=model,
        trainset=trainset,
        movies=movies,
    )
    hybrid_recommender = HybridRecommender(
        content_recommender=content_recommender,
        collaborative_recommender=collaborative_recommender,
        movies=movies,
        ratings=train_ratings,
    )

    recommendations_by_user = {}
    for user_id, user_train_ratings in train_ratings.groupby("userId", sort=True):
        seed_movie_id = (
            user_train_ratings.sort_values(
                ["rating", "movieId"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0]["movieId"]
        )
        sections = hybrid_recommender.recommend(
            user_id=int(user_id),
            seed_movie_id=int(seed_movie_id),
            genre_matrix=genre_matrix,
            top_n_per_model=max(k, TOP_N_PER_MODEL),
            special_candidate_pool_size=max(
                SPECIAL_CANDIDATE_POOL_SIZE,
                k,
            ),
            special_top_n=max(k, SPECIAL_TOP_N),
        )
        recommendations_by_user[user_id] = _ranked_movie_ids(sections)

    results = evaluator.evaluate(
        recommendations_by_user=recommendations_by_user,
        held_out_ratings=held_out_ratings,
        k=k,
        relevance_threshold=relevance_threshold,
        user_ids=held_out_ratings["userId"].drop_duplicates().tolist(),
    )
    results.update(
        {
            "train_interactions": len(train_ratings),
            "held_out_interactions": len(held_out_ratings),
            "test_fraction": test_fraction,
            "random_state": random_state,
            "training_users": train_ratings["userId"].nunique(),
            "training_movies": train_ratings["movieId"].nunique(),
        }
    )
    return results


def _build_report(results: dict) -> str:
    lines = [
        "HYBRID HELD-OUT RANKING EVALUATION",
        "=" * 40,
        f"Training interactions: {results['train_interactions']}",
        f"Held-out interactions: {results['held_out_interactions']}",
        f"Training users: {results['training_users']}",
        f"Training movies: {results['training_movies']}",
        f"Split fraction: {results['test_fraction']:.2f}",
        f"Random state: {results['random_state']}",
        f"Relevance rule: held-out rating >= {results['relevance_threshold']}",
        f"Cutoff K: {results['k']}",
        "",
        "Ranking metrics (macro averages)",
        "-" * 40,
        f"Precision@{results['k']}: {results['precision_at_k']:.4f}",
        f"Recall@{results['k']}: {results['recall_at_k']:.4f}",
        f"Users evaluated: {results['evaluated_users']}",
        (
            "Users without relevant held-out movies (excluded from macro "
            f"averages): {results['users_without_relevant_held_out']}"
        ),
        "",
        "Precision@K uses K as the denominator even if fewer than K movies "
        "are returned. Recall@K uses the count of relevant held-out movies. "
        "For each user, the ranked list orders shared recommendations first, "
        "then content-only recommendations, then collaborative-only "
        "recommendations. Held-out interactions are excluded from model "
        "training and seen-movie filtering.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid recommendation ranking against held-out ratings."
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    loader = DataLoader()
    movies = loader.load_movies()
    ratings = loader.load_ratings()
    results = run_evaluation(
        ratings=ratings,
        movies=movies,
        k=args.k,
        relevance_threshold=args.relevance_threshold,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )
    report = _build_report(results)
    HYBRID_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
