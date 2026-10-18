from typing import Dict, Iterable

import pandas as pd


class HybridRankingEvaluator:
    """Evaluate ranked Hybrid recommendations against held-out ratings."""

    REQUIRED_RATING_COLUMNS = {"userId", "movieId", "rating"}

    @classmethod
    def split_ratings(
        cls,
        ratings: pd.DataFrame,
        test_fraction: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split each user's interactions while retaining training history."""
        cls._validate_ratings(ratings)

        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")

        train_parts = []
        held_out_parts = []

        for _, user_ratings in ratings.groupby("userId", sort=True):
            if len(user_ratings) < 2:
                train_parts.append(user_ratings)
                continue

            held_out_count = min(
                len(user_ratings) - 1,
                max(1, round(len(user_ratings) * test_fraction)),
            )
            held_out = user_ratings.sample(
                n=held_out_count,
                random_state=random_state,
            )
            train = user_ratings.drop(index=held_out.index)

            train_parts.append(train)
            held_out_parts.append(held_out)

        train = pd.concat(train_parts).sort_index().reset_index(drop=True)
        held_out = (
            pd.concat(held_out_parts).sort_index().reset_index(drop=True)
            if held_out_parts
            else ratings.iloc[0:0].copy()
        )

        return train, held_out

    def evaluate(
        self,
        recommendations_by_user: Dict[object, Iterable],
        held_out_ratings: pd.DataFrame,
        k: int = 10,
        relevance_threshold: float = 4.0,
        user_ids: Iterable | None = None,
    ) -> dict:
        """Calculate macro Precision@K and Recall@K for users with relevant holdouts.

        Recommendation order is preserved. Precision@K uses K as its denominator,
        even when fewer than K recommendations are available. Users with no
        relevant held-out movies are excluded from both macro averages and counted
        separately in the report.
        """
        self._validate_ratings(held_out_ratings)

        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer.")

        if not isinstance(relevance_threshold, (int, float)):
            raise ValueError("relevance_threshold must be numeric.")

        relevant_by_user = {
            user_id: set(group.loc[group["rating"] >= relevance_threshold, "movieId"])
            for user_id, group in held_out_ratings.groupby("userId", sort=False)
        }

        users = list(user_ids) if user_ids is not None else list(
            dict.fromkeys(
                list(recommendations_by_user.keys())
                + list(relevant_by_user.keys())
            )
        )

        per_user = {}
        no_relevant_count = 0

        for user_id in users:
            relevant = relevant_by_user.get(user_id, set())
            if not relevant:
                no_relevant_count += 1
                continue

            recommendations = recommendations_by_user.get(user_id, [])
            if isinstance(recommendations, pd.DataFrame):
                if "movieId" not in recommendations.columns:
                    raise ValueError(
                        "Recommendation DataFrames must contain movieId."
                    )
                ranked_ids = recommendations["movieId"].tolist()
            else:
                ranked_ids = list(recommendations)

            top_ids = ranked_ids[:k]
            hits = len(set(top_ids) & relevant)

            per_user[user_id] = {
                "precision_at_k": hits / k,
                "recall_at_k": hits / len(relevant),
                "relevant_held_out": len(relevant),
                "recommended_at_k": len(top_ids),
                "relevant_recommended": hits,
            }

        evaluated_count = len(per_user)
        return {
            "k": k,
            "relevance_threshold": relevance_threshold,
            "precision_at_k": (
                sum(item["precision_at_k"] for item in per_user.values())
                / evaluated_count
                if evaluated_count
                else 0.0
            ),
            "recall_at_k": (
                sum(item["recall_at_k"] for item in per_user.values())
                / evaluated_count
                if evaluated_count
                else 0.0
            ),
            "evaluated_users": evaluated_count,
            "users_without_relevant_held_out": no_relevant_count,
            "per_user": per_user,
        }

    @classmethod
    def _validate_ratings(cls, ratings: pd.DataFrame) -> None:
        if not isinstance(ratings, pd.DataFrame):
            raise TypeError("ratings must be a pandas DataFrame.")

        missing = cls.REQUIRED_RATING_COLUMNS - set(ratings.columns)
        if missing:
            raise ValueError(
                f"ratings is missing required columns: {sorted(missing)}"
            )

        if ratings[list(cls.REQUIRED_RATING_COLUMNS)].isna().any().any():
            raise ValueError("ratings contains missing userId, movieId, or rating values.")
