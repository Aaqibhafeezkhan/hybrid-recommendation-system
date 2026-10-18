import pandas as pd

from src.collaborative.recommender import CollaborativeRecommender
from src.content_based.recommender import ContentBasedRecommender


class HybridRecommender:
    """
    Coordinate two independent recommenders without score-level fusion.

    The content-based and collaborative models each produce a candidate pool.
    Movies returned by both models are moved to a separate "special" section.
    The remaining movies preserve their source identity.
    """

    REQUIRED_MOVIE_COLUMNS = {
        "movieId",
        "title",
        "genres",
    }

    REQUIRED_RATING_COLUMNS = {
        "userId",
        "movieId",
        "rating",
    }

    OUTPUT_COLUMNS = [
        "movieId",
        "title",
        "genres",
        "category",
        "content_rank",
        "collaborative_rank",
        "similarity",
        "estimated_rating",
    ]

    SPECIAL_CATEGORY = "special"
    CONTENT_CATEGORY = "content_based"
    COLLABORATIVE_CATEGORY = "collaborative"

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collaborative_recommender: CollaborativeRecommender,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
    ):
        """
        Initialize the hybrid coordinator and validate source datasets.
        """

        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.movies = movies
        self.ratings = ratings

        self._validate_input_data()

    def recommend(
        self,
        user_id: int | None = None,
        seed_movie_id: int | list[int] | None = None,
        genre_matrix: pd.DataFrame | None = None,
        top_n_per_model: int = 10,
        special_candidate_pool_size: int = 500,
        special_top_n: int = 10,
    ) -> dict[str, pd.DataFrame]:
        """
        Return three recommendation sections using whichever valid signal exists.

        Content-based recommendations require seed movie IDs and a genre matrix.
        Collaborative recommendations require a user with rating history.
        Unavailable sources produce empty sections with the standard schema.
        """

        has_content_signal, has_collaborative_signal = self._validate_request(
            user_id=user_id,
            seed_movie_id=seed_movie_id,
            genre_matrix=genre_matrix,
            top_n_per_model=top_n_per_model,
            special_candidate_pool_size=special_candidate_pool_size,
            special_top_n=special_top_n,
        )

        seen_movie_ids = set()
        if has_collaborative_signal:
            seen_movie_ids = set(self._get_seen_movie_ids(user_id).tolist())

        candidate_pool_size = max(
            top_n_per_model,
            special_candidate_pool_size,
        )

        if has_content_signal:
            content_recommendations = self.content_recommender.recommend(
                movie_id=seed_movie_id,
                genre_matrix=genre_matrix,
                top_n=candidate_pool_size,
            )
            content_candidates = self._prepare_content_candidates(
                content_recommendations
            )
        else:
            content_candidates = pd.DataFrame(
                columns=["movieId", "similarity", "content_rank"]
            )

        if has_collaborative_signal:
            collaborative_recommendations = self.collaborative_recommender.recommend(
                user_id=user_id,
                n_recommendations=candidate_pool_size,
            )
            collaborative_candidates = self._prepare_collaborative_candidates(
                collaborative_recommendations
            )
        else:
            collaborative_candidates = pd.DataFrame(
                columns=["movieId", "estimated_rating", "collaborative_rank"]
            )

        content_candidates = content_candidates[
            ~content_candidates["movieId"].isin(seen_movie_ids)
        ].reset_index(drop=True)

        collaborative_candidates = collaborative_candidates[
            ~collaborative_candidates["movieId"].isin(seen_movie_ids)
        ].reset_index(drop=True)

        special_common_movie_ids = set(
            content_candidates["movieId"]
        ).intersection(
            collaborative_candidates["movieId"]
        )

        special = self._build_special_section(
            content_candidates=content_candidates,
            collaborative_candidates=collaborative_candidates,
            common_movie_ids=special_common_movie_ids,
            top_n=special_top_n,
        )

        content_candidates = content_candidates.head(top_n_per_model)
        collaborative_candidates = collaborative_candidates.head(top_n_per_model)

        displayed_common_movie_ids = set(
            content_candidates["movieId"]
        ).intersection(
            collaborative_candidates["movieId"]
        )

        excluded_movie_ids = (
            displayed_common_movie_ids
            | set(special["movieId"])
        )

        content_only = self._build_content_section(
            content_candidates=content_candidates,
            common_movie_ids=excluded_movie_ids,
        )

        collaborative_only = self._build_collaborative_section(
            collaborative_candidates=collaborative_candidates,
            common_movie_ids=excluded_movie_ids,
        )

        return {
            "special": special,
            "content_based": content_only,
            "collaborative": collaborative_only,
        }

    def _build_special_section(
        self,
        content_candidates: pd.DataFrame,
        collaborative_candidates: pd.DataFrame,
        common_movie_ids: set,
        top_n: int,
    ) -> pd.DataFrame:
        if not common_movie_ids:
            return self._empty_result()

        content_common = content_candidates[
            content_candidates["movieId"].isin(common_movie_ids)
        ][
            [
                "movieId",
                "content_rank",
                "similarity",
            ]
        ]

        collaborative_common = collaborative_candidates[
            collaborative_candidates["movieId"].isin(common_movie_ids)
        ][
            [
                "movieId",
                "collaborative_rank",
                "estimated_rating",
            ]
        ]

        special = content_common.merge(
            collaborative_common,
            on="movieId",
            how="inner",
            validate="one_to_one",
        )

        # Rank sum is used only to order the special section.
        # Source scores are not normalized or combined.
        special["_rank_sum"] = (
            special["content_rank"]
            + special["collaborative_rank"]
        )

        special = special.sort_values(
            by=[
                "_rank_sum",
                "content_rank",
                "collaborative_rank",
                "movieId",
            ],
            ascending=True,
            kind="mergesort",
        ).drop(columns="_rank_sum")

        special = special.head(
            top_n
        )

        special["category"] = self.SPECIAL_CATEGORY

        return self._attach_metadata_and_format(special)

    def _build_content_section(
        self,
        content_candidates: pd.DataFrame,
        common_movie_ids: set,
    ) -> pd.DataFrame:
        content_only = content_candidates[
            ~content_candidates["movieId"].isin(common_movie_ids)
        ].copy()

        if content_only.empty:
            return self._empty_result()

        content_only["category"] = self.CONTENT_CATEGORY
        content_only["collaborative_rank"] = pd.NA
        content_only["estimated_rating"] = pd.NA

        content_only = content_only.sort_values(
            by=[
                "content_rank",
                "movieId",
            ],
            ascending=True,
            kind="mergesort",
        )

        return self._attach_metadata_and_format(content_only)

    def _build_collaborative_section(
        self,
        collaborative_candidates: pd.DataFrame,
        common_movie_ids: set,
    ) -> pd.DataFrame:
        collaborative_only = collaborative_candidates[
            ~collaborative_candidates["movieId"].isin(common_movie_ids)
        ].copy()

        if collaborative_only.empty:
            return self._empty_result()

        collaborative_only["category"] = (
            self.COLLABORATIVE_CATEGORY
        )
        collaborative_only["content_rank"] = pd.NA
        collaborative_only["similarity"] = pd.NA

        collaborative_only = collaborative_only.sort_values(
            by=[
                "collaborative_rank",
                "movieId",
            ],
            ascending=True,
            kind="mergesort",
        )

        return self._attach_metadata_and_format(
            collaborative_only
        )

    def _attach_metadata_and_format(
        self,
        recommendations: pd.DataFrame,
    ) -> pd.DataFrame:
        recommendations = recommendations.merge(
            self.movies[
                [
                    "movieId",
                    "title",
                    "genres",
                ]
            ],
            on="movieId",
            how="left",
            validate="many_to_one",
        )

        if recommendations[
            ["title", "genres"]
        ].isna().any().any():
            raise ValueError(
                "Recommendation candidates are missing movie metadata."
            )

        for column in self.OUTPUT_COLUMNS:
            if column not in recommendations.columns:
                recommendations[column] = pd.NA

        return recommendations[
            self.OUTPUT_COLUMNS
        ].reset_index(drop=True)

    def _prepare_content_candidates(
        self,
        recommendations: pd.DataFrame,
    ) -> pd.DataFrame:
        self._validate_recommendation_frame(
            recommendations=recommendations,
            required_columns={
                "movieId",
                "similarity",
            },
            source_name="Content-Based",
        )

        candidates = recommendations[
            [
                "movieId",
                "similarity",
            ]
        ].copy()

        candidates["similarity"] = pd.to_numeric(
            candidates["similarity"],
            errors="raise",
        )

        if candidates["similarity"].isna().any():
            raise ValueError(
                "Content-Based scores contain missing values."
            )

        if (
            (candidates["similarity"] < 0).any()
            or (candidates["similarity"] > 1).any()
        ):
            raise ValueError(
                "Content-Based similarity scores must be "
                "between 0 and 1."
            )

        candidates["content_rank"] = range(
            1,
            len(candidates) + 1,
        )

        return candidates

    def _prepare_collaborative_candidates(
        self,
        recommendations: pd.DataFrame,
    ) -> pd.DataFrame:
        self._validate_recommendation_frame(
            recommendations=recommendations,
            required_columns={
                "movieId",
                "estimated_rating",
            },
            source_name="Collaborative",
        )

        candidates = recommendations[
            [
                "movieId",
                "estimated_rating",
            ]
        ].copy()

        candidates["estimated_rating"] = pd.to_numeric(
            candidates["estimated_rating"],
            errors="raise",
        )

        if candidates["estimated_rating"].isna().any():
            raise ValueError(
                "Collaborative scores contain missing values."
            )

        candidates["collaborative_rank"] = range(
            1,
            len(candidates) + 1,
        )

        return candidates

    def _validate_input_data(self) -> None:
        self._validate_columns(
            dataframe=self.movies,
            required_columns=self.REQUIRED_MOVIE_COLUMNS,
            dataframe_name="movies",
        )

        self._validate_columns(
            dataframe=self.ratings,
            required_columns=self.REQUIRED_RATING_COLUMNS,
            dataframe_name="ratings",
        )

        if self.movies["movieId"].isna().any():
            raise ValueError(
                "movies contains missing movieId values."
            )

        if self.movies["movieId"].duplicated().any():
            raise ValueError(
                "movies contains duplicate movieId values."
            )

        if self.ratings[
            ["userId", "movieId", "rating"]
        ].isna().any().any():
            raise ValueError(
                "ratings contains missing userId, movieId, "
                "or rating values."
            )

    def _validate_request(
        self,
        user_id: int | None,
        seed_movie_id: int | list[int] | None,
        genre_matrix: pd.DataFrame | None,
        top_n_per_model: int,
        special_candidate_pool_size: int,
        special_top_n: int,
    ) -> tuple[bool, bool]:
        if (
            not isinstance(top_n_per_model, int)
            or isinstance(top_n_per_model, bool)
            or top_n_per_model <= 0
        ):
            raise ValueError("top_n_per_model must be a positive integer.")

        if (
            not isinstance(special_candidate_pool_size, int)
            or isinstance(special_candidate_pool_size, bool)
            or special_candidate_pool_size <= 0
        ):
            raise ValueError(
                "special_candidate_pool_size must be a positive integer."
            )

        if (
            not isinstance(special_top_n, int)
            or isinstance(special_top_n, bool)
            or special_top_n <= 0
        ):
            raise ValueError("special_top_n must be a positive integer.")

        if user_id is not None and (
            not isinstance(user_id, int) or isinstance(user_id, bool)
        ):
            raise ValueError("user_id must be an integer or None.")

        has_collaborative_signal = (
            user_id is not None
            and not self._get_seen_movie_ids(user_id).empty
        )
        has_content_signal = seed_movie_id is not None

        if not has_content_signal and not has_collaborative_signal:
            raise ValueError(
                "At least one valid recommendation signal is required: "
                "provide a valid seed movie or a user with rating history."
            )

        if not has_content_signal:
            return False, has_collaborative_signal

        if isinstance(seed_movie_id, int) and not isinstance(seed_movie_id, bool):
            seed_movie_ids = [seed_movie_id]
        elif isinstance(seed_movie_id, list):
            seed_movie_ids = seed_movie_id
        else:
            raise ValueError("seed_movie_id must be an integer, list[int], or None.")

        if not seed_movie_ids:
            raise ValueError("At least one seed movie ID is required.")

        if not all(
            isinstance(movie_id, int) and not isinstance(movie_id, bool)
            for movie_id in seed_movie_ids
        ):
            raise ValueError("All seed movie IDs must be integers.")

        if len(seed_movie_ids) != len(set(seed_movie_ids)):
            raise ValueError("Duplicate seed movie IDs are not allowed.")

        movie_ids = set(self.movies["movieId"])
        missing_movie_ids = set(seed_movie_ids) - movie_ids
        if missing_movie_ids:
            raise ValueError(
                f"Seed movie IDs were not found in movies: "
                f"{sorted(missing_movie_ids)}"
            )

        if not isinstance(genre_matrix, pd.DataFrame):
            raise ValueError(
                "genre_matrix must be a pandas DataFrame when seed movies are provided."
            )

        if "movieId" not in genre_matrix.columns:
            raise ValueError("genre_matrix must contain a movieId column.")

        genre_movie_ids = set(genre_matrix["movieId"])
        missing_genre_ids = set(seed_movie_ids) - genre_movie_ids
        if missing_genre_ids:
            raise ValueError(
                f"Seed movie IDs were not found in genre_matrix: "
                f"{sorted(missing_genre_ids)}"
            )

        return True, has_collaborative_signal

    def _get_seen_movie_ids(
        self,
        user_id: int,
    ) -> pd.Series:
        return self.ratings.loc[
            self.ratings["userId"] == user_id,
            "movieId",
        ]

    @staticmethod
    def _validate_recommendation_frame(
        recommendations: pd.DataFrame,
        required_columns: set,
        source_name: str,
    ) -> None:
        if not isinstance(recommendations, pd.DataFrame):
            raise ValueError(
                f"{source_name} recommender must return "
                "a pandas DataFrame."
            )

        missing_columns = required_columns.difference(
            recommendations.columns
        )

        if missing_columns:
            missing_columns_text = ", ".join(
                sorted(missing_columns)
            )
            raise ValueError(
                f"{source_name} output is missing required "
                f"columns: {missing_columns_text}."
            )

        if recommendations["movieId"].isna().any():
            raise ValueError(
                f"{source_name} output contains missing "
                "movieId values."
            )

        if recommendations["movieId"].duplicated().any():
            raise ValueError(
                f"{source_name} output contains duplicate "
                "movieId values."
            )

    @classmethod
    def _validate_columns(
        cls,
        dataframe: pd.DataFrame,
        required_columns: set,
        dataframe_name: str,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError(
                f"{dataframe_name} must be "
                "a pandas DataFrame."
            )

        missing_columns = required_columns.difference(
            dataframe.columns
        )

        if missing_columns:
            missing_columns_text = ", ".join(
                sorted(missing_columns)
            )
            raise ValueError(
                f"{dataframe_name} is missing required "
                f"columns: {missing_columns_text}."
            )

    @classmethod
    def _empty_result(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls.OUTPUT_COLUMNS)
