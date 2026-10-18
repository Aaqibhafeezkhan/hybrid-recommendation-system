from typing import Dict

import pandas as pd


class HybridEvaluator:
    """
    Evaluate HybridRecommender output independently.

    This class does not train, load, or call any recommender.
    It only evaluates the Hybrid output sections.
    """

    REQUIRED_CATEGORIES = {
        "special",
        "content_based",
        "collaborative",
    }

    REQUIRED_COLUMNS = {
        "movieId",
        "title",
        "genres",
        "category",
    }

    def evaluate(
        self,
        sections: Dict[str, pd.DataFrame],
    ) -> Dict:

        self._validate_sections(sections)

        combined = pd.concat(
            [
                sections["special"],
                sections["content_based"],
                sections["collaborative"],
            ],
            ignore_index=True,
        )

        return {
            "counts": self._evaluate_counts(
                sections,
                combined,
            ),
            "integrity": self._evaluate_integrity(
                sections,
                combined,
            ),
            "diversity": self._evaluate_diversity(
                combined,
            ),
            "scores": self._evaluate_scores(
                combined,
            ),
            "overlap": self._evaluate_overlap(
                sections,
            ),
        }

    def _validate_sections(
        self,
        sections: Dict[str, pd.DataFrame],
    ) -> None:

        missing_sections = (
            self.REQUIRED_CATEGORIES
            - set(sections.keys())
        )

        if missing_sections:
            raise ValueError(
                "Missing Hybrid sections: "
                f"{sorted(missing_sections)}"
            )

        for category in self.REQUIRED_CATEGORIES:

            data = sections[category]

            if not isinstance(
                data,
                pd.DataFrame,
            ):
                raise TypeError(
                    f"Section '{category}' must be "
                    "a pandas DataFrame."
                )

            missing_columns = (
                self.REQUIRED_COLUMNS
                - set(data.columns)
            )

            if missing_columns:
                raise ValueError(
                    f"Section '{category}' is missing "
                    f"columns: {sorted(missing_columns)}"
                )

    @staticmethod
    def _evaluate_counts(
        sections: Dict[str, pd.DataFrame],
        combined: pd.DataFrame,
    ) -> Dict:

        return {
            "special": len(
                sections["special"]
            ),
            "content_based": len(
                sections["content_based"]
            ),
            "collaborative": len(
                sections["collaborative"]
            ),
            "total_rows": len(combined),
            "total_unique": combined[
                "movieId"
            ].nunique(),
        }

    @staticmethod
    def _evaluate_integrity(
        sections: Dict[str, pd.DataFrame],
        combined: pd.DataFrame,
    ) -> Dict:

        duplicate_movie_ids = int(
            combined["movieId"]
            .duplicated()
            .sum()
        )

        missing_titles = int(
            combined["title"]
            .isna()
            .sum()
        )

        missing_genres = int(
            combined["genres"]
            .isna()
            .sum()
        )

        invalid_categories = sorted(
            set(
                combined["category"]
                .dropna()
                .unique()
            )
            - HybridEvaluator.REQUIRED_CATEGORIES
        )

        section_category_mismatches = []

        for section in (
            "special",
            "content_based",
            "collaborative",
        ):
            section_rows = sections[section]
            mismatched_rows = section_rows[
                section_rows["category"].isna()
                | (section_rows["category"] != section)
            ]

            for row in mismatched_rows[
                ["movieId", "category"]
            ].to_dict("records"):
                actual_category = row["category"]
                section_category_mismatches.append(
                    {
                        "section": section,
                        "movieId": row["movieId"],
                        "category": (
                            None
                            if pd.isna(actual_category)
                            else actual_category
                        ),
                    }
                )

        special_ids = set(
            sections["special"]["movieId"]
        )

        other_ids = set(
            pd.concat(
                [
                    sections["content_based"],
                    sections["collaborative"],
                ],
                ignore_index=True,
            )["movieId"]
        )

        special_overlap_violation = bool(
            special_ids.intersection(
                other_ids
            )
        )

        passed = (
            duplicate_movie_ids == 0
            and missing_titles == 0
            and missing_genres == 0
            and not invalid_categories
            and not section_category_mismatches
            and not special_overlap_violation
        )

        return {
            "duplicate_movie_ids":
                duplicate_movie_ids,
            "missing_titles":
                missing_titles,
            "missing_genres":
                missing_genres,
            "invalid_categories":
                invalid_categories,
            "section_category_mismatches":
                section_category_mismatches,
            "special_overlap_violation":
                special_overlap_violation,
            "passed":
                passed,
        }

    @staticmethod
    def _evaluate_diversity(
        combined: pd.DataFrame,
    ) -> Dict:

        genre_counts = {}

        for genres in (
            combined["genres"]
            .dropna()
            .astype(str)
        ):

            for genre in genres.split("|"):

                if not genre:
                    continue

                genre_counts[genre] = (
                    genre_counts.get(
                        genre,
                        0,
                    )
                    + 1
                )

        return {
            "unique_movies":
                int(
                    combined[
                        "movieId"
                    ].nunique()
                ),
            "unique_genres":
                len(genre_counts),
            "genre_distribution":
                dict(
                    sorted(
                        genre_counts.items(),
                        key=lambda item:
                            item[1],
                        reverse=True,
                    )
                ),
        }

    @staticmethod
    def _evaluate_scores(
        combined: pd.DataFrame,
    ) -> Dict:

        similarity = pd.to_numeric(
            combined.get(
                "similarity",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        )

        estimated_rating = pd.to_numeric(
            combined.get(
                "estimated_rating",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        )

        return {
            "average_similarity":
                (
                    float(
                        similarity.mean()
                    )
                    if similarity.notna().any()
                    else None
                ),
            "average_estimated_rating":
                (
                    float(
                        estimated_rating.mean()
                    )
                    if estimated_rating.notna().any()
                    else None
                ),
        }

    @staticmethod
    def _evaluate_overlap(
        sections: Dict[str, pd.DataFrame],
    ) -> Dict:

        content_ids = set(
            sections["content_based"][
                "movieId"
            ]
        )

        collaborative_ids = set(
            sections["collaborative"][
                "movieId"
            ]
        )

        shared_ids = (
            content_ids
            & collaborative_ids
        )

        return {
            "shared_movies":
                len(shared_ids),
            "content_candidates":
                len(content_ids),
            "collaborative_candidates":
                len(collaborative_ids),
            "content_overlap_rate":
                (
                    len(shared_ids)
                    / len(content_ids)
                    if content_ids
                    else 0.0
                ),
            "collaborative_overlap_rate":
                (
                    len(shared_ids)
                    / len(collaborative_ids)
                    if collaborative_ids
                    else 0.0
                ),
        }
