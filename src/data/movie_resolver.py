import pandas as pd


class MovieTitleResolver:
    """Resolve movie titles to dataset movie IDs with clear validation errors."""

    @staticmethod
    def resolve_titles(
        movies: pd.DataFrame,
        movie_titles: list[str],
    ) -> list[int]:
        if not isinstance(movies, pd.DataFrame):
            raise ValueError("movies must be a pandas DataFrame.")

        if "movieId" not in movies.columns or "title" not in movies.columns:
            raise ValueError(
                "movies must contain movieId and title columns."
            )

        if not movie_titles:
            raise ValueError("At least one movie title is required.")

        movie_ids = []

        for movie_title in movie_titles:
            if not isinstance(movie_title, str) or not movie_title.strip():
                raise ValueError(
                    f"Movie title must be a non-empty string: {movie_title!r}"
                )

            normalized_title = movie_title.strip().casefold()

            matches = movies[
                movies["title"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
                == normalized_title
            ]

            if matches.empty:
                raise ValueError(f"Movie not found: {movie_title}")

            if len(matches) > 1:
                raise ValueError(
                    f"Multiple movies found for title: {movie_title}"
                )

            movie_ids.append(int(matches["movieId"].iloc[0]))

        return movie_ids
