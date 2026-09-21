import unittest

import pandas as pd

from src.data.movie_resolver import MovieTitleResolver


class MovieTitleResolverTests(unittest.TestCase):

    def setUp(self):
        self.movies = pd.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["Jumanji", "Toy Story"],
            }
        )

    def test_resolves_valid_titles(self):
        movie_ids = MovieTitleResolver.resolve_titles(
            movies=self.movies,
            movie_titles=["Jumanji", " toy story "],
        )

        self.assertEqual(movie_ids, [1, 2])

    def test_rejects_unknown_title(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Movie not found: Unknown Movie",
        ):
            MovieTitleResolver.resolve_titles(
                movies=self.movies,
                movie_titles=["Unknown Movie"],
            )


if __name__ == "__main__":
    unittest.main()
