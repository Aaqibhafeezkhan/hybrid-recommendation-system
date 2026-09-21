from src.config.settings import (
    CONTENT_RECOMMENDER_RECOMMENDATIONS_DIR,
)

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.movie_resolver import MovieTitleResolver

from src.content_based.feature_engineering import FeatureEngineer
from src.content_based.recommender import ContentBasedRecommender


# ================================
# Test Configuration
# ================================


# ================================
# chose movie name from the dataset 
# and drop in the MOVIE_TITLE variable below
# ================================
MOVIE_TITLE = [
    "Jumanji",
    "Toy Story",
]
TOP_N = 10
# ================================




OUTPUT_FILE = (
    CONTENT_RECOMMENDER_RECOMMENDATIONS_DIR
    / "test_recommendations.csv"
)


def main():

    print("=" * 50)
    print("CONTENT BASED RECOMMENDATION TEST")
    print("=" * 50)


    # -------------------------------
    # Load Dataset
    # -------------------------------

    loader = DataLoader()

    movies = loader.load_movies()


    # -------------------------------
    # Preprocessing
    # -------------------------------

    preprocessor = DataPreprocessor()

    movies = preprocessor.preprocess_movies(
        movies
    )


    # -------------------------------
    # Feature Engineering
    # -------------------------------

    feature_engineer = FeatureEngineer()

    genre_matrix = (
        feature_engineer.build_genre_matrix(
            movies
        )
    )


    # -------------------------------
    # Find Movie
    # -------------------------------

    movie_ids = MovieTitleResolver.resolve_titles(
        movies=genre_matrix,
        movie_titles=MOVIE_TITLE,
    )

    print("\nSelected Movies:")

    for movie_title, movie_id in zip(
        MOVIE_TITLE,
        movie_ids,
    ):
        print(
            f"- {movie_title} "
            f"(ID: {movie_id})"
        )


    # -------------------------------
    # Recommendation
    # -------------------------------

    recommender = ContentBasedRecommender()


    recommendations = recommender.recommend(
        movie_id=movie_ids,
        genre_matrix=genre_matrix,
        top_n=TOP_N,
    )

    # -------------------------------
    # Save Output
    # -------------------------------

    CONTENT_RECOMMENDER_RECOMMENDATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    recommendations.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print("\nTop Recommendations:")
    print("-" * 50)

    print(recommendations)


    print("\nOutput saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()