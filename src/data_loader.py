import pandas as pd
import kagglehub
import os


class DataLoader:
    def __init__(self, review_limit=500_000, chunk_size=20_000):
        if review_limit <= 0 or chunk_size <= 0:
            raise ValueError("Limity muszą być większe od zera.")

        self.review_limit = review_limit
        self.chunk_size = chunk_size
        self.output_dir = os.path.join("data")
        self.reviews_output = os.path.join(self.output_dir, "reviews.csv")
        self.books_output = os.path.join(self.output_dir, "books.csv")

    def _download_dataset(self):
        """Pobiera dane z Kaggle i sprawdza obecność wymaganych plików."""

        try:
            path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
        except Exception as e:
            raise RuntimeError("Nie udało się pobrać danych z Kaggle.") from e

        reviews_path = os.path.join(path, "Books_rating.csv")
        books_path = os.path.join(path, "books_data.csv")

        for file_path in (reviews_path, books_path):
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Nie znaleziono pliku: {file_path}")

        return reviews_path, books_path

    def _prepare_reviews(self, reviews_path):
        """Wczytuje ustaloną liczbę recenzji z pliku CSV."""

        try:
            reviews = pd.read_csv(reviews_path, nrows=self.review_limit)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
            raise RuntimeError(
                f"Wystąpił błąd podczas przetwarzania pliku {reviews_path}"
            ) from e
        if "Title" not in reviews.columns:
            raise ValueError(f"Brak kolumny Title w pliku {reviews_path}")

        return reviews

    def _prepare_books(self, books_path, reviews):
        """Wczytuje książki pasujące do tytułów ze zbioru recenzji."""

        titles_from_reviews = (
            reviews["Title"].dropna().astype("string").str.casefold().str.strip()
        )
        titles_from_reviews = set(titles_from_reviews[titles_from_reviews.ne("")])

        books_parts = []
        try:
            for chunk in pd.read_csv(books_path, chunksize=self.chunk_size):
                if "Title" not in chunk.columns:
                    raise ValueError(f"Brak kolumny Title w pliku {books_path}")

                titles = chunk["Title"].astype("string").str.casefold().str.strip()
                mask = titles.isin(titles_from_reviews) & titles.ne("")

                matched = chunk.loc[mask].copy()
                if not matched.empty:
                    books_parts.append(matched)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
            raise RuntimeError(
                f"Wystąpił błąd podczas przetwarzania pliku {books_path}"
            ) from e
        if not books_parts:
            raise ValueError(
                "Nie znaleziono żadnych książek pasujących do wybranych recenzji"
            )

        books = pd.concat(books_parts, ignore_index=True)
        return books

    def prepare_data(self):
        """Pobiera, przetwarza i zapisuje dane książek oraz recenzji do plików CSV, a następnie zwraca je jako obiekty DataFrame."""

        print("Rozpoczęto przygotowywanie danych...")
        reviews_path, books_path = self._download_dataset()

        reviews = self._prepare_reviews(reviews_path)
        books = self._prepare_books(books_path, reviews)

        os.makedirs(self.output_dir, exist_ok=True)

        reviews.to_csv(self.reviews_output, index=False)
        books.to_csv(self.books_output, index=False)

        return reviews, books
