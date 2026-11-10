from src.data_loader import DataLoader
from src.data_cleaner import clean_data
from src.analysis.reviews_analysis import calculate_weighted_score_reviews


def main():
    try:
        loader = DataLoader()
        reviews_df, books_df = loader.prepare_data()
        reviews_df, books_df = clean_data(reviews_df, books_df)
        top_books = calculate_weighted_score_reviews(reviews_df)

    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        print(f"Wystąpił błąd podczas przetwarzania  danych: {e}")
        return
    else:
        print("Przygotowanie danych zakończono pomyślnie")
        print("Recenzje (po czyszczeniu): ", reviews_df.shape)
        print("Książki (po czyszczeniu): ", books_df.shape)
        print("\nTOP 10 książek według oceny ważonej:")
        print(top_books)


if __name__ == "__main__":
    main()
