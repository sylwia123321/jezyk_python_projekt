from src.data_loader import DataLoader
from cleaned_data import cleaned_data
from reviews_analysis import calculate_weighted_score_reviews

# Krok 1
def main():
    try:
        loader = DataLoader()
        reviews_df, books_df = loader.prepare_data()
        reviews_df, books_df = cleaned_data(reviews_df, books_df)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Wystąpił błąd poczas ładowania danych: {e}")
    print("Przygotowanie danych zakończono pomyślnie")
    print("Recenzje (po czyszczeniu): ", reviews_df.shape)
    print("Książki (po czyszczeniu): ", books_df.shape)
    print(f"Ważone oceny: {calculate_weighted_score_reviews(reviews_df)}")
    #return reviews_df, books_df


if __name__ == "__main__":
    main()

