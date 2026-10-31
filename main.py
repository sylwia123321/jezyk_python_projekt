from src.data_loader import DataLoader


def main():
    try:
        loader = DataLoader()
        reviews_df, books_df = loader.prepare_data()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Wystąpił błąd poczas ładowania danych: {e}")
    else:
        print("Przygotowanie danych zakończono pomyślnie")
        print("Recenzje: ", reviews_df.shape)
        print("Książki: ", books_df.shape)


if __name__ == "__main__":
    main()
