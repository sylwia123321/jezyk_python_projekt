# Krok 2: Czyszczenie danych
# - Reviews 
# opis bazy danych - dodać

def cleaned_data(reviews_df, books_df):
    reviews_df = reviews_df[["Id", "Title", "review/score", "review/text"]].dropna()
    if reviews_df.empty:
        raise ValueError("Brak dostępnych danych recenzji po oczyszczeniu.")
    return reviews_df, books_df



