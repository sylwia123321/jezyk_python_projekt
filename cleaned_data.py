# Krok 2: Czyszczenie danych
# - Reviews & books

def cleaned_data(reviews_df, books_df):
    # Czyszczenie tabeli Reviews
    reviews_df = reviews_df[["Id", "Title", "review/score", "review/text"]].dropna()
    if reviews_df.empty:
        raise ValueError("Brak dostępnych danych recenzji po oczyszczeniu.")
    # Czyszczenie tabeli Books
    selected_book_columns = ["Title", "authors", "categories", "description"]
    existing_book_columns = [column for column in selected_book_columns if column in books_df.columns]
    books_df = books_df[existing_book_columns].dropna() # wyrzucenie samych NaNów
    for column in ["authors", "categories"]:
        if column in books_df.columns:
            books_df = books_df[~books_df[column].astype(str).str.strip().isin(["[]", "['']", ""])] # wyrzucenie pustych list
    books_df = books_df.reset_index(drop=True) # spójne indeksy
    if books_df.empty:
        raise ValueError("Brak dostępnych danych książek po oczyszczeniu.")
    return reviews_df, books_df

