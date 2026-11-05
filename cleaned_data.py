from main import main

# Krok 2 - Reviews
reviews_df, books_df = main()
reviews_df = reviews_df[["Id", "Title", "review/score", "review/text"]].dropna()
print("Recenzje po dropna: ", reviews_df.shape)

# Krok 3:
global_mean = reviews_df["review/score"].mean()
trust_level = 20
# normalization + weighting
reviews_df["title_key"] = reviews_df["Title"].astype("string").str.casefold().str.strip()# do stringu -> lower letter-> usuwanie znakow bialych
average_score = reviews_df.groupby("title_key")["review/score"].mean()
average_amount = reviews_df.groupby("title_key")["review/score"].count()


# Dziewczyny: Czy ta meoda będzie w tym miejscu dobra? Jak myslicie?
# WR - Bayesian Weighted Rating
#weighted_score = (average_amount * average_score + trust_level * global_mean) / (average_amount + trust_level)
#print(weighted_score.sort_values(ascending=False).head())


# Czyszczenie tabeli Books
selected_book_columns = ["Title", "authors", "categories", "description"]
existing_book_columns = [column for column in selected_book_columns if column in books_df.columns]

books_df = books_df[existing_book_columns].dropna() # wyrzucenie samych NaNów

for column in ["authors", "categories"]:
    if column in books_df.columns:
        books_df = books_df[~books_df[column].astype(str).str.strip().isin(["[]", "['']", ""])] # wyrzucenie pustych list

books_df = books_df.reset_index(drop=True) # spójne indeksy

print("Książki po czyszczeniu: ", books_df.shape)