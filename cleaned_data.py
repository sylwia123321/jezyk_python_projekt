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
