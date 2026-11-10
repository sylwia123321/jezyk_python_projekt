# Krok 3: analiza recenzji
def calculate_weighted_score_reviews(reviews_df):
    global_mean = reviews_df["review/score"].mean()
    trust_level = 20
    # normalization + weighting:
    # do stringu -> lower letter-> usuwanie znakow bialych
    reviews_df["title_key"] = reviews_df["Title"].astype("string").str.casefold().str.strip()
    average_score = reviews_df.groupby("title_key")["review/score"].mean() # zwykła średnia globalna
    average_amount = reviews_df.groupby("title_key")["review/score"].count() # liczba recenzji dla danej książki
    # WR - Bayesian Weighted Rating equation
    weighted_score = (average_amount * average_score + trust_level * global_mean) / (average_amount + trust_level)
    return weighted_score.sort_values(ascending=False).head(10)
