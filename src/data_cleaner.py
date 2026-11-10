import html
import spacy

# Krok 2: Czyszczenie danych
# - Reviews & books


def clean_data(reviews_df, books_df, lemmatize=False, nlp=None):
    # Czyszczenie tabeli Reviews
    reviews_df = reviews_df.copy()
    books_df = books_df.copy()
    reviews_df = reviews_df[["Id", "Title", "review/score", "review/text"]].dropna()
    if reviews_df.empty:
        raise ValueError("Brak dostępnych danych recenzji po oczyszczeniu.")
    reviews_df = preprocess_reviews_text(reviews_df, lemmatize, nlp)

    # Czyszczenie tabeli Books
    selected_book_columns = ["Title", "authors", "categories", "description"]
    existing_book_columns = [
        column for column in selected_book_columns if column in books_df.columns
    ]
    books_df = books_df[existing_book_columns].dropna()  # wyrzucenie samych NaNów
    for column in ["authors", "categories"]:
        if column in books_df.columns:
            books_df = books_df[
                ~books_df[column].astype(str).str.strip().isin(["[]", "['']", ""])
            ]  # wyrzucenie pustych list
    books_df = books_df.reset_index(drop=True)  # spójne indeksy
    if books_df.empty:
        raise ValueError("Brak dostępnych danych książek po oczyszczeniu.")
    return reviews_df, books_df


def preprocess_reviews_text(reviews_df, lemmatize, nlp):
    """Przygotowuje tekst recenzji i opcjonalnie wykonuje lematyzację."""

    reviews_df = clean_review_text(reviews_df)

    if lemmatize:
        if nlp is None:
            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                raise RuntimeError(
                    "Brak modelu spaCy. Wykonaj w terminalu: "
                    "python -m spacy download en_core_web_sm"
                )
        reviews_df = lematize_review_text(reviews_df, nlp)

    return reviews_df


def clean_review_text(review_df):
    """Czyści tekst recenzji ze zbędnych znaków, nadmiarowych spacji, adresów URL i elementów HTML."""

    review_df = review_df.copy()

    if "review/text" not in review_df.columns:
        raise ValueError("Brak kolumny review/text")

    review_df["review/text"] = (
        review_df["review/text"]
        .astype("string")
        .apply(html.unescape)  # zamiana html na zwykłe znaki, np. &amp -> &
        .str.replace(r"<[^>]+>", " ", regex=True)  # usuwanie znaczników html
        .str.replace(
            r"https?://\S+|www\.\S+", " ", regex=True
        )  # usuwanie adresów stron internetowych
        .str.lower()
        .str.replace(
            r"[^\w\s']", " ", regex=True
        )  # usuwanie znaków specjalnych i interpunkcyjnych (zostają litery, cyfry, spacje i apostrofy)
        .str.replace(r"\s+", " ", regex=True)  # zamiana wielu spacji na jedną
        .str.strip()
    )

    review_df = review_df[review_df["review/text"].ne("")]

    return review_df


def lematize_review_text(review_df, nlp):
    """Wykonuje lematyzację słów w recenzjach ze pomocą spaCy."""

    review_df = review_df.copy()

    if "review/text" not in review_df.columns:
        raise ValueError("Brak kolumny review/text")

    # przetwarzanie partiami
    processed_reviews = nlp.pipe(review_df["review/text"], batch_size=2000)

    # zamiana słów na formy podstawowe, np. running -> run
    review_df["review/lemma"] = [
        " ".join(
            token.lemma_
            for token in processed_review
            if not token.is_punct and not token.is_space
        )
        for processed_review in processed_reviews
    ]

    return review_df
