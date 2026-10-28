# jezyk_python_projekt

**Wybrany temat projektu:** Analiza recenzji książek oraz ich rekomendacja.

Miejsce pierwszych pomysłów (bałagan twórczy): https://colab.research.google.com/drive/1x4MpsCJqCmkBeYS00JPCn4BTG3FLKiiM?usp=sharing

Projekt będzie obejmować następujące etapy:
1. Wybór datasetu: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews (pliki 'Books_rating.csv' z recenzjami oraz 'books_data.csv' z autorami, gatunkami i linkami – po wczytaniu łączymy je po tytule książki);
2. Uzupełnienie danych istotnych dla recenzji – usunięcie wierszy gdzie jest dużo NaNów, a dla zmiennych liczbowych uśrednienie lub mediana, o ile braków nie jest zbyt dużo (w przeciwnym razie kolumnę pomijamy). Braki w autorach i gatunkach najprawdopodobniej usuwamy;
3. Analiza recenzji: 1) przypisanie sentymentu słowom oraz nauczenie modelu rozumienia tekstu – zamiana recenzji na liczby (np. TF-IDF) i nauczenie klasyfikatora na podstawie ocen 'review/score'; 2) uśrednienie ocen ('ratings') dla każdej książki;
4. Rekomendacja książek na podstawie autorów, gatunku, ocen oraz słów kluczowych i podobieństwa – grupowanie książek za pomocą K-means, a następnie wybór najbliższych książek w klastrze - raczej podobieństwem cosinusowym. Na koniec sprawdzamy, czy rekomendacje mają sens;
5. Wizualizacja wybranych rekomendacji – np. wyświetlenie autorów, tytułu oraz linku do strony książki lub jej podglądu online;
6. **OPCJONALNIE** (jeśli starczy czasu): wizualizacja projektu na stronie internetowej przy użyciu HTML i CSS.
