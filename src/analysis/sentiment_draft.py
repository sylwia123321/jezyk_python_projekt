#def sentiment_analysis(text):
import matplotlib.pyplot as plt
import re
from IPython.display import display
import data_loader



# BAŁAGAN TWORCZY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


# Do colabU?
train = data_loader()
labels = ['Positive', 'Negative']
pos = train['Sentiment'].sum()
neg = len(train)-pos
sizes=[pos, neg]
colors = ['gold', 'yellowgreen']

# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

wyraz = train['SentimentText'].str.split(expand=True).stack().value_counts()
print("Najczesciej używane wyrazy:")
wyraz.head(10) 

# 3
# przekształcenie to do listy
top_10_words = wyraz.head(10).index.tolist()

for word in top_10_words:
    # nowa kolumna do sprawdzenia - 1 - słowo jest; 0 - nie ma
    train[word] = train['SentimentText'].apply(lambda x: 1 if word in str(x) else 0)

#  wyliczam korelację:
# kolumny do korelacji
cols_to_correlate = top_10_words + ['Sentiment']

# tworzenie bezposrednie
correlation_matrix = train[cols_to_correlate].corr()


correlation_to_sentiment = correlation_matrix['Sentiment'].drop('Sentiment') # wyrzucam korelacje same z sobą

display(correlation_to_sentiment)


# dodac w data_cleaner????


def def_emotki(text):
    # Lista popularnych pozytywnych i negatywnych emotek
    positive_emotions = r'[:;=](?:\s*-?)[)\]DP]|:P|:p'
    negative_emotions = r'[:;=](?:\s*-?)[(\[/{|]t_t'
    other_emotions = r'[:;=](?:\s*-?)[O/\\]|:O|:/|:\|'
    text = re.sub(positive_emotions, 'happy_emotka', text)# Zamiana pozytywnych emotek
    text = re.sub(negative_emotions, 'smutna_emotka', text)  # Zamiana negatywnych emotek
    text = re.sub(other_emotions, 'emotka_neutralna', text) # Zamiana innych emotek
    # Normalizacja białych znaków po podstawieniach
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def shorten_repetitions(text):
    # Zastępuje np. "suuuuper" na "super"
    text = re.sub(r'([a-ząćęłńóśźż])\1{2,}', r'\1', text, flags=re.IGNORECASE)
    return text.strip()