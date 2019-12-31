import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# Import danych
credits = pd.read_csv("tmdb_5000_credits.csv")
movies_incomplete = pd.read_csv("tmdb_5000_movies.csv")

# Metody klasyczne

# Czyszczenie i sortowanie danych
credits_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_dirty = movies_incomplete.merge(credits_renamed, on='id')
movies_dirty.head()
movies_clean = movies_dirty.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
movies_clean.head()

# Tablice z cechami filmów 
V = movies_clean['vote_count']
R = movies_clean['vote_average']
C = movies_clean['vote_average'].mean()
m = movies_clean['vote_count'].quantile(0.70)

# Dodanie kolumny z wartocią wyliczoną ze wzoru
movies_clean['weighted_average'] = (V / (V + m) * R) + (m / (m + V) * C)
movies_ranked = movies_clean.sort_values('weighted_average', ascending=False)
movies_ranked[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20)
wavg = movies_ranked.sort_values('weighted_average', ascending=False)

# Wyswietlenie efektów
plt.figure(figsize=(16, 6))
ax = sns.barplot(x=wavg['weighted_average'].head(10), y=wavg['original_title'].head(10), data=wavg, palette='deep')
plt.xlim(6.75, 8.35)
plt.title('Najlepsze filmy według głosów', weight='bold')
plt.xlabel('Średnia głosów', weight='bold')
plt.ylabel('Nazwa filmu', weight='bold')
plt.show()

# Najbardziej popularne filmy
popular = movies_ranked.sort_values('popularity', ascending=False)
plt.figure(figsize=(16, 6))
ax = sns.barplot(x=popular['popularity'].head(10), y=popular['original_title'].head(10), data=popular, palette='deep')
plt.title('Najbardziej popularne filmy', weight='bold')
plt.xlabel('Popularnoć', weight='bold')
plt.ylabel('Nazwa filmu', weight='bold')
plt.show()

# Połączenie dwóch poprzednich rankingów
min_max_scaler = preprocessing.MinMaxScaler()
movies_scaled = min_max_scaler.fit_transform(movies_clean[['weighted_average', 'popularity']])
movies_norm = pd.DataFrame(movies_scaled, columns=['weighted_average', 'popularity'])
movies_norm.head()
movies_clean[['norm_weighted_average', 'norm_popularity']] = movies_norm
movies_clean['score'] = movies_clean['norm_weighted_average'] * 0.5 + movies_clean['norm_popularity'] * 0.5
movies_scored = movies_clean.sort_values(['score'], ascending=False)
movies_scored[['original_title', 'norm_weighted_average', 'norm_popularity', 'score']].head(20)
scored = movies_clean.sort_values('score', ascending=False)

# Wyswietlenie efektów
plt.figure(figsize=(16, 6))
ax = sns.barplot(x=scored['score'].head(10), y=scored['original_title'].head(10), data=scored, palette='deep')
plt.title('Najlepsze filmy', weight='bold')
plt.xlabel('Wartosć', weight='bold')
plt.ylabel('Nazwa filmu', weight='bold')
plt.show()



# Filtrowanie na podstawie zawrtosci

# Stworzenie modelu na podstawie częstotliwosci słów
tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words='english')
movies_clean['overview'] = movies_clean['overview'].fillna('')

# Nauka modelu
tfv_matrix = tfv.fit_transform(movies_clean['overview'])
tfv_matrix.shape
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(movies_clean.index, index=movies_clean['original_title']).drop_duplicates()

# Funkcja do sugestii filmów
def give_rec(title, sig=sig):
    idx = indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1: 11]
    movie_indices = [i[0] for i in sig_scores]
    return movies_clean['original_title'].iloc[movie_indices]

nazwa = input("Podaj tytuł filmu:")
print(give_rec(nazwa))
