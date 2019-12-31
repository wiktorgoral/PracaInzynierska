import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Wczytanie danych
data = pd.read_csv("clustering.csv")  
# Cechy
X = data.iloc[:, 0: 20]
y = data.iloc[:, -1]

# Funkcja oceniająca ważnosć cech
bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Cechy', 'Wartosć']

# Wyswietlenie wartosci cech
print(featureScores.nlargest(10, 'Wartosć'))  



# Klasteryzacja K-means

# Deklaracja modelu z czteroma klastrami
model = KMeans(n_clusters=4) 
# Nauka modelu
model.fit(data)  
all_predictions = model.predict(data)

# Wyswietlenie efektów
figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(X["ApplicantIncome"], X["CoapplicantIncome"], c=all_predictions)
plt.xlabel("Zarobki klienta")
plt.ylabel("Zarobki drugiej osoby")
plt.show() 




# Klasteryzacja hierarchiczna 

# Wczytanie danych
seeds_df = pd.read_csv("seeds-less-rows.csv")
varieties = list(seeds_df.pop('grain_variety'))
samples = seeds_df.values

# Tworzenie modelu
mergings = linkage(samples, method='complete')

# Wyswietlenie efektów
figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k') 
dendrogram(mergings, labels=varieties,leaf_rotation=90,leaf_font_size=6) 
plt.show() 



# Klasteryzacja DBSCAN

# Wczytanie danych
iris = load_iris()

# Deklaracja modelu
dbscan = DBSCAN()

# Nauka modelu
dbscan.fit(iris.data)
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)

# Wyswietlenie efektów
figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
plt.legend([c1, c2, c3], ['Klaster 1', 'Klaster 2', 'Szum'])
plt.show()
