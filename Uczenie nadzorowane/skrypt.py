from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#Klasyfikator K-NN

# Import danych
iris = datasets.load_iris()
# Tabela poprawnych wyników
expected = iris.target
# Deklaracja klasyfikatora 
knn = KNeighborsClassifier(n_neighbors=1)
#Nauka modelu
knn.fit(iris['data'], iris['target'])
prediction = knn.predict(iris.data)

# Wyswietlenie efektów
figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
Y = iris.data[:, 0: 4]
scatter = plt.scatter(Y[:, 2], Y[:, 3], c=prediction)
plt.xlabel('Szerokość płatka')
plt.ylabel('Długość płatka')
plt.show()  

# Celność algorytmu
print(metrics.classification_report(expected, prediction))
print(metrics.confusion_matrix(expected, prediction))




#Klasyfikator Naive-Bayes


iris = datasets.load_iris()
# Deklaracja klasyfikatora 
model = GaussianNB()
#Nauka modelu
model.fit(iris.data, iris.target)

predicted = model.predict(iris.data)

# Wyswietlenie efektów
figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(Y[:, 2], Y[:, 3], c=predicted)
plt.xlabel('Szerokość płatka')
plt.ylabel('Długość płatka')
plt.show()  

# Celność algorytmu
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
