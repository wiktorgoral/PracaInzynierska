from PIL import Image
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.preprocessing import StandardScaler
import pydotplus
from sklearn.ensemble import RandomForestClassifier
import os

# Problem z importem export_graphviz
os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['CONDA_PREFIX'] + r"\Library\bin\graphviz"



# Klasyfikator drzewa decyzji


# Wczytanie danych
pima = pd.read_csv("diabetes.csv")
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']  
pima.columns = col_names
pima.head()

# Cechy
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]  
y = pima.label  

# Rozdzielenie danych na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70 / 30 trening / test

# Deklaracja modelu
clf = DecisionTreeClassifier()

# Nauka modelu
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Wywietlenie efektów
print("Celnosć drzewa decyzji:", metrics.accuracy_score(y_test, y_pred))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetess.png')
img = Image.open('diabetess.png')
img.show()

# Optymalizacja clasyfikatora
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Wywietlenie efektów
print("Celnosć drzewa decyzji po optymalizcji:", metrics.accuracy_score(y_test, y_pred))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
img = Image.open('diabetes.png')
img.show()




# Klasyfikator losowych drzew


# Deklaracja modelu
regressor = RandomForestClassifier(n_estimators=30, random_state=0)
sc = StandardScaler()

# Nauka modelu
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Wywietlenie efektów
print("Celnosć klasyfikatora losowych drzew: ", end ="")
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

