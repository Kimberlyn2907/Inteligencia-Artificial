from pyexpat import features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

#Obtener datos ( datasert tortugas )
data = pd.read_csv('C:/Users/kimbe/OneDrive/Documentos/IA/KNN/dataset-tortuga.csv')
print(data.columns)
data.info()

#Quitar columnas innecesarias
data = data.drop(columns=['NAME', 'USER_ID'])
features = data.columns.drop('PROFILE')
data = data.drop(columns=[col for col in ['id', 'Unnamed: 32']if col in data.columns])
imputer = SimpleImputer(strategy='mean')
data[features] = pd.DataFrame(imputer.fit_transform(data[features]), columns=features)
data.isna().sum()

#division de datos
#X = data.drop(['PROFILE'], axis = 1)
X = data[features]
y = data['PROFILE']
features = data.columns.drop('PROFILE')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Escalamiento de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Entrenamiento y evaluacion del modelo KNN
acc = {}
for k in range(3, 30, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc[k] = accuracy_score(y_test, y_pred)
    
# PLotting K v/s accuracy graph
#plt.plot(range(3,30,2), acc.values())
plt.plot(list(acc.keys()), list(acc.values()))
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

#Entrenamiento final 
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
