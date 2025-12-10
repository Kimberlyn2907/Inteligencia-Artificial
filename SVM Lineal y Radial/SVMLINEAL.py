# Maquina de vectores de soporte (SVM) para clasificacion
# Se utiliza para clasificar datos en base a un hiperplano que maximiza la separacion entre clases
#SVM maneja datos lineales y no lineales mediante el uso de kernels
#kernels es una funcion que transforma los datos a un espacio de mayor dimension
# donde es posible encontrar un hiperplano de separacion lineal 


# Tratamiento de datos
import pandas as pd
import numpy as np # manejo de arreglos y operaciones matematicas

# Gráficos
import matplotlib.pyplot as plt
from matplotlib import style # sirve para dar estilo a las graficas
import seaborn as sns # sirve para graficar datos estadisticos 
from mlxtend.plotting import plot_decision_regions #sirve para graficar regiones de decision que es
#donde el modelo clasifica los datos

# Preprocesado y modelado
from sklearn.svm import SVC #importa el clasificador SVM
from sklearn.model_selection import train_test_split #Sirve para dividir los datos en 
#entrenamiento y prueba 
from sklearn.model_selection import GridSearchCV # sirve para buscar los mejores hiperparametros
from sklearn.metrics import accuracy_score #sirve para evaluar el modelo 

# Configuración matplotlib
plt.rcParams['image.cmap'] = "bwr" #mapa de colores azul-blanco-rojo
#plt.rcParams['figure.dpi'] = "100" #resolucion de la figura que se guarda 
plt.rcParams['savefig.bbox'] = "tight" #ajusta automaticamente los bordes de la figura al guardarla
style.use('ggplot') or plt.style.use('ggplot') # estilo de graficas ggplot que significa grafico de puntos

# Configuración warnings
import warnings #sirve para manejar advertencias 
warnings.filterwarnings('ignore') #ignora las advertencias para no saturar la salida

# Obtener datos 
datos = pd.read_csv('C:/Users/kimbe/OneDrive/Documentos/IACOPIA/Maquinasdevector/ESL.mixture.csv')
print(datos.head())


#vizualizacion de los datos
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(datos.X1, datos.X2, c=datos.y);
ax.set_title("Datos ESL.mixture");
plt.show()

# División de los datos en train y test
X = datos.drop(columns = 'y') #caracteristicas de los datos
y = datos['y'] #etiquetas o clases de los datos 

X_train, X_test, y_train, y_test = train_test_split( # divide los datos en entrenamiento y prueba
                                        X,
                                        y.values.reshape(-1,1), # reshape para asegurar que y es una matriz columna
                                        train_size   = 0.8, #80% para entrenamiento
                                        random_state = 1234, 
                                        shuffle      = True # desordena los datos antes de dividirlos 
                                        #para evitar sesgos que es cuando los datos estan ordenados de alguna manera 
                                    )
# Creación del modelo SVM lineal

modelo = SVC(C = 100, kernel = 'linear', random_state=123) #C es el parametro de regularizacion 
modelo.fit(X_train, y_train) # Entrena el modelo con los datos de entrenamiento 

# Representación gráfica de los límites de clasificación

# Grid de valores
x = np.linspace(np.min(X_train.X1), np.max(X_train.X1), 50) #np.linspace crea un array de valores equiespaciados
#equiespaciados es que la distancia entre valores es la misma 
#x1 es la primera caracteristica 
#50 es la cantidad de puntos a generar
y = np.linspace(np.min(X_train.X2), np.max(X_train.X2), 50) 
Y, X = np.meshgrid(y, x)#crea una malla de coordenadas a partir de los arrays x e y
# meshgrid: crea matrices de coordenadas a partir de vectores de coordenadas
# X e Y son matrices 2D que contienen las coordenadas de la malla
# cada punto en la malla representa una combinacion de valores de las caracteristicas
grid = np.vstack([X.ravel(), Y.ravel()]).T
# vstack apila arrays en secuencia verticalmente
#ravel aplana las matrices X e Y en vextores 1D
#T transpone el array resultante para que cada fila represente un punto en el espacio de caracteristicas
#en conjunto esta linea de codigo crea un conjunto de puntos en el espacio de caracteristicas

# Predicción valores grid
pred_grid = modelo.predict(grid)# realiza predicciones para cada punto en la malla de coordenadas

fig, ax = plt.subplots(figsize=(6,4))#crea una figura y un conjunto de ejes para graficar
#plt.subplots funcion que crea una figura y un conjunto de ejes para graficar
ax.scatter(grid[:,0], grid[:,1], c=pred_grid, alpha = 0.2) # grafica los puntos del grid con color segun la prediccion
ax.scatter(X_train.X1, X_train.X2, c=y_train, alpha = 1)# se grafica los datos de entrenamiento

# Vectores soporte
ax.scatter( #aqui se grafican los vectores de soporte en donde el modelo toma decisiones para clasificar 
    modelo.support_vectors_[:, 0],
    modelo.support_vectors_[:, 1],#coordenadas de los vectores, el 1 es la segunda caracteristica
    s=200, linewidth=1,#Tamaño y grosor del borde de los puntos
    facecolors='none', edgecolors='black'
)

# Hiperplano de separación

ax.contour(# aqui se grafican los hiperplanos de separacion
    X,
    Y,
    modelo.decision_function(grid).reshape(X.shape),#calcula la funcion de decision del modelo 
    #para cada punto en la malla de coordenadas y lo remodela a la forma de la malla X
    colors = 'k',
    levels = [-1, 0, 1],
    alpha  = 0.5,
    linestyles = ['--', '-', '--']
)

ax.set_title("Resultados clasificación SVM lineal")
plt.show()

# Predicciones test
#realiza predicciones para los datos de prueba en donde el modelo.predict es la funcion que realiza las predicciones
predicciones = modelo.predict(X_test)
predicciones
#

# Accuracy de test del modelo 
#evalua la precision del modelo comparando las prediccciones con las etiquetas reales
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = predicciones,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy}%")
plt.show()