import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

###########################
# Aprendizaje supervisado #
###########################

# Este dataset de regresion multi-valuada tiene datos de ejercicios fisicos y resultados de parametros fisiologico sobre 20 personas
# El objetivo de su uso es para ver la relacion que hay entre tipos de ejercicios como dominadas, sentadillas y saltos
# y los parametros como el peso, circunferencia de la cintura y la frecuencia cardiaca
# y poder explorar las relaciones multivariadas entre ambas matrices (Predecir los parametros fisiologicos a traves de la performance de los ejercicios)
# En resumen, se busca cuantificar como las performances de los ejercicios impactan en las variables fisiologicas
# Tambien se puede decir que se busca predecir 
from sklearn.datasets import load_linnerud

# Cargo el dataset
datosResultadosClinicosPorEjercicios = load_linnerud(as_frame=True)

# Datos para el eje X correspondientes a los datos de los tipos de ejercicios (variables de entrada).
# Esta relacionado con los features names (serian las columnas de estos datos) que en este caso serian Chins, Situps, Jumps.
X = datosResultadosClinicosPorEjercicios.data

# Datos para el eje Y correspondientes a los datos de los resultados fisiologicos (variables objetivos). En este caso serian Weight, Waist, Pulse.
# Esta relacionado con los target names (serian las columnas de estos datos) que en este caso serian Weight, Waist, Pulse.
Y = datosResultadosClinicosPorEjercicios.target

# Los features son los predictores o datos input
# Los features names son los nombres de estos predictores 
feature_names = datosResultadosClinicosPorEjercicios.feature_names

# Los target son los labels que dependen de las variables de entrada o datos output
# Los target names tienen que ver con etiquetas de clase, usadas para la clasificacion
target_names = datosResultadosClinicosPorEjercicios.target_names

# Muestro algunos datos sueltos del dataset
print("Feature names:", feature_names)
print("Target names:", target_names)
print("\nMuestro las filas de los datos de ejercicios por persona:\n", X[:20])
print("\nMuestro las filas de los datos de resultados fisiologicos por pesona:\n", Y[:20])

##############################################################################
# EDA (Exploratory Data Analysis) para este dataset y visualizacion de datos #
##############################################################################

######################################################################
# Muestro todos los datos que tengo para las 20 personas del dataset #
######################################################################
df = pd.concat([X, Y], axis=1) # Concateno los features y targets para mostrar los datos en una sola tabla
df.index.name = "IDPersona" # Defino la columna que va a ser la PK del identificador de la persona

print(df.shape) # Muestro el tamaño que tiene la tabla (20 filas x 6 columnas)
print(df.head(20)) # Muestro todos los datos de la tabla defininda previamente
print(df.describe(include='number').T) # Muestro los valores estadisticos comunes asociados a la muestra

############################################
# Muestro la distribucion de cada variable #
############################################

# Armo un grafico histograma para ver como se distribuyen sus valores por cada columna y muestro 10 datos con pandas
# Defino que el tamaño del ancho (10) y alto (6) en pulgadas
df.hist(figsize=(10, 6))

# Le agrego un titulo superior a los graficos
plt.suptitle("Distribuciones individuales")

# Ayuda a ajustar los margenes para que no se vea todo apretado cuando no esta maximizada la pantalla
plt.tight_layout()

# Muestro el grafico
plt.show()

####################################
# Correlacion entre datos de X e Y #
####################################

# Correlación dentro de X (ejercicios)

# Creo un grafico de 4 de ancho por 3 de alto
plt.figure(figsize=(4,3))

# Armo un mapa de calor (heatmap) en base a la matriz de correlacion de Pearson (X.corr) con datos numericos entre -1 y 1 para los ejercicios
# Con annot=True escribimos el valor de correlacion dentro de cada cuadro
# La matriz de correlacion de pearson nos sirve para medir que tan lineal es la relacion entre cada variable cruzada, donde +1 significa que cuando
# sube X tambien sube Y, 0 significa que no hay una relacion lineal y -1 significa que cuando X sube Y baja
sns.heatmap(X.corr(numeric_only=True), annot=True, vmin=-1, vmax=1)
plt.title("Correlacion dentro de X (ejercicios)")
plt.show()

# Correlación dentro de Y (fisiología)

# Creo un grafico de 4 de ancho por 3 de alto
plt.figure(figsize=(4,3))

# Armo un mapa de calor (heatmap) en base a la matriz de correlacion de Pearson (X.corr) con datos numericos entre -1 y 1 para los ejercicios
sns.heatmap(Y.corr(numeric_only=True), annot=True, vmin=-1, vmax=1)
plt.title("Correlacion dentro de Y (fisiología)")
plt.show()

# Correlaciones cruzadas X↔Y

# Armo un dataframe con pandas
# results es un diccionario que se arma con la correlacion entre cada elemento de X e Y para la misma persona
datosDF = {}
for nombreColumna in Y.columns:
    correlacion = X.corrwith(Y[nombreColumna])
    datosDF[nombreColumna] = correlacion

correlacionCruzada = pd.DataFrame(datosDF).rename_axis(index="X", columns="Y")

# Creo un grafico de 4 de ancho por 3 de alto
plt.figure(figsize=(5,3))
sns.heatmap(correlacionCruzada, annot=True, vmin=-1, vmax=1)
plt.title("Correlaciones cruzadas X vs Y")
plt.show()

###########################
# Entrenamiento de modelo #
###########################


# Arrays para el modelo
# Transformo los dataframes en arrays de Numpy, para evitar problemas de indexado con schikit-learn
X_np = X.to_numpy()
Y_np = Y.to_numpy()

# Separo los datos con un ratio de 70:30 (70% de entrenamiento y 30% de pruebas)
# random_state=42 sirve para fijar la semilla y que sean mas reproducible los casos
# Esto lo hacemos para saber que tan bien generaliza el modelo
X_train, X_test, y_train, y_test = train_test_split(
    X_np, Y_np, test_size=0.30, random_state=42
)

# Defino el modelo con un pipeline
modelo = make_pipeline(
    # Estandariza las variables predictoras (cada valor queda con media 0 y desvio entandar 1) para que todos los valores esten en la misma escala
    StandardScaler(), 
    # Ridge es para generar una regresion lineal para estabilizar el  sobreajuste
    # alpha=1.0 controla la regularizacion de la regresion
    # MultiOutputRegressor arma un wrapper con la regresion para poder predecir varias salidas al mismo tiempo
    MultiOutputRegressor(Ridge(alpha=1.0))
)

# Entreno el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Armo las predicciones de Y con el modelo entrenado
y_pred = modelo.predict(X_test)


# 4) Métricas (calidad del modelo)
# Métricas por salida (vectorizadas)
r2s  = r2_score(y_test, y_pred, multioutput="raw_values")
maes = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

print("\nMétricas en test por salida (train/test 70/30):")
for nombre, r2, mae in zip(target_names, r2s, maes):
    print(f"  {nombre:>6}: R²={r2:.3f} | MAE={mae:.2f}")

# --- Gráficos de evaluación ---

# Predicho vs Real (por salida)
fig, axes_pr = plt.subplots(1, 3, figsize=(12, 4))
for i, nombre in enumerate(target_names):
    ax = axes_pr[i]
    ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.9)
    # línea diagonal (perfecto)
    lo = min(y_test[:, i].min(), y_pred[:, i].min())
    hi = max(y_test[:, i].max(), y_pred[:, i].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel(f"Real {nombre}")
    ax.set_ylabel(f"Predicho {nombre}")
    ax.set_title(f"Predicho vs Real ({nombre})")
plt.tight_layout()
plt.show()

# Residuos (real - predicho) vs predicho (por salida)
res = y_test - y_pred
fig, axes_res = plt.subplots(1, 3, figsize=(12, 4))
for i, nombre in enumerate(target_names):
    ax = axes_res[i]
    ax.scatter(y_pred[:, i], res[:, i], alpha=0.9)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel(f"Predicho {nombre}")
    ax.set_ylabel("Residuo")
    ax.set_title(f"Residuos vs Predicho ({nombre})")
plt.tight_layout()
plt.show()