# =================
# Regresión lineal
# =================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# [ML] Formulación del problema: REGRESIÓN (predecir 'nota' continua a partir de 'horas_estudio')
# 1) Dataset sintético: horas -> nota (relación lineal con ruido)
rng = np.random.RandomState(42)  # [ML] Reproducibilidad de la generación de datos
horas = rng.uniform(0, 10, size=60).reshape(-1, 1)  # [ML] Feature (X): variable explicativa en formato (n_samples, n_features)
nota = 20 + 6 * horas + rng.normal(0, 3, size=(60, 1))  # [ML] Función verdadera lineal + ruido gaussiano (error irreducible)
df = pd.DataFrame({"horas_estudio": horas.ravel(), "nota": nota.ravel()})  # [ML] Dataset tabular para el flujo de ML

# [ML] Separación de variables: X (features) / y (target)
# 2) X / y
X = df[["horas_estudio"]]  # [ML] Matriz de diseño 2D requerida por scikit-learn
y = df["nota"]             # [ML] Variable objetivo continua

# [ML] Validación hold-out: partición Train/Test para estimar generalización y evitar overfitting en evaluación
# 3) Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# [ML] Selección/instanciación del modelo de regresión lineal (OLS: mínimos cuadrados)
# 4) Modelo y ajuste
model = LinearRegression()
model.fit(X_train, y_train)  # [ML] Entrenamiento: estima intercepto y coeficientes minimizando el error cuadrático en train

# [ML] Predicción sobre datos no vistos (test) y evaluación con métricas de regresión
# 5) Predicción y métricas
y_pred = model.predict(X_test)                 # [ML] Inferencia: \hat{y} = \hat{β0} + \hat{β1}*x
mae = mean_absolute_error(y_test, y_pred)      # [ML] MAE: error medio absoluto (interpretación en unidades de 'nota')
r2 = r2_score(y_test, y_pred)                  # [ML] R^2: proporción de varianza explicada (0–1; puede ser <0)

print("=== REGRESIÓN LINEAL ===")
print(f"Modelo: y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * horas_estudio")
# [ML] Interpretabilidad: el coeficiente indica cuánto cambia 'nota' por cada hora adicional de estudio (efecto marginal)
print(f"MAE: {mae:.2f}")
print(f"R^2: {r2:.3f}")

# [ML] Comunicación y diagnóstico visual: dispersión train/test + recta aprendida
# 6) Gráfico: dispersión (train/test) + recta aprendida
plt.figure()
plt.scatter(X_train["horas_estudio"], y_train, label="Train")  # [ML] Ver distribución y posible heterocedasticidad
plt.scatter(X_test["horas_estudio"], y_test, marker="s", label="Test")  # [ML] Distinguir conjunto de prueba

# [ML] Generación de puntos ordenados para trazar la función aprendida en el rango observado
x_line = np.linspace(X["horas_estudio"].min(), X["horas_estudio"].max(), 100).reshape(-1, 1)
y_line = model.predict(pd.DataFrame({"horas_estudio": x_line.ravel()}))  # [ML] Predicción de la recta de regresión

plt.plot(x_line, y_line, label="Recta de regresión")
plt.title("Regresión lineal: horas de estudio vs. nota")
plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.legend()
plt.tight_layout()
plt.show()  # [ML] Cierre del ciclo: comunicación clara del modelo y su ajuste sobre los datos