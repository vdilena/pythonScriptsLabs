# ===================================
# Clasificación logística + Curva ROC
# ===================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# [ML] Formulación del problema: CLASIFICACIÓN BINARIA (aprueba: 0/1)
# 1) Dataset sintético: horas, asistencia -> aprueba (0/1)
rng = np.random.RandomState(42) # [ML] Reproducibilidad de los datos
horas = rng.uniform(0, 10, size=200) # [ML] Feature 1: variable explicativa (X1)
asistencia = rng.uniform(50, 100, size=200) # [ML] Feature 2: variable explicativa (X2)

# [ML] Generación del "logit" subyacente (modelo verdadero) + ruido
# El logit subyacente es la combinación lineal previa a la sigmoide en una clasificación logística. 
# Es decir, es el valor “latente” que el modelo calcula antes de convertirlo en probabilidad.
logit_real = 0.6 * horas + 0.04 * asistencia - 5.0 + rng.normal(0, 0.5, size=200)

# [ML] Etiquetado binario (y): umbral en 0 sobre el logit → aprueba (1) / no aprueba (0)
aprueba = (logit_real >= 0).astype(int)

# [ML] Construcción del dataset tabular para el flujo de ML
df = pd.DataFrame({
    "horas_estudio": horas,
    "asistencia_pct": asistencia,
    "aprueba": aprueba
})

# [ML] Separación de variables: X (features) / y (target)
# 2) X / y
X = df[["horas_estudio", "asistencia_pct"]] # [ML] Matriz de diseño (n_samples, n_features)
y = df["aprueba"] # [ML] Variable objetivo binaria

# [ML] Validación hold-out: partición Train/Test (estratificada para mantener proporciones de clases)
# 3) Train / Test (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# [ML] Selección/instanciación del modelo: Regresión Logística (clasificador lineal con función sigmoide)
# 4) Modelo y ajuste
model = LogisticRegression()
model.fit(X_train, y_train)  # [ML] Entrenamiento: estima coeficientes maximizando la verosimilitud

# [ML] Predicción de probabilidades para evaluar umbrales y construir la curva ROC
# 5) Curva ROC + AUC
# Las curvas ROC y el AUC son herramientas para evaluar clasificadores binarios (p. ej., tu Regresión Logística) 
# independientes del umbral elegido.
# ROC = Receiver Operating Characteristic.
# AUC = Area Under the Curve (ROC).
y_proba = model.predict_proba(X_test)[:, 1] # [ML] p(y=1|x) → necesaria para ROC/AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)  # [ML] Métrica umbral-dependiente: pares (FPR, TPR)
roc_auc = auc(fpr, tpr) # [ML] Métrica umbral-independiente: área bajo la ROC

print("=== CLASIFICACIÓN LOGÍSTICA ===")
print(f"AUC ROC: {roc_auc:.3f}") # [ML] Evaluación de desempeño global del clasificador
# [ML] Interpretabilidad: signo y magnitud de coeficientes en el espacio del logit
print(f"Modelo: sigmoide({model.intercept_[0]:.3f} + " 
      f"{model.coef_[0,0]:.3f}*horas + {model.coef_[0,1]:.3f}*asistencia)")

# [ML] Comunicación de resultados: visualización de la curva ROC y la línea de azar
# 6) Gráfico: ROC
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Azar")
plt.title("Curva ROC — Clasificación logística")
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend()
plt.tight_layout()
plt.show()