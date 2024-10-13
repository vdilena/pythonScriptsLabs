import pandas as pd

# Rango de 20 dias a partir del 01/01/2024
datos = pd.date_range("20240101", periods=20)
print(datos)

# Lectura de archivo csv y se transforma en un dataframe
estadisticasUniv = pd.read_csv("university_enrollment_data.csv", index_col="Student_ID")

# Leo las primeras 5 filas de los datos
print(estadisticasUniv.head())

# Leo las ultimas 20 filas de los datos
print(estadisticasUniv.tail(20))

# Obtengo analitica descriptiva del datframe
print(estadisticasUniv.describe())

# Limpieza de datos (minuto 8.23)