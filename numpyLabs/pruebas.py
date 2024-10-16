
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# Lista original
numeros = [1, 8, -5, 7, 1, 45, -12, 4, 9, 11, 77, 102, -200, 40, 44]
print("Lista original")
print(numeros)

# Matriz original
matrizNumeros = [[1, 8, -5], [7, 1, 45], [-12, 4, 9], [11, 77, 102], [-200, 40, 44]]
print("Matriz original")
print(matrizNumeros)

# Transformo una lista de una dimension en un array
arrayNumeros = np.array(numeros)
print("Transformo una lista en un array")
print(arrayNumeros)


# Transformo una lista de 2 dimensiones en un array tipo matriz
arrayMatrizNumeros = np.array(matrizNumeros)
print("Transformo una lista de 2 dimensiones en un array tipo matriz")
print(arrayMatrizNumeros)

# Quiero saber las dimensiones de los arrays
print(f"Dimension del primer array: {arrayNumeros.ndim}")
print(f"Dimension del segundo array: {arrayMatrizNumeros.ndim}")

# Quiero saber las estructuras de los arrays
print(f"Estructura del array unidimensional: {arrayNumeros.shape}")
print(f"Estructura del array multidimensional: {arrayMatrizNumeros.shape}")

# Quiero saber los tipos de datos de los arrays
print(f"Tipo de dato de elemento del array unidimensional: {arrayNumeros.dtype}")
print(f"Tipo de dato de elemento del array multidimensional: {arrayMatrizNumeros.dtype}")

# Quiero saber la cantidad de datos de los arrays
print(f"Cantidad de datos del array unidimensional: {arrayNumeros.size}")
print(f"Cantidad de datos del array multidimensional: {arrayMatrizNumeros.size}")

# Quiero crear una matriz inicializada con ceros de 5 x 4
matrizCincoPorCuatro = np.zeros((5,4),dtype=int)
print(f"Matriz creada con ceros de 5 x 4: {matrizCincoPorCuatro}")

# Quiero crear una matriz con una secuencia del 0 al 16
matrizConSecuencia = np.arange(16).reshape(4,4)
print(f"Matriz con secuencia de numeros hasta el 16: {matrizConSecuencia}")

# Quiero crear un array con una secuencia del 5 al 50, de 5 en 5
arrayConSecuencia = np.arange(5, 51, 5)
print(f"Array con secuencia del 5 al 50, de 5 en 5: {arrayConSecuencia}")

# Quiero crear un array con 20 numeros consecutivos distribuidos entre el 1 y el 100
arrayConNumerosConsecutivos = np.linspace(1, 100, 20,dtype=int)
print(f"Array con 20 numeros consecutivos distribuidos entre el 1 y el 100: {arrayConNumerosConsecutivos}")

# Quiero graficar como se distribuye el seno, coseno y tangente de un circulo y ver el grafico para 20 valores consecutivos entre 0 y 2π
circunferencia = np.linspace(0, 2*pi, 20)
seno = np.sin(circunferencia)
coseno = np.cos(circunferencia)
tangente = np.tan(circunferencia)
#plt.plot(seno)
#plt.plot(coseno)
#plt.plot(tangente)
#plt.show()

""" Operaciones con arrays """

# Quiero concatenar arreglos
numerosPares = [ 0, 8, 10, 4, 6, 54, 12]
numerosImpares = [1, 9, 5, 7, 11, 3, 17, 99, 103]
numerosCompleto = np.concatenate((numerosPares, numerosImpares))
print(f"Array de numeros concatenados: {numerosCompleto}")

# Quiero ordenar uno de los arrays anteriores de forma descendente
numerosCompleto.sort()
print(f"Array completo ordenado en forma descendente: {numerosCompleto[::-1]}")

# Quiero generar 500 numeros aleatorios entre 0 y 1, con una semilla de 2 bits para la reproducibilidad
numerosEnRango = np.random.default_rng(2)
numerosAletorios = numerosEnRango.random(500)
print(f"Numeros random entre 0 y 1: {numerosAletorios}")
plt.hist(numerosAletorios, bins = 50)
#plt.show()

# Quiero que me haga un calculo de una normal con los numeros aleatorios con una media de 5 y una desviacion estandar de 2 para 2000 numeros
numerosAletoriosNormal = numerosEnRango.normal(5, 2, 2000)
print(f"Distribucion normal: {numerosAletoriosNormal}")
plt.hist(numerosAletoriosNormal, bins = 50)
#plt.show()

""" Operaciones con funciones agregadas """

# Cargo 200 numeros enteros del 0 al 50
numerosAleatoriosEnteros = numerosEnRango.integers(50, size=200)
print(f"Numeros del 0 al 50 {numerosAleatoriosEnteros}")

# Obtengo el minimo del array numerosAleatoriosEnteros
print(f"Minimo en array: {numerosAleatoriosEnteros.min()}")