import numpy as np
import matplotlib.pyplot as plt

arch=np.genfromtxt("WDBC.dat", delimiter=',')
letras=arch[:,1]
arch=np.delete(arch,1,1)	#Elimina la columna con letras
arch=np.delete(arch,0,1)	#Elimina la primera columna del ID

n=arch.shape[0]

def varianza_diagonal(ar,columna):
	suma=0
	for i in range(ar.shape[0]):
		suma+=(ar[i,columna]-np.mean(ar[:,columna]))**2
	varianza=suma/n
	return varianza
	#Resta a cada elemento el promedio de su columna (al cuadrado) y lo divide por el numero de datos para la varianza de cada variable. Esto lo aniade a la diagonal de la matriz de covarianza.	
	
def covarianza_simetrica(ar, j, contador_columna):
	suma=0
	for i in range(ar.shape[0]):
		suma+=(ar[i,j]-np.mean(ar[:,j]))*(ar[i,contador_columna]-np.mean(ar[:,contador_columna]))
	covarianza=suma/n
	return covarianza
	#Retorna la covarianza de las partes inferior y superior de la diagonal de la matriz
	
def covarianza(ar):
	covar=np.zeros((ar.shape[1],ar.shape[1]))
	#inicializa la matriz de covarianza en ceros con una forma m x m donde m es el numero de variables
	contador_columna=0 #Este contador cambia la posicion del "pivote" cada vez que hace una iteracion
	###Metodo de covarianza###
	for i in range(covar.shape[0]):
		for j in range(ar.shape[1]):
			if i==j:
				covar[i][j]=varianza_diagonal(ar,j)
			elif i!=j:
				covar[i][j]=covarianza_simetrica(ar,j,contador_columna)
			contador_columna+=1
	return covar
	
covar=covarianza(arch)
print(covar)
