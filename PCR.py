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
	
def covarianza_simetrica(ar,columna):
	suma=0
	for i in range(ar.shape[0]):
		suma+=(ar[i,columna]-np.mean(ar[:,columna]))*(ar[i,columna+1]-np.mean(ar[:,columna+1]))
	varianza=suma/n
	
def covarianza(ar):
	covar=np.zeros((ar.shape[1],ar.shape[1]))
	#inicializa la matriz de covarianza en ceros con una forma m x m donde m es el numero de variables
	
	###Metodo de covarianza###
	for i in range(covar.shape[0]):
		for j in range(covar.shape[0]):
			if i==j:
				covar[i][j]=varianza_diagonal(ar,j)
			else:
				covar[i][j]=covarianza_simetrica(ar,j)
	return covar
	
covar=covarianza(arch)
