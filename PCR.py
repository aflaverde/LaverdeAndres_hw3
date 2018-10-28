import numpy as np

arch=np.genfromtxt("WDBC.dat", delimiter=',')
letras=arch[:,1]
arch=np.delete(arch,1,1)	#Elimina la columna con letras
arch=np.delete(arch,0,1)	#Elimina la primera columna del ID

print(arch.shape)

for j in range(arch.shape[1]):
	col_j=arch[:,j]		#Guarda cada columna como variable
	mean_j=np.mean(col_j) #Halla el promedio de cada columna
	col_j-=mean_j		#Resta a cada columna el promedio de esa columna
	#print(sum(col_j))	Para comprobar que esta correctamente hecho se suman todos los elementos de cada columna y deberia dar 0

print(arch)
