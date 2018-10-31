import numpy as np
import matplotlib.pyplot as plt

arch=np.genfromtxt("WDBC.dat", delimiter=',')
ID=arch[:,0]
tipo=arch[:,1]
arch=np.delete(arch,1,1)	#Elimina la columna con letras
arch=np.delete(arch,0,1)	#Elimina la primera columna del ID

def covarianza(data):
	var=data.shape[1]
	n=float(data.shape[0])
	covar=np.zeros((var,var)) #inicializa la matriz de covarianza en ceros con una forma m x m donde m es el numero de variables 
	for i in range(var):
		for j in range(var):
			mean_var1=np.mean(data[:,i])
			mean_var2=np.mean(data[:,j])
			covar[i,j]=np.sum((data[:,i]-mean_var1)*(data[:,j]-mean_var2))/n
			#Resta a cada elemento el promedio de su columna multiplicando por lo mismo o por la siguiente variable sea el caso, suma todo lo anterior y luego lo divide por el numero de datos para la varianza de cada variable.
	return covar
			
covar=covarianza(arch)

eig_val=np.linalg.eig(covar)[0]
eig_vec=np.linalg.eig(covar)[1]

print("--------------EIGENVALUES Y EIGENVECTORS------------")
for i in range(len(eig_val)):
	print("Eigenvalue",i,":",eig_val[i]," Respectivo Eigenvector:", eig_vec[i])
	
print("-->")

######Proyeccion
