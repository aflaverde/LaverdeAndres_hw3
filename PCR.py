import numpy as np
import matplotlib.pyplot as plt

arch=np.genfromtxt("WDBC.dat", delimiter=',')
data_tipo=np.genfromtxt("WDBC.dat", delimiter=',', dtype='|U5')
ID=arch[:,0]
tipo=data_tipo[:,1]

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

eig_vals=np.linalg.eig(covar)[0]
eig_vecs=np.linalg.eig(covar)[1]

print("--------------EIGENVALUES Y EIGENVECTORS------------")
for i in range(len(eig_vals)):
	print("Eigenvalue",i,":",eig_vals[i]," Respectivo Eigenvector:", eig_vecs[i])
	
print("-->")

######Proyeccion######
#Discriminar los datos entre maligno y benigno
M=[]
B=[]
for i in range(len(tipo)):
	if tipo[i]=='M':	#Si es maligno crea una lista con los indices de datos que son malignos
		M.append(i)
	else:	#Lista de indices para benignos
		B.append(i)

#Una vez se saben los indices de cada cancer se buscan los datos que corresponden a estos indices en las variables originales
data_M=[]
data_B=[]
for i in range(len(M)):
	data_M.append(arch[M[i]])
for j in range(len(B)):
	data_B.append(arch[B[j]])
	
#Proyectar los datos
PC1_vec=eig_vecs[0]
PC2_vec=eig_vecs[1]	#Estos son los vectores de las 2 componentes principales (vectores con los mayores eigenvalues, vistos en el mensaje de la terminal)

x_malignos=np.matmul(data_M, PC1_vec)	#Proyeccion de los datos en las componentes principales
y_malignos=np.matmul(data_M, PC2_vec)
x_benignos=np.matmul(data_B, PC1_vec)
y_benignos=np.matmul(data_B, PC2_vec)

plt.figure()
plt.scatter(x_malignos, y_malignos, label="Malignos")
plt.scatter(x_benignos, y_benignos, label="Benignos")
plt.legend()
plt.title("PCA para las 2 componentes principales de los datos")
plt.show()
