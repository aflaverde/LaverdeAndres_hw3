import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft

#Almacena los datos de los archivos existentes en el directorio actual
signal=np.genfromtxt("signal.dat", delimiter=',')
incomp=np.genfromtxt("incompletos.dat", delimiter=',')

#Extrae los datos en x y y de cada archivo
x_sig=signal[:,0]
y_sig=signal[:,1]

x_inc=incomp[:,0]
y_inc=incomp[:,1]

######·······signal.dat······#######
#Grafica los datos de signal.dat y lo guarda en LaverdeAndres_signal.pdf
plt.figure()
plt.plot(x_sig, y_sig)
plt.title("Datos originales de signal.dat")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("LaverdeAndres_signal.pdf")

######FOURIER######
def DFT(x,y):
	N=len(x)
	suma_F=np.linspace(0,0,N)
	for n in range(N):
		for k in range(len(suma_F)):
			suma_F[n]+=y[k]*(math.e**((-1j)*2*np.pi*k*n/N))
	return suma_F/N

n_sig=len(x_sig)
dx_sig=x_sig[1]-x_sig[0]
f_sig=fftfreq(n_sig, dx_sig) #Frecuencia de los datos
print("-->Para la grafica de la transformada de Fourier SI se uso el paquete de fftfreq")

FOURIER_DFT=DFT(x_sig,y_sig)
plt.figure()
plt.plot(f_sig, abs(FOURIER_DFT))
plt.title("Transformada de Fourier de los datos")
plt.xlabel("Frecuencia $f$")
plt.ylabel("Transformada de Fourier")
plt.savefig("LaverdeAndres_TF.pdf")

######FRECUENCIAS PRINCIPALES###########
print("-->Las frecuencias principales de la transformada de Fourier son...")


######FILTRADO######
filter1=1000.0 #Frecuencia a filtrar
FOURIER_DFT1=DFT(x_sig,y_sig)	#Copia la transformada para modificarla al filtrar las frecuencias menores a 1000Hz
FOURIER_DFT1[abs(f_sig)>filter1]=0 #Iguala los valores mayores a la frecuencia de corte a 0 para ignorarlos en el filtro

inversa1=ifft(FOURIER_DFT1) 	#Transformada inversa de la funcion filtrada

plt.figure()
plt.plot(x_sig, inversa1)
plt.title("Funcion original filtrada con $f_c=1000$")
plt.xlabel("x")
plt.ylabel("f(x) filtrada")
plt.savefig("LaverdeAndres_filtrada.pdf")


######·······incompletos.dat······#######
print("Tamanio de datos de signal.dat",len(x_sig))
print("Tamanio de datos de incompletos.dat", len(x_inc))
print("Como la cantidad de datos en incompletos.dat es mucho menor a la del anterior archi, la derivada de la funcion no es continua y por lo tanto la transformada de Fourier no lo sera para los puntos problematicos de la derivada.")

#####Interpolacion#####
fq=interp1d(x_inc, y_inc, kind='quadratic') #interpolacion cuadratica
fc=interp1d(x_inc, y_inc, kind='cubic') #interpolacion cubica

xnew=np.linspace(min(x_inc), max(x_inc), 512)

dft_q=DFT(xnew, fq)	#Transformada de Fourier para cada interpolacion
dft_c=DFT(xnew, fc)




