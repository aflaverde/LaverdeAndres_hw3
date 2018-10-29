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

#Grafica los datos de signal.dat y lo guarda en LaverdeAndres_signal.pdf
plt.figure()
plt.plot(x_sig, y_sig)
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
print("Para la grafica de la transformada de Fourier SI se uso el paquete de fftfreq")

FOURIER_DFT=DFT(x_sig,y_sig)
plt.figure()
plt.plot(f_sig, abs(FOURIER_DFT))
plt.show()




