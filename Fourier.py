import numpy as np
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
n_sig=len(x_sig)
fourier_sig=fft(y_sig)/n_sig #Funcion normalizada con Fourier
fourier_sig2=fft(y_sig)/n_sig #Otras dos funciones para filtrar las frecuencias requeridas
fourier_sig3=fft(y_sig)/n_sig 

dx_sig=x_sig[1]-x_sig[0]
f_sig=fftfreq(n_sig, dx_sig) #Frecuencia de los datos

#Figura de la transformada de Fourier
plt.figure()
plt.plot(f_sig, abs(fourier_sig))
plt.xlabel("Frecuencia")
plt.ylabel("Transformada de Fourier")
plt.title("Transformada de Fourier de signal.dat")
plt.xlim(-1000,1000)
plt.show()
#plt.savefig("LaverdeAndres_TF.pdf")
print("Para la grafica de la transformada de Fourier SI se uso el paquete de fftfreq")


