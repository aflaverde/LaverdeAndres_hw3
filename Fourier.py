import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft
from scipy.interpolate import interp1d


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
print("\n Tamanio de datos de signal.dat",len(x_sig))
print("Tamanio de datos de incompletos.dat", len(x_inc))
print("-->Como la cantidad de datos en incompletos.dat es mucho menor a la del anterior archivo, la derivada de la funcion no es continua y por lo tanto la transformada de Fourier tampoco lo sera para los puntos problematicos de la derivada.")

#####Interpolacion#####
fq=interp1d(x_inc, y_inc, kind='quadratic') #interpolacion cuadratica
fc=interp1d(x_inc, y_inc, kind='cubic') #interpolacion cubica

xnew=np.linspace(min(x_inc), max(x_inc), 512)

func_quad=fq(xnew)
func_cubi=fc(xnew)

dft_orig=DFT(x_inc, y_inc)	#Transformada de Fourier para los datos originales
dft_q=DFT(xnew, func_quad)	#Transformada de Fourier para cada interpolacion
dft_c=DFT(xnew, func_cubi)

n_int=len(xnew)
dx_int=xnew[1]-xnew[0]
f_int=fftfreq(n_int, dx_int) #Frecuencia de los datos interpolados

n_inc=len(x_inc)
dx_inc=x_inc[1]-x_inc[0]
f_inc=fftfreq(n_inc, dx_inc) #Frecuencia de los datos originales

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(f_inc, abs(dft_orig), c='b')
ax2.plot(f_int, abs(dft_q), c='g')
ax3.plot(f_int, abs(dft_c), c='orangered')
ax1.set_title("Transformada de los datos originales")
ax2.set_title("Transformada de la interpolacion cuadratica")
ax3.set_title("Transformada de la interpolacion cubica")
plt.savefig('LaverdeAndres_TF_interpola.pdf')

##---->DIFERENCIAS
print("-->En cuanto a la grafica de la transformada de Fourier de los datos originales se observan frecuencias mucho mayores que los observados en las interpoladas. Los picos de la transformada en los datos originales son mucho mas amplios, mientras que en las funciones interpoladas son locales y definidos para una determinada frecuencia.")

######FILTRADO 1000######
filter1000=1000.0 #Frecuencia a filtrar	
dft_orig1=DFT(x_inc, y_inc)
dft_q1=DFT(xnew, func_quad)
dft_c1=DFT(xnew, func_cubi)	#Copia la transformada para modificarla al filtrar las frecuencias menores a 1000Hz

dft_orig1[abs(f_inc)>filter1000]=0 #Filtro para datos originales en 1000Hz
dft_q1[abs(f_int)>filter1000]=0 #Filtro para datos cuadraticos
dft_c1[abs(f_int)>filter1000]=0 #Filtro para datos cubicos

inversa10=ifft(dft_orig1) 	#Transformada inversa de la funcion original filtrada
inversa_q10=ifft(dft_q1) 	#Transformada inversa de la interp. cuad. filtrada
inversa_c10=ifft(dft_c1) 	#Transformada inversa de la interp. cub. filtrada

######FILTRADO 500######
filter500=500.0
dft_orig2=DFT(x_inc, y_inc)
dft_q2=DFT(xnew, func_quad)
dft_c2=DFT(xnew, func_cubi)	#Copia la transformada para modificarla al filtrar las frecuencias menores a 1000Hz

dft_orig2[abs(f_inc)>filter1000]=0 #Filtro para datos originales en 1000Hz
dft_q2[abs(f_int)>filter1000]=0 #Filtro para datos cuadraticos
dft_c2[abs(f_int)>filter1000]=0 #Filtro para datos cubicos

inversa05=ifft(dft_orig2) 	#Transformada inversa de la funcion original filtrada
inversa_q5=ifft(dft_q2) 	#Transformada inversa de la interp. cuad. filtrada
inversa_c5=ifft(dft_c2) 	#Transformada inversa de la interp. cub. filtrada

#####SUBPLOTS
plt.subplot(321)
plt.plot(x_inc, inversa10, c='b', label="Originales 1000Hz")
plt.title("Filtradas 1000Hz")
plt.subplot(323)
plt.plot(xnew, inversa_q10, c='g', label="Cuadr. 1000Hz")
plt.subplot(325)
plt.plot(xnew, inversa_c10, c='orangered', label="Cubica 1000Hz")
plt.subplot(322)
plt.plot(x_inc, inversa05, c='b', label="Originales")
plt.legend()
plt.title("Filtradas 500Hz")
plt.subplot(324)
plt.plot(xnew, inversa_q5, c='g', label="Cuadr")
plt.legend()
plt.subplot(326)
plt.plot(xnew, inversa_c5, c='orangered', label="Cubica")
plt.legend()
plt.savefig('LaverdeAndres_2Filtros.pdf')


