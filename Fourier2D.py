import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2
from matplotlib.colors import LogNorm

#####Almacena la imagen como un array
img=Image.open("Arboles.png")
arreglo=np.array(img)			

####SOLUCION 1#####
#Lo que hace esta solucion es una transformada de fourier para cada fila en una funcion a la cual le entra por parametro el indice de la fila. Posteriormente, la funcion es llamada dentro de otra con el numero de fila en un iterado, guardando las transformaciones de Fourier de cada fila en un arreglo matricial del tamaño original de la imagen
#POSIBLES ERRORES
# - La matriz no tiene en cuenta la frecuencia de la transformada
# - No se puede aplicar una escala correcta 

#def fourier_fila(arreglo, fila):
#	filas=arreglo.shape[0]
#	cols=arreglo.shape[1]
#	fourier_ifila=fft(arreglo[fila,:])/cols	#Hace la transformada de fourier de una determinada fila (dada como parametro)
#	dx=arreglo[fila][1]-arreglo[fila][0]
#	f_fila=fftfreq(cols,dx)
#	return(fourier_ifila, f_fila)
#	
#def fourier_matriz(arreglo):
#	filas=arreglo.shape[0]
#	cols=arreglo.shape[1]
#	fourier_img=[]
#	for i in range(filas):
#		fourier_img.append(fourier_fila(arreglo, i)[0])
#	return np.array(fourier_img)

#####SOLUCION 2#####--------------------------------------------
#Hace borrosa la imagen al aplicar una transformacion de Fourier bidimesional (fft2) y le elimina el ruido con esto

fourier_img=fft2(arreglo)
fourier_img2=fourier_img.copy()

plt.figure()
plt.imshow(abs(fourier_img), norm=LogNorm(vmin=1))
plt.savefig("LaverdeAndres_FT2D.pdf")

fraccion=0.088
filas=fourier_img2.shape[0]
cols=fourier_img2.shape[1]
fourier_img2[int(filas*fraccion):int(filas*(1-fraccion))]=0 #Hace un filtrado para una parte de la imagen
fourier_img2[:,int(cols*fraccion):int(cols*(1-fraccion))]=0

plt.figure()
plt.imshow(abs(fourier_img2), norm=LogNorm(vmin=1))
plt.savefig("LaverdeAndres_FT2D_filtrada.pdf")

nueva=ifft2(fourier_img2).real
plt.figure()
plt.imshow(nueva, plt.cm.gray)
plt.savefig("LaverdeAndres_Imagen_filtrada.pdf")

#Solucion 2 inspirada en
#https://www.scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html

