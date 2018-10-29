import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2
from matplotlib.colors import LogNorm

img=Image.open("Arboles.png")
arreglo=np.array(img)			#Almacena la imagen como un array

#def fourier_fila(arreglo, fila):
#	filas=arreglo.shape[0]
#	cols=arreglo.shape[1]
#	fourier_ifila=fft(arreglo[fila][:])/cols	#Hace la transformada de fourier de una determinada fila (parametro)
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


fourier_img=fft2(arreglo)
fourier_img2=fourier_img.copy()

plt.figure()
plt.imshow(abs(fourier_img), norm=LogNorm(vmin=1))
plt.savefig("LaverdeAndres_FT2D.pdf")

fraccion=0.089
filas=fourier_img2.shape[0]
cols=fourier_img2.shape[1]
fourier_img2[int(filas*fraccion):int(filas*(1-fraccion))]=0
fourier_img2[:,int(cols*fraccion):int(cols*(1-fraccion))]=0

plt.figure()
plt.imshow(abs(fourier_img2), norm=LogNorm(vmin=1))
plt.show()

nueva=ifft2(fourier_img2).real
plt.figure()
plt.imshow(nueva, plt.cm.gray)
plt.savefig("LaverdeAndres_Imagen_filtrada.pdf")
