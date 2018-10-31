import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2, fftshift, ifftshift
from matplotlib.colors import LogNorm

#####Almacena la imagen como un array
img=plt.imread('Arboles.png')

x=img.shape[0]
y=img.shape[1]

#Transformada de Fourier bidimensional de la imagen
fourier_img=fft2(img, axes=(0,1))

#Cambia la frecuencia 0 al centro del espectro
shift=fftshift(fourier_img)

#######IMAGEN TRANSFORMADA DE FOURIER#######
plt.figure()
plt.title("Transformada de Fourier 2D de la imagen")
plt.imshow(np.log(abs(shift)))
plt.savefig("LaverdeAndres_FT2D.pdf")

#####FILTRADO####
shift2=fftshift(fourier_img) #Copia la transformada para filtrarla

f_cut1=2200	#Frecuencias de corte, rango de frecuencias
f_cut2=25000

#Modifica el arreglo copiado de la transformada de fourier para filtrar las frecuencias de ruido
for i in range(len(shift2)):
	for j in range(len(shift2)):
		if shift2[i,j]<=f_cut2 and shift2[i,j]>=f_cut1:
			shift2[i,j]=0
		else:
			shift2[i,j]=shift2[i,j]
	
#######IMAGEN TRANSFORMADA DE FOURIER - FILTRADA#######	
plt.figure()	
plt.title("Transformada de Fourier filtrada")
plt.imshow(np.log(abs(shift2)))
plt.savefig("LaverdeAndres_FT2D_filtrada.pdf")

###Inversa del shift
inverse_shift=ifftshift(shift2)

###Inversa de la transformada de fourier para la imagen original
img_filtrada=ifft2(inverse_shift)

#######IMAGEN FILTRADA - DENOISED#######	
plt.figure()	
plt.title("Imagen filtrada")
plt.imshow(abs(img_filtrada), plt.cm.gray)
plt.savefig("LaverdeAndres_Imagen_filtrada.pdf")
