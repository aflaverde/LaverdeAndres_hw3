import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import fft, fftfreq, ifft

img=Image.open("Arboles.png")
array=np.array(img)			#Almacena la imagen como un array

#new_img=Image.fromarray(imarr)
#new_img.save("asdd.png")

def fourier_fila(arreglo, fila):
	filas=arreglo.shape[0]
	cols=arreglo.shape[1]
	for j in range(cols):
		fourier_ifila=fft(arreglo[fila][j])/cols
		dx=arreglo[fila][1]-arreglo[fila][0]
		f_fila=fftfreq(cols,dx)
	plt.plot(f_fila, abs(fourier_ifila))
	plt.show()

fourier_fila(array, 1)
