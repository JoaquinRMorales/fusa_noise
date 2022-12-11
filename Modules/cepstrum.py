import numpy as np
import scipy.io.wavfile as waves
import scipy.fftpack as fourier
import contextlib
import wave

def cepstrum(ruta):
	audio= ruta
	muestreo, sonido = waves.read(audio)
	with contextlib.closing(wave.open(audio,'r')) as f:
    		frames = f.getnframes()
    		rate = f.getframerate()
    		duracion = frames / float(rate)
    
	

	tamano = np.shape(sonido) 
	m = len(tamano)
	canales = 1  
	if (m>1):  
		canales = tamano[1]

	if (canales>1):
    		canal = 0
    		uncanal = sonido[:,canal] 
	else:
    		uncanal = sonido
			
	inicia = 1.5
	termina = duracion
	a = int(inicia*muestreo)
	b = int(termina*muestreo)
	parte = uncanal[a:b]







	mean = np.mean(parte)
	standard_deviation = np.std(parte)
	distance_from_mean = np.abs(parte - mean)
	max_deviations = 0.76
	not_outlier = distance_from_mean < max_deviations * standard_deviation
	no_outliers = parte[not_outlier]



	shift = np.mean(no_outliers)
	parte_shift = no_outliers-shift

	mult = 100000
	parte_mult = mult*parte_shift
	n_fft = 2**(14)
	fft_audio = fourier.fft(parte_mult,n_fft)/muestreo
	

	
	
	m_fft_audio = np.abs(fft_audio)
	eps			= 10**(-3)
	log_m_fft_audio = np.log(m_fft_audio+eps)
	
	
	ifft_audio = fourier.ifft(log_m_fft_audio,n_fft)*muestreo
	ceps = np.abs(ifft_audio)

	return ceps

