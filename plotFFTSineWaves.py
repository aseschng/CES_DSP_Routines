""" 
Title: Testing Pythonista
and learning about fftshift and freq
Author: Chng Eng Siong
Date: 24 Nov 2018

"""
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def postProcessPhase(inMag, inPhase, threshold):
	mask = inMag > threshold*max(inMag)
	processPhase = inPhase*mask
	return processPhase


def doubleSidedFFTDisplay(inMag, inPhase, Fs):
	outMag = np.fft.fftshift(inMag)
	outPhase = np.fft.fftshift(inPhase)
	N=len(inMag)
	xf = np.zeros(N)
	# making my own frequency bin for even and off length fft
	# need Fs and also ensuring the end points are ok
	if (N%2 == 0):  # even N
		xf[0:int(N/2)]  = np.arange(0,int(N/2))*Fs/N
		xf[int(N/2):N]  = np.arange(int(-N/2),0)*Fs/N
	else:
		xf[0:int((N-1)/2)+1]	    = np.arange(0,int((N+1)/2))*Fs/N
		xf[int((N-1)/2)+1:N]     = np.arange(int((-N+1)/2),0)*Fs/N
		
	out_xfreq = np.fft.fftshift(xf)
	return out_xfreq, outMag, outPhase



# This is an example to plot a sine wave
def plot_sineWav():
		Fs= 16000
		N = 256
		t = np.arange(0,N)*(1/Fs)
		F1 = 2000
		F2 = 6000
		y = 4*np.sin(2*np.pi*F1*t)+10*np.cos(2*np.pi*F2*t-np.pi/3)
		fig,ax = plt.subplots()
		ax.plot(t,y,'g-+')
		plt.show()
		
		
		# Lets get the DFT
		# good example from https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
		#and timer operation, https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
		
		startT = timer()
		Y     = np.fft.fft(y)
		endT  = timer()
		print('Elapsed time of FFT =', endT-startT)
		
		mag_Y = np.abs(Y)/N
		angle_Y = np.angle(Y)
		angle_Y_processed = postProcessPhase(mag_Y,angle_Y,0.1)
		xf2 = np.fft.fftfreq(mag_Y.shape[-1],1.0/Fs)
		
		fig,ax = plt.subplots(2,1)
		ax[0].plot(xf2, mag_Y)
		ax[1].plot(xf2, angle_Y_processed)
		plt.title('Fig1')
		plt.show()


		outFreq, outMag, outPhase = doubleSidedFFTDisplay(mag_Y, angle_Y_processed, Fs)
		fig,ax = plt.subplots(2,1)
		ax[0].plot(outFreq,outMag)
		ax[1].plot(outFreq, outPhase)
		plt.title('Fig2')
		plt.show()
		
				
if __name__ == '__main__':
	plot_sineWav()

