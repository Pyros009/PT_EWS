import matplotlib.pyplot as plt
import numpy as np
from obspy import Trace, UTCDateTime, Stream

def fourier_transform(st2 ,st):

    # FFT before filtering
    data = st.data
    n=len(data)
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=st.stats.delta)
    freqs2 = np.fft.fftfreq(len(st2.data), d=st.stats.delta)

    # Calculate the amplitude spectrum
    amplitude = np.abs(np.fft.fft(st2.data))

    # Find the dominant frequency
    dominant_freq_index = np.argmax(amplitude)
    dominant_frequency = freqs2[dominant_freq_index]

    #print(f"The dominant frequency is: {dominant_frequency} Hz")

    #plt.figure(figsize=(12, 6))
    #plt.subplot(2, 1, 1)
    #plt.plot(freqs[:n//2], np.abs(fft_data)[:n//2])  # Magnitude spectrum
    #plt.title('Original FFT Magnitude')
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Magnitude')

    # FFT after filtering
    filtered_fft_data = np.fft.fft(st2.data)
    #plt.subplot(2, 1, 2)
    #plt.plot(freqs2[:n//2], np.abs(filtered_fft_data)[:n//2])  # Magnitude spectrum
    #plt.title('Filtered FFT Magnitude')
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Magnitude')

    #plt.tight_layout()
    #plt.show()
    
    return  dominant_frequency, amplitude