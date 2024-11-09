from obspy.signal.trigger import plot_trigger
from obspy.signal.filter import bandpass
from scipy.signal import wiener
from obspy import Trace, UTCDateTime, Stream
import numpy as np
import matplotlib.pyplot as plt

## Fourier filter
def fourier_filter(trace):
    data = trace.data
    n = len(data)
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(n, d=trace.stats.delta)  # Frequency bins

    # Define your frequency limits
    low_cut = 0.0001 # Low cutoff frequency (Hz)
    high_cut = 9.9  # High cutoff frequency (Hz)

    # Create a mask for the filter
    filter_mask = (np.abs(freqs) >= low_cut) & (np.abs(freqs) <= high_cut)

    # Apply the filter
    filtered_fft_data = fft_data * filter_mask
    filtered_data = np.fft.ifft(filtered_fft_data).real
     
    # Apply Wiener filter, adjust 'mysize' based on signal characteristics
    filtered_data = wiener(filtered_data, mysize=4, noise=None)

    trace.data = filtered_data

    return trace

# applies other filters to the data
def other_filters(trace):
    trace.taper(max_percentage=0.05, type="cosine")
    trace.filter("highpass", freq=2, corners=4, zerophase=True)   
    return trace 

# Create a time vector for the sliced data
def time_scale(trace):
    time_vector = np.arange(len(trace)) / trace.stats.sampling_rate
    return time_vector


###########
###########
###########

def filtering_data(trace):
    ### initialize

    sensitivity = trace.meta.response.instrument_sensitivity.value


    ## converting from counts to speed (m/s)
    trace.remove_sensitivity()

    ### Other filters
    trace = fourier_filter(trace)

    other_filters(trace)

    #Detrending
    trace.detrend('linear')
    trace.detrend('demean')
    
    time_vector = time_scale(trace)
    
    return trace, time_vector

