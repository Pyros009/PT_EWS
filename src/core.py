import numpy as np
from scipy.signal import find_peaks, convolve, welch, periodogram
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, z_detect
from scipy.fft import fft, fftfreq
from scipy.integrate import simpson
from obspy.taup import TauPyModel
import logging
import logging.handlers

def gaussian_window(size, std):
    """Create a Gaussian window."""
    return np.exp(-0.5 * (np.linspace(-1, 1, size) / std) ** 2)

def get_peakpeak_time_psd(cft, start_index, end_index, sampling_rate): # from https://auspass.edu.au/help/python_script4.html
    data = cft[start_index:end_index]
    f,psd = welch(data,nperseg=64,fs=sampling_rate,nfft=64)
    f = f[2:]; psd=psd[2:] #avoid 0
    maxfreq = f[np.argmax(psd)]
    timespan = 1/(2*maxfreq)
    #plt.plot(f, psd)
    #plt.show()
    return timespan

### MAIN 

def main_processing(trace, gap, dominant_frequency):
                    
    # Get the sampling rate
    sampling_rate = trace.stats.sampling_rate

    # Method for CFT
    method = "recursive_sta_lta"

    if method == "z_detect":
        cft = z_detect(trace.data, int(10 * sampling_rate))
    elif method == "recursive_sta_lta":
        cft = recursive_sta_lta(trace.data, int(5 * sampling_rate), int(10 * sampling_rate))
    else:
        cft = classic_sta_lta(trace.data, int(5 * sampling_rate), int(10 * sampling_rate))

    # CFT setup: ignore start and end portions
    ignore_start_duration = 25
    ignore_end_duration = 25
    ignore_start_samples = int(ignore_start_duration * sampling_rate)
    ignore_end_samples = int(ignore_end_duration * sampling_rate)
    total_length = len(cft)
    start_index = ignore_start_samples
    end_index = total_length - ignore_end_samples

    # Slice the CFT and raw data
    cft_sliced = cft[start_index:end_index]
    time_vector = np.arange(len(cft_sliced)) / sampling_rate
    raw_data_sliced = trace.data[start_index:end_index]
    time_vector_raw = time_vector[:len(raw_data_sliced)]

    # Smooth the CFT using a Gaussian filter
    window_size = int(0.5 * sampling_rate)
    std_dev = window_size / 10
    gauss_window = gaussian_window(window_size, std_dev)
    smoothed_cft = convolve(cft_sliced, gauss_window / np.sum(gauss_window), mode='same')

    # Adaptive thresholding based on percentiles
    low_percentile = 90
    high_percentile = 99
    low_threshold = np.percentile(smoothed_cft, low_percentile)
    high_threshold = np.percentile(smoothed_cft, high_percentile)

    # Define the time of the seismic event (in seconds or as an index)
    event_time = (gap-ignore_start_duration)  # Replace with the actual event time in seconds

    # Convert event time to index
    event_index = int(event_time * sampling_rate)-1  

    # Peak detection (dual-pass)
    # First pass: Detect high-prominence peaks
    high_prominence_peaks, _ = find_peaks(
        smoothed_cft,
        height=high_threshold,
        distance=int(1 * sampling_rate),
        prominence=0.5
    )

    # Second pass: Detect lower prominence peaks near the high-prominence peaks
    additional_peaks, _ = find_peaks(
        smoothed_cft,
        height=low_threshold,
        distance=int(0.5 * sampling_rate),
        prominence=0.2
    )

    # Combine the peaks without duplicates
    combined_peaks = np.unique(np.concatenate((high_prominence_peaks, additional_peaks)))

    # Filter out peaks that occur before the event
    filtered_peaks = [peak for peak in combined_peaks if peak >= event_index]

    # Identify valleys between the peaks
    valleys = []
    valley_positions = []
    for i in range(len(filtered_peaks) - 1):
        peak_start = filtered_peaks[i]
        peak_end = filtered_peaks[i + 1]
        region = smoothed_cft[peak_start:peak_end]
        valley = np.min(region)
        valley_index = peak_start + np.argmin(region)
        valleys.append(valley)
        valley_positions.append(valley_index)

    # Calculate amplitudes with adaptive window sizing
    max_amplitudes = []
    max_positive_peaks = []
    max_negative_peaks = []
    windows = []
    window_duration = 10
    window_samples = int(window_duration * sampling_rate)
    std_threshold = np.std(smoothed_cft[:int(5 * sampling_rate)]) * 1e-2  # Noise-adjusted flatness threshold
    details = {}
    for i, peak in enumerate(filtered_peaks):
        # Determine valley bounds around each peak
        if i == 0:
            valley_coord_before = 0
        else:
            valley_coord_before = valley_positions[i - 1]

        if i == len(filtered_peaks) - 1:
            valley_coord_next = len(time_vector)
        else:
            valley_coord_next = valley_positions[i]

        start_window_index = max(valley_coord_before, peak - window_samples // 2)
        end_window_index = min(valley_coord_next, peak + window_samples // 2)
        start_window_index = max(0, min(len(raw_data_sliced) - 1, start_window_index))
        end_window_index = max(0, min(len(raw_data_sliced), end_window_index))
        windowed_raw_data = raw_data_sliced[start_window_index:end_window_index]

        if windowed_raw_data.size > 0:
            max_amplitude = np.max(windowed_raw_data)
            min_amplitude = np.min(windowed_raw_data)
        else:
            max_amplitude = 0
            min_amplitude = 0

        max_amplitudes.append(max_amplitude - min_amplitude)
        max_positive_peaks.append(max_amplitude if max_amplitude > 0 else 0)
        max_negative_peaks.append(abs(min_amplitude) if min_amplitude < 0 else 0)
        windows.append((time_vector_raw[start_window_index], time_vector_raw[end_window_index]))
        
        peak2peak = get_peakpeak_time_psd(raw_data_sliced, start_window_index, end_window_index, sampling_rate)

        #print(f"Peak2Peak for peak {i+1}: {peak2peak}")
        details[i+1]= {"peak2peak": peak2peak}
    # Plotting
    #fig, axs = plt.subplots(2, 1, figsize=(30, 10), sharex=True)

    # Plot smoothed CFT and peaks/valleys
    #axs[0].plot(time_vector, smoothed_cft, label='Smoothed CFT', color='green')
    #axs[0].scatter(time_vector[filtered_peaks], smoothed_cft[filtered_peaks], color='red', label='Detected Peaks', zorder=5)
    #axs[0].scatter(time_vector[valley_positions], smoothed_cft[valley_positions], color='pink', label='Detected Valleys', zorder=5)

    # Plot raw data with amplitude markers
    #axs[1].plot(time_vector_raw, raw_data_sliced, label='Raw Data', color='blue')
    
    for i, peak in enumerate(filtered_peaks):
        peak_time = time_vector_raw[peak]
        max_pos = max_positive_peaks[i]
        max_neg = -max_negative_peaks[i]

        # Vertical line for each detected peak
#        axs[1].axvline(x=peak_time, color='red', linestyle='--', label='Detected Peak' if i == 0 else "")
        
        # Draw horizontal lines for max positive and negative amplitudes
        start_time = time_vector_raw[start_window_index]
        end_time = time_vector_raw[end_window_index]
#        axs[1].hlines(max_pos, start_time, end_time, color='purple', linestyle='--', label='Max Positive Amplitude' if i == 0 else "")
#        axs[1].hlines(max_neg, start_time, end_time, color='orange', linestyle='--', label='Max Negative Amplitude' if i == 0 else "")
        
        # Print the absolute amplitude for each peak-window
        absolute_amplitude = max(max_pos, abs(max_neg))
        
        ## to be tested
        # Assuming raw_data_sliced is your velocity data
        max_velocity_amplitude = absolute_amplitude  # Get maximum absolute velocity amplitude

        # Calculate displacement amplitude
        displacement_amplitude = max_velocity_amplitude / (2 * np.pi * dominant_frequency)
        
        # Print the results
        #print(f"Maximum Velocity Amplitude for {i+1}: {max_velocity_amplitude} m/s")
        
        #print(f"Calculated Displacement Amplitude for {i+1}: {displacement_amplitude} m")
        
        details[i+1]["Vel Amp (m/s)"] = max_velocity_amplitude
        details[i+1]["Disp Amp (m)"] = displacement_amplitude

    # Add vertical lines for detected valleys
    #for valley_index in valley_positions:
        #axs[1].axvline(x=time_vector[valley_index], color='pink', linestyle='--', label='Detected Valley' if valley_index == valley_positions[0] else "")

    # Configure legend and grid
    #axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    #axs[0].grid()
    #axs[1].set_title("Z-component")
    #axs[1].set_xlabel("Time (s)")
    #axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    #axs[1].grid()

    #plt.tight_layout(rect=[0, 0, 0.85, 1])
    #plt.show()
    

        
    # Find the maximum amplitude
    max_amplitude = np.max(np.abs(trace.data))
    #print(f"Maximum Amplitude: {max_amplitude}")

    # Perform FFT to find frequencies
    N = len(trace.data)
    yf = fft(trace.data)
    xf = fftfreq(N, 1 / sampling_rate)

    # Find the peak frequency
    peak_freq = np.abs(xf[np.argmax(np.abs(yf))])

    # Assume a wave speed (for example, P-wave speed in crust)
    wave_speed = 6000  # in meters per second

    # Calculate wavelength
    wavelength = wave_speed / peak_freq

    #print(f"Peak Frequency: {peak_freq:.2f} Hz")
    #print(f"Wavelength: {wavelength:.2f} meters")

    # Calculate RMS
    rms_amplitude = np.sqrt(np.mean(trace.data**2))

    # Estimate energy (empirical relationship)
    energy_estimate = rms_amplitude**2  # Simplified relationship

    #print(f"RMS Amplitude: {rms_amplitude}")
    #print(f"Estimated Energy (relative): {energy_estimate}")
    
    details["RMS"] = rms_amplitude
    details["Energy"] = energy_estimate
    details["peak_freq"] = peak_freq
    details["wavelength"] = wavelength
    
    time_ref = 3 #secs
    tau_c_rep = []  # to store tau_c values

    for i in range(len(filtered_peaks)):
        # Define the time window based on filtered_peaks
        t_window_start = int(filtered_peaks[i])  # start at the filtered peak
        t_window_end = int(filtered_peaks[i] + time_ref * sampling_rate)  # 3-10 seconds duration
        logging.info("t_windows",t_window_start, t_window_end)
        # Create time vector for this window (assuming sampling_rate is in Hz)
        t_window = np.linspace(t_window_start / sampling_rate, t_window_end / sampling_rate, t_window_end - t_window_start)

        # Extract the velocity data for the current time window
        if t_window_end <= len(raw_data_sliced):  # Ensure window doesn't go out of bounds
            v_t_window = raw_data_sliced[t_window_start:t_window_end]
        else:
            v_t_window = raw_data_sliced[t_window_start:len(raw_data_sliced)]  # Avoid undefined velocity window
            logging.warning(f"Invalid velocity window for peak {i+1}.")
            
        if v_t_window is not None and len(v_t_window) > 0:    
            # Calculate r using the extracted velocity data
            numerator = simpson(v_t_window**2, x=t_window)  # Integral of velocity squared
            denominator = simpson(v_t_window, x=t_window)   # Integral of velocity

            # Calculate r (should take the absolute value of denominator to avoid negative values)
            r = numerator / abs(denominator)

            # Calculate tau_c (characteristic time constant)
            tau_c = 2 * np.pi / np.sqrt(r)
            tau_c_rep.append(tau_c)

            # Output the results for r and tau_c
            #print(f"r {i+1}: {r}")
            #print(f"tau_c {i+1}: {tau_c} seconds")
                
            # First Integration: Velocity → Displacement using Simpson's Rule
            # Use cumulative sum to integrate over time, giving a time-series of displacement.
            displacement_t = np.cumsum(v_t_window) * (t_window[1] - t_window[0])

            # Second Integration: Displacement → Moment History using Simpson's Rule
            # Integrate the displacement time-series to get the moment history
            moment_history_t_simpson = simpson(displacement_t, x=t_window)

            # Output the results for moment history
            #print(f"Moment History (using Simpson's Rule): {moment_history_t_simpson}")
                
            details[i+1]["r"] = r
            details[i+1]["moment_history"] = moment_history_t_simpson
            details[i+1]["tau_c"]=tau_c
            
            # Calculate maximum displacement amplitude
            max_velocity_amplitude = np.max(np.abs(v_t_window))  # Max velocity amplitude
            displacement_amplitude = max_velocity_amplitude / (2 * np.pi * dominant_frequency)  # Displacement amplitude
            details[i + 1]["Vel Amp (m/s)"] = max_velocity_amplitude
            details[i + 1]["Disp Amp (m)"] = displacement_amplitude
                  
            # Assuming you already have velocity data and time vector
            # Example for displacement calculation (from velocity) using cumulative sum
            displacement_t = np.cumsum(v_t_window) * (t_window[1] - t_window[0])  # Displacement after integrating velocity

            # Find the first peak in the displacement signal after P-wave arrival
            # Let's assume filtered_peaks[i] is the P-wave arrival time index
            # We look for peaks in the displacement after P-wave arrival (i.e., after filtered_peaks[i])
            peak_start = int(filtered_peaks[i] + 1)  # Skip the initial P-wave
            #peak_start = np.arange(len(peak_start)) / sampling_rate
            displacement_peaks, _ = find_peaks(displacement_t)

                # Check if any peaks were found in displacement_t
            if len(displacement_peaks) > 0:
                    # First peak in the displacement_t slice (at index 0 in the normalized slice)
                    first_peak_index = displacement_peaks[0]
                    first_peak_displacement = displacement_t[first_peak_index]
                    #print(f"First Peak Displacement (normalized): {first_peak_displacement} meters")
                    
                    # Now, let's calculate the scalar moment M0 using the first peak's displacement:
                    mu = 30e9  # Rigidity (in Pa, for most tectonic plates)
                    A = 10**7    # Fault area (example, in m^2, assuming 1 km²)
                    
                    # Calculate the scalar moment M0
                    M0 = mu * A * first_peak_displacement  # Scalar moment in Newton-meters (Nm)
                    #print(f"Scalar Moment (M0): {M0} Nm")
                    details["peak_disp"] = first_peak_displacement
                    details["M0"] = M0
                      # Calculate RMS and Energy estimate
                    rms_amplitude = np.sqrt(np.mean(trace.data**2))
                    energy_estimate = rms_amplitude**2  # Simplified relationship
                    details["RMS"] = rms_amplitude
                    details["Energy"] = energy_estimate
                    
                    if not details:
                        logging.warning("Details dictionary is empty, returning None.")
                        return None

                    return details 
            else:
                    #print("No significant peak found in the displacement slice.")
                    return details