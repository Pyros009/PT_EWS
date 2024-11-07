from obspy import Trace, UTCDateTime, Stream
import matplotlib.pyplot as plt
from multiprocessing import Queue
import seaborn as sns
import numpy as np
import logging
import logging.handlers
import time
from collections import deque
import pandas as pd        
from filtering import filtering_data
from fourier import fourier_transform
from scipy.signal import find_peaks, convolve, welch
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, z_detect
from scipy.fft import fft, fftfreq
from scipy.integrate import simpson
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configure logging (if not already configured in your main module)
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
)

def gaussian_window(size, std):
    """Create a Gaussian window."""
    return np.exp(-0.5 * (np.linspace(-1, 1, size) / std) ** 2)

def get_peakpeak_time_psd(cft, start_index, end_index, sampling_rate):
    data = cft[start_index:end_index]
    f,psd = welch(data,nperseg=64,fs=sampling_rate,nfft=64)
    f = f[2:]; psd=psd[2:] #avoid 0
    maxfreq = f[np.argmax(psd)]
    timespan = 1/(2*maxfreq)
    #plt.plot(f, psd)
    #plt.show()
    return timespan

def main_processing(trace, gap, dominant_frequency):
                    
    # Get the sampling rate
    sampling_rate = trace.stats.sampling_rate
    logging.error(f"sampling_rate {sampling_rate}.")
    # Method for CFT
    method = "recursive_sta_lta"

    if method == "z_detect":
        cft = z_detect(trace.data, int(10 * sampling_rate))
    elif method == "recursive_sta_lta":
        cft = recursive_sta_lta(trace.data, int(5 * sampling_rate), int(10 * sampling_rate))
    else:
        cft = classic_sta_lta(trace.data, int(5 * sampling_rate), int(10 * sampling_rate))

    # CFT setup: ignore start and end portions
    end_index = len(cft)
    start_index=0


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
    logging.error("Ran Gauss nd smoothed.")
    # Adaptive thresholding based on percentiles
    low_percentile = 90
    high_percentile = 99
    low_threshold = np.percentile(smoothed_cft, low_percentile)
    high_threshold = np.percentile(smoothed_cft, high_percentile)

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

    # Identify valleys between the peaks
    valleys = []
    valley_positions = []
    for i in range(len(combined_peaks) - 1):
        peak_start = combined_peaks[i]
        peak_end = combined_peaks[i + 1]
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
    for i, peak in enumerate(combined_peaks):
        # Determine valley bounds around each peak
        if i == 0:
            valley_coord_before = 0
        else:
            valley_coord_before = valley_positions[i - 1]

        if i == len(combined_peaks) - 1:
            valley_coord_next = len(time_vector)
        else:
            valley_coord_next = valley_positions[i]

        start_window_index = max(valley_coord_before, peak - window_samples // 2)
        end_window_index = min(valley_coord_next, peak + window_samples // 2)
        start_window_index = max(0, min(len(raw_data_sliced) - 1, start_window_index))
        end_window_index = max(0, min(len(raw_data_sliced), end_window_index))-1
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
        
        if start_window_index < len(time_vector_raw) and end_window_index < len(time_vector_raw):
            windows.append((time_vector_raw[start_window_index], time_vector_raw[end_window_index]))
        else:
            # Handle the error case (maybe skip this window or log a warning)
            logging.error(f"Indices {start_window_index}, {end_window_index} are out of bounds.")
    
        peak2peak = get_peakpeak_time_psd(raw_data_sliced, start_window_index, end_window_index, sampling_rate)

        #print(f"Peak2Peak for peak {i+1}: {peak2peak}")
        details[i+1]= {"peak2peak": peak2peak}
    # Plotting
    #fig, axs = plt.subplots(2, 1, figsize=(30, 10), sharex=True)

    # Plot smoothed CFT and peaks/valleys
    #axs[0].plot(time_vector, smoothed_cft, label='Smoothed CFT', color='green')
    #axs[0].scatter(time_vector[combined_peaks], smoothed_cft[combined_peaks], color='red', label='Detected Peaks', zorder=5)
    #axs[0].scatter(time_vector[valley_positions], smoothed_cft[valley_positions], color='pink', label='Detected Valleys', zorder=5)

    # Plot raw data with amplitude markers
    #axs[1].plot(time_vector_raw, raw_data_sliced, label='Raw Data', color='blue')
    
    for i, peak in enumerate(combined_peaks):
        peak_time = time_vector_raw[peak]
        max_pos = max_positive_peaks[i]
        max_neg = -max_negative_peaks[i]
    
        # Print the absolute amplitude for each peak-window
        absolute_amplitude = max(max_pos, abs(max_neg))
        
        # Assuming raw_data_sliced is your velocity data
        max_velocity_amplitude = absolute_amplitude  # Get maximum absolute velocity amplitude

        # Calculate displacement amplitude
        displacement_amplitude = max_velocity_amplitude / (2 * np.pi * dominant_frequency)
        
        details[i+1]["Vel Amp (m/s)"] = max_velocity_amplitude
        details[i+1]["Disp Amp (m)"] = displacement_amplitude

    # Find the maximum amplitude
    max_amplitude = np.max(np.abs(trace.data))

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

    for i in range(len(combined_peaks)):
        # Define the time window based on combined_peaks
        t_window_start = int(combined_peaks[i])  # start at the filtered peak
        t_window_end = min(len(raw_data_sliced), int(combined_peaks[i] + time_ref * sampling_rate))  # 3-10 seconds duration
        
        # Create time vector for this window (assuming sampling_rate is in Hz)
        t_window = np.linspace(t_window_start / sampling_rate, t_window_end / sampling_rate, t_window_end - t_window_start)

        # Extract the velocity data for the current time window
        if t_window_end <= len(raw_data_sliced):  # Ensure window doesn't go out of bounds
            v_t_window = raw_data_sliced[t_window_start:t_window_end]
        else:
            v_t_window = None  # Avoid undefined velocity window
            logging.error(f"Indices for v_t_window {t_window_start}, {t_window_end} are out of bounds.")
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
        
        else:
            details[i + 1] = {
                "r": None,
                "moment_history": None,
                "tau_c": None,
                "Vel Amp (m/s)": None,
                "Disp Amp (m)": None
            }
            # Optionally log or print a message if the velocity window is invalid
            logging.warning(f"Invalid velocity window for peak {i + 1}.")
            
            
            # Assuming you already have velocity data and time vector
            # Example for displacement calculation (from velocity) using cumulative sum
        displacement_t = np.cumsum(v_t_window) * (t_window[1] - t_window[0])  # Displacement after integrating velocity

            # Find the first peak in the displacement signal after P-wave arrival
            # Let's assume combined_peaks[i] is the P-wave arrival time index
            # We look for peaks in the displacement after P-wave arrival (i.e., after combined_peaks[i])
        peak_start = int(combined_peaks[i] + 1)  # Skip the initial P-wave
        displacement_peaks, _ = find_peaks(displacement_t)
                # Check if any peaks were found in displacement_t
        if len(displacement_peaks) > 0:
                # First peak in the displacement_t slice (at index 0 in the normalized slice)
                first_peak_index = displacement_peaks[0]
                first_peak_displacement = displacement_t[first_peak_index]
                    
                    # Now, let's calculate the scalar moment M0 using the first peak's displacement:
                mu = 30e9  # Rigidity (in Pa, for most tectonic plates)
                A = 10**7    # Fault area (example, in m^2, assuming 1 km²)
                    
                    # Calculate the scalar moment M0
                M0 = mu * A * first_peak_displacement  # Scalar moment in Newton-meters (Nm)
                    #print(f"Scalar Moment (M0): {M0} Nm")
                details["peak_disp"] = first_peak_displacement
                details["M0"] = M0
                   
                return details 
        else:
                logging.info("No peak found")
                return details
                
def warn_check(st, time_window):
    logging.info(f"Starting warn_check with {len(st)} traces")
    ## last 4mins (240s) -> data_buffer
    # Merge all traces in the data_buffer into a single Stream object
                   
    st2 = st.copy()
                    
    event_stat = {}
                    
    # Process components
    st_z = st.select(component="Z")
    st_n = st.select(component="N")
    st_e = st.select(component="E")

    logging.info(f"st_z: {st_z}, st_n: {st_n}, st_e: {st_e}")

    st2_z = st2.select(component="Z")
    st2_n = st2.select(component="N")
    st2_e = st2.select(component="E")
    
    logging.info(f"st2_z: {st2_z}, st2_n: {st2_n}, st2_e: {st2_e}")
    
    filtered_z, time_vector_z = filtering_data(st2_z[0])
    filtered_n, time_vector_n = filtering_data(st2_n[0])
    filtered_e, time_vector_e = filtering_data(st2_e[0])
    logging.info(f"Filtered_z: {filtered_z}")
    # Process Z component
    if len(st_z) > 0:
        dom_freq_z, amp_z = fourier_transform(filtered_z, st_z[0])
        event_stat["Domin_freq_z"] = dom_freq_z
        logging.info(f"dom_freq_z: {dom_freq_z}")
    else:
        print("No Z component traces available.")

    # Process N component
    if len(st_n) > 0:
        dom_freq_n, amp_n = fourier_transform(filtered_n, st_n[0])
        event_stat["Domin_freq_n"] = dom_freq_n
    else:
        print("No N component traces available.")

    # Process E component
    if len(st_e) > 0:
        dom_freq_e, amp_e = fourier_transform(filtered_e, st_e[0])
        event_stat["Domin_freq_e"] = dom_freq_e
    else:
        print("No E component traces available.")
                        
    details = main_processing(filtered_z, time_window, dom_freq_z)
    # Check if details is None, and handle accordingly
    if details is None:
        logging.error("No details returned from main_processing.")
        return
    
    for key, dicts in details.items():
        if key == 1:
            for stat, values in dicts.items():
                if stat == "Vel Amp (m/s)":
                    event_stat["P-Vel amp (m/s)"] = values
                elif stat == "Disp Amp (m)":
                    event_stat["P-Disp amp (m)"] = values
                elif stat == "peak2peak":
                    event_stat["P-peak2peak"] = values
                elif stat == "r":
                    event_stat["P-r"] = values
                elif stat == "moment_history":
                    event_stat["P-moment_history"] = values
                elif stat == "tau_c":
                    event_stat["P-tau_c"] = values    
                else:
                    print(f"{key} error")
        elif key == 2:
            for stat, values in dicts.items():
                if stat == "Vel Amp (m/s)":
                    event_stat["S-Vel amp (m/s)"] = values
                elif stat == "Disp Amp (m)":
                    event_stat["S-Disp amp (m)"] = values
                elif stat == "peak2peak":
                    event_stat["S-peak2peak"] = values
                elif stat == "r":
                    event_stat["S-r"] = values
                elif stat == "moment_history":
                    event_stat["S-moment_history"] = values
                elif stat == "tau_c":
                    event_stat["S-tau_c"] = values    
                else:
                    print(f"{key} error")
        elif key == "RMS":
            event_stat["RMS"] = dicts    
        elif key == "Energy":
            event_stat["Energy"] = dicts 
        elif key == "peak_freq":
            event_stat["peak_freq"] = dicts 
        elif key == "wavelength":
            event_stat["wavelength"] = dicts 
        elif key == "peak_disp":
            event_stat["peak_disp"] = dicts 
        elif key == "M0":
            event_stat["M0"] = dicts 
        else:
            event_stat["other infos"] = dicts
                    
        event_data = pd.DataFrame()
        event_data = pd.concat([event_data, pd.Series(event_stat, name=key).to_frame().T], ignore_index=True, axis=0)
        event_data.reset_index(drop=True, inplace=True)
        event_data["Domin_freq_n"] = abs(event_data["Domin_freq_n"])
        event_data["Domin_freq_e"] = abs(event_data["Domin_freq_e"])
        event_data["Domin_freq_z"] = abs(event_data["Domin_freq_z"])
        event_data["P-Vel amp (m/s)"] = abs(event_data["P-Vel amp (m/s)"])
        if "S-Vel amp (m/s)" in event_data.columns:
            event_data["S-Vel amp (m/s)"] = abs(event_data["S-Vel amp (m/s)"])
        else:
            logging.info("Column 'S-Vel amp (m/s)' not found!")
        event_data["P-Disp amp (m)"] = abs(event_data["P-Disp amp (m)"])
        if "S-Disp amp (m)" in event_data.columns:
            event_data["S-Disp amp (m)"] = abs(event_data["S-Disp amp (m)"])
        else:
            logging.info("Column 'S-Disp amp (m)' not found!")            
            
        required_cols = ['Domin_freq_z','P-peak2peak', 'P-Disp amp (m)', 'P-r', 'P-moment_history', 'P-tau_c','RMS','Energy']
        #error avoidance
        missing_columns = [col for col in required_cols if col not in event_data.columns]
        for col in missing_columns:
            event_data[col] = np.nan
            
        features = event_data [['Domin_freq_z','P-peak2peak', 'P-Disp amp (m)', 'P-r', 'P-moment_history', 'P-tau_c','RMS','Energy']]
        



        # normalize
        normalizer = MinMaxScaler()
        normalizer.fit(features)
        features_norm = normalizer.transform(features)
        features_norm = pd.DataFrame(features_norm, columns=features.columns)
                    
        return features_norm

                        

def process_trace(queue):
    time_window = 300  # Time window in seconds (4 minutes)
    data_buffer = deque()  # Create a deque to hold traces (ordered by time)
    current_time = 0  # Track the time of the last trace (in seconds)
    
    try:
        while True:
            # Check if there is a new trace from the pipe
            
            if not queue.empty():  # Check if there's data available in the pipe
                logging.info("P3- Processing initiated...")
                trace_record = queue.get()  # Receive the trace from P3
                
                logging.info(f"P3 - Received new trace: {trace_record}, initiating processing...")                              
                             
                st = trace_record.copy()
                
                new_trace_starttime = st[0].stats.starttime        
                 # Add the new trace to the data buffer with the associated timestamp
                data_buffer.append(st)
                
                while len(data_buffer) > 1:
                    # Pop the oldest trace from the buffer (out of the time window)
                    oldest_trace_starttime = data_buffer[0][0].stats.starttime
                    # Check if starttime is a UTCDateTime or a Unix timestamp (float)
                    if isinstance(new_trace_starttime, float):
                        new_trace_starttime = time.gmtime(new_trace_starttime)
                    if isinstance(oldest_trace_starttime, float):
                        oldest_trace_starttime = time.gmtime(oldest_trace_starttime)
                    # If they are `UTCDateTime` objects, subtract and check the difference
                    if isinstance(new_trace_starttime, float) or isinstance(oldest_trace_starttime, float):
                        # If float (Unix timestamp), calculate the difference in seconds directly
                        time_difference = new_trace_starttime - oldest_trace_starttime
                    else:
                        # Subtract `UTCDateTime` objects and get the difference in seconds
                        time_difference = new_trace_starttime - oldest_trace_starttime
                    
                    # Compare time difference with time window (in seconds)
                    if time_difference > time_window:
                        # Remove the oldest trace from the buffer if it exceeds the time window
                        data_buffer.popleft()
                    else:
                        break  # Stop if the time window is valid
                                 
                if len(data_buffer) >= 3:
                    #logging.info(f"Data_buffer contains {len(data_buffer)} traces: {data_buffer}")
                    st_merged = Stream()
                    for trace in data_buffer:
                        st_merged += trace  # Merge each trace into the Stream
                    st_merged.sort()
                    st_merged.merge()
                    logging.info(f"st_merged: {st_merged}")
                    # Ensure that the time difference between the first and last trace is >= time_window
                    oldest_trace_starttime = data_buffer[0][0].stats.starttime
                    latest_trace_starttime = st_merged[-1].stats.endtime
                    time_diff = (latest_trace_starttime - oldest_trace_starttime)

                    if time_diff >= time_window:
                        logging.info(f"Time window condition met: {time_diff} seconds")
                        logging.info(f"Processing {len(data_buffer)} traces within the last {time_window} seconds...")

                        # Call the warn_check function to process the traces
                        features_norm = warn_check(st_merged, time_window)
                        
                        if features_norm is None:
                            logging.error("warn_check returned None. Skipping processing for this batch of traces.")
                            continue  # Skip this batch and continue with the next set of traces

                        # Load the model from the .pkl file
                        model = joblib.load('C:/Users/Utilizador/Desktop/IRONHACK/FinalProj/model_FINAL.pkl')
                        best_model = model.best_estimator_

                        # Perform predictions
                        for index, event in features_norm.iterrows():
                            y_prob = best_model.predict_proba([event])[:, 1]  # Probabilities for the positive class
                            custom_threshold = 0.01
                            y_pred_custom_threshold = (y_prob >= custom_threshold).astype(int)
                            if y_pred_custom_threshold == 1:
                                logging.error(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ALERT! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                            else:
                                logging.error("xxxxxxxxxxxxxxxxx  Nothing to report! xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ")
                    
                    else:
                        logging.info(f"Waiting for enough time window. Time difference: {time_difference} seconds")

                else:
                    logging.info(f"Waiting for 3 traces. Current trace count: {len(data_buffer)}")

            else:
                time.sleep(1)  # Avoid busy waiting 
                
    except Exception as e:
        logging.error(f"Error in process_trace: {e}", exc_info=True)  # Log with traceback

    logging.info("Finished processing all traces.")
    
  