"""
    STIMIC Project - Stim Event List Creation - step 2
                    Stim Detection

Semi-automatic detector for stimulations. Creates an output csv file containing the events detected
by the algorithm. Run this script for every file.
MAKE SURE that the mid_freqz_hz variable contains all the stimulation frequencies used for the specified file, apart
from 1Hz and 50Hz.
For 50Hz artifact detection, you will need the artifact template, called mean_artifact_50Hz_ori.p

See the doc here : https://tinmard.github.io/stimic-stimulation-listing.html

"""

import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy import signal, stats
import _pickle
import utils_sigprocessing
from stim_detection_fun import *
import tqdm
import re

sns.set()
sns.set_context('paper')

# MODIFY THESE 3 VARIABLES :
stim_file_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P56_AB28\data_clean\AB28-stim4_clean_raw-1.fif'
mean_artifact_50Hz_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\50Hz_data\macro\mean_artifact_50Hz_ori.p'
mid_freqs_hz = [6, 10, 30]  # Stim frequencies (apart from 1Hz and 50Hz)

csv_out_file_path = '{}.csv'.format(stim_file_path[:-4])
patient_ID = re.split('\\\\', stim_file_path)[-2]

stim_thresh_1hz, stim_thresh_midfreq, stim_thresh_50Hz = 3150, 3150, 2000
stim_dur_1hz, stim_dur_midfreq, stim_dur_50hz = 10, 10, 5
time_int_1hz, time_int_midfreq, time_int_50hz = [-0.1, 10.1], [-0.1, 10.1], [+0.1, 5]
merge_duration_s = 4    # If a threshold crossing event occurerd less than merge_duration_s sec. ago, delete it
merge_events_s = 5
spearman_corr_thresh_50hz, zero_crossing_limits = 0.75, [1.2, 1.8]

# Load EEG file
if stim_file_path[-3:] == 'edf':
    raw = mne.io.read_raw_edf(stim_file_path, preload=True)
elif stim_file_path[-3:] == 'fif':
    raw = mne.io.read_raw_fif(stim_file_path, preload=True)
else:
    raise ValueError('Wrong data extension in file {}'.format(stim_file_path))
srate, n_chan, ch_names, tmax = raw.info['sfreq'], raw.info['nchan'], raw.info['ch_names'], raw.times[-1]

stim_times, stim_chan_ind, stim_freq, stim_dur = [], [], [], []

f_hzton = lambda f, fmin, n, fs: int((2 + n) / fs * (f-fmin))

# Spectral Analysis parameters
window_duration_s = 2
overlap_s = 0.5*window_duration_s
nfft = 16384

# Load mean 50Hz artifact
with open(mean_artifact_50Hz_path, 'rb') as f:
    mean_50hz_artifact = _pickle.load(f)

# For each channel
for i in tqdm.tqdm(range(n_chan)):
    if 'EEG' not in ch_names[i]:
        continue
    channel_i = np.squeeze(raw.get_data(i)*1E6)

    # 1Hz stimulation
    for thresh_1hz_i in [abs(stim_thresh_1hz), -abs(stim_thresh_1hz)]:
        thresh_crossed_times, _ = stim_threshold_crossing_points(channel_i, srate, thresh_1hz_i, merge_duration_s)
        thresh_crossed_times = thresh_crossed_times[(thresh_crossed_times > -time_int_1hz[0]) &
                                                    (thresh_crossed_times < (tmax-time_int_1hz[1]))]
        for t_thresh in thresh_crossed_times:
            trace_j = channel_i[int((t_thresh+time_int_1hz[0])*srate):int((t_thresh+time_int_1hz[1])*srate)]
            peak_times, _ = stim_get_threshold_peaks(trace_j, srate, thresh_1hz_i, 0.7)
            if 8 < peak_times.size < 12:
                stim_times.append(t_thresh), stim_chan_ind.append(i)
                stim_freq.append(1), stim_dur.append(stim_dur_1hz)

    # Mid-frequency stimulations
    for thresh_mid_freq_i in [abs(stim_thresh_midfreq), -abs(stim_thresh_midfreq)]:
        thresh_crossed_times, _ = stim_threshold_crossing_points(channel_i, srate, thresh_mid_freq_i, merge_duration_s)
        thresh_crossed_times = thresh_crossed_times[(thresh_crossed_times > -time_int_midfreq[0]) &
                                                    (thresh_crossed_times < (tmax - time_int_midfreq[1]))]
        for t_thresh in thresh_crossed_times:
            trace_j = channel_i[int((t_thresh + time_int_midfreq[0]) * srate):int((t_thresh + time_int_midfreq[1]) * srate)]
            # Compute PSD
            window, noverlap = signal.windows.hann(int(srate*window_duration_s)), int(srate*overlap_s)
            freqs, pxx = signal.welch(trace_j, srate, window=window, noverlap=noverlap, nfft=nfft)
            pxx_db = np.log10(pxx)
            freqs, pxx_db = freqs[(0.3 < freqs) & (freqs < 100)], pxx_db[(0.3 < freqs) & (freqs < 100)]
            # Compute PSD baseline
            pxx_db_base = utils_sigprocessing.get_signal_baseline(pxx_db, f_hzton(6, freqs[0], nfft, srate), 0.01, forder=5)
            pxx_db_corrected = pxx_db - pxx_db_base
            for f_hz in mid_freqs_hz:
                f_n = f_hzton(f_hz, freqs[0], nfft, srate)
                if pxx_db_corrected[f_n] > 15:
                    stim_times.append(t_thresh), stim_chan_ind.append(i)
                    stim_freq.append(f_hz), stim_dur.append(stim_dur_midfreq)

    # 50Hz stimulation
    for thresh_50hz_i in [abs(stim_thresh_50Hz), -abs(stim_thresh_50Hz)]:
        thresh_crossed_times, _ = stim_threshold_crossing_points(channel_i, srate, thresh_50hz_i, merge_duration_s)
        thresh_crossed_times = thresh_crossed_times[(thresh_crossed_times > -time_int_50hz[0]) &
                                                    (thresh_crossed_times < (tmax - time_int_50hz[1]))]
        for t_thresh in thresh_crossed_times:
            trace_j = channel_i[int((t_thresh + time_int_50hz[0]) * srate):int((t_thresh + time_int_50hz[1]) * srate)]
            corr_with_mean_pattern = stim_50hz_correlation_with_mean_artifact(trace_j, mean_50hz_artifact)
            zero_cross_times, _ = stim_50Hz_get_zero_crossing(trace_j, srate)

            if corr_with_mean_pattern > spearman_corr_thresh_50hz and zero_cross_times.size == 1:
                if zero_crossing_limits[0] < zero_cross_times[0] < zero_crossing_limits[1]:
                    stim_times.append(t_thresh), stim_chan_ind.append(i)
                    stim_freq.append(50), stim_dur.append(stim_dur_50hz)

# Sort the stim by increasing times
if len(stim_times) == 0:
    print('No stimulations found')
else:
    sort_vect = np.argsort(np.array(stim_times))
    stim_times, stim_freq = np.array(stim_times)[sort_vect], np.array(stim_freq)[sort_vect]
    stim_dur, stim_chan_ind = np.array(stim_dur)[sort_vect], np.array(stim_chan_ind)[sort_vect]
    stim_chan_name = np.array([ch_names[i] for i in stim_chan_ind])
    stim_type = np.array(['Stim {} Hz'.format(freq_i) for freq_i in stim_freq])

    # Often stim can be detected on many channel, thus creating many events
    # Clean this by keeping only the 2 events where the amplitude is maximal when events are close one to another
    stim_to_keep_pos = []
    i = 0
    while i < len(stim_times):
        stim_time_i = stim_times[i]
        # Find events merge_events_s second apart from this one (events are time sorted)
        stim_grp_pos = np.where((stim_times >= stim_time_i) & (stim_times < stim_time_i + merge_events_s))[0]
        if stim_grp_pos.size <= 2:
            stim_to_keep_pos.append(i)
            i += 1
            continue
        else:
            time_grp, chan_grp, dur_grp = stim_times[stim_grp_pos], stim_chan_ind[stim_grp_pos], stim_dur[stim_grp_pos]
            n_grp_events = time_grp.size
            max_amp = np.zeros(n_grp_events)
            # For each of this event, look at the max amplitude
            for j in range(n_grp_events):
                stim_trace_j = 1E6*raw.get_data(picks=chan_grp[j], start=int(srate*time_grp[j]),
                                                stop=int(srate*(time_grp[j]+dur_grp[j])))
                max_amp[j] = max(np.abs(stim_trace_j).squeeze())
            for j in stim_grp_pos[np.argsort(max_amp)[-2:]]:
                stim_to_keep_pos.append(j)
            i = stim_grp_pos[-1] + 1

    stim_to_keep_pos = np.array(stim_to_keep_pos)
    # Select stims
    stim_times, stim_freq = stim_times[stim_to_keep_pos], stim_freq[stim_to_keep_pos]
    stim_dur, stim_chan_ind = stim_dur[stim_to_keep_pos], stim_chan_ind[stim_to_keep_pos]
    stim_chan_name, stim_type = stim_chan_name[stim_to_keep_pos], stim_type[stim_to_keep_pos]

    # Create pandas DataFrame
    df_dict = {'ID': patient_ID, 'filepath': stim_file_path, 'type': stim_type, 'tpos': stim_times, 'channelname': stim_chan_name, 'channelind': stim_chan_ind,
               'duration': stim_dur,  'freq': stim_freq}
    df = pd.DataFrame(df_dict)
    # Remove events on the same channel, close in time, and with the same frequency of stimulation
    to_remove = []
    for ch_name in df['channelname'].unique():
        df_i = df[df['channelname'] == ch_name]
        tpos_i, freq_i = np.array(df_i.tpos), np.array(df_i.freq)
        to_remove_i = np.hstack([False, (np.diff(tpos_i) < merge_duration_s) & (np.diff(freq_i) == 0)])
        to_remove.append(df_i.index[to_remove_i])
    to_remove_arr = np.hstack(to_remove)
    df = df.drop(to_remove_arr).reset_index(drop=True)
    df.to_csv(csv_out_file_path, sep=',')
