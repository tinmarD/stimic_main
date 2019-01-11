##############################################
########      STIMIC Project       ###########
#

import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import utils_sigprocessing
import peakutils
import _pickle
from template_detection_50Hz import detect_50Hz_artifact

sns.set()
sns.set_context('paper')

stim_file_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P51_SC23\SC23-stimulations-1.edf'
csv_file_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P51_SC23\SC23-stimulations-1.csv'
mean_artifact_50Hz_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\50Hz_data\macro\mean_artifact_50Hz.p'
artifact_duration_50Hz, t_offset_50Hz = 5, 0.1
f_stim_hz = [6, 7, 20]

edf_raw = mne.io.read_raw_edf(stim_file_path, preload=True)
srate, n_chan, ch_names = edf_raw.info['sfreq'], edf_raw.info['nchan'], edf_raw.info['ch_names']
stim_thresh = 2000
merge_dur_s = 4
default_duration = 10

choc_times, choc_el, choc_freq, choc_dur = [], [], [], []
amp_max = 3150

nfft = 16384
f_hzton = lambda x, fmin, n, fs: int((2 + n) / fs * (x-fmin))

# Load mean 50Hz artifact
with open(mean_artifact_50Hz_path, 'rb') as f:
    mean_artifact = _pickle.load(f)

# For each channel
for i in range(n_chan):
    if 'EEG' not in ch_names[i]:
        continue
    channel_i = np.squeeze(edf_raw.get_data(i)*1E6)
    for stim_thresh_i in [abs(stim_thresh), -abs(stim_thresh)]:
        i_thresh_s = edf_raw.times[channel_i > stim_thresh_i] if stim_thresh_i > 0 else edf_raw.times[channel_i < stim_thresh_i]
        if i_thresh_s.size > 0:
            # Delete event if the previous one occured less than merge_dur_s sec. ago
            i_thresh_sel_s = i_thresh_s[np.hstack([True, np.array(i_thresh_s[1:] - i_thresh_s[:-1]) > merge_dur_s])]
            # Count the number of peaks in the time interval [-1s, +11s]
            for time_j in i_thresh_sel_s:
                start_j_sample = int((time_j-0.1)*srate)
                trace_j = channel_i[start_j_sample:int(start_j_sample + 10.1*srate)]
                trace_j_thresh = trace_j > stim_thresh_i if stim_thresh_i > 0 else trace_j < stim_thresh_i
                peaks_ind_j = np.where(trace_j_thresh & (~np.hstack([False, trace_j_thresh[:-1]])))[0]
                t_vect_j = np.linspace(time_j, time_j + trace_j.size / srate, trace_j.size)
                # Delete peak if the previous one occured less than 0.7 sec. ago
                peaks_ind_sel_j = peaks_ind_j[np.hstack([True, (peaks_ind_j[1:] - peaks_ind_j[:-1]) > int(0.8 * srate)])]
                if 7 < peaks_ind_sel_j.size < 13:
                    choc_times.append(time_j)
                    choc_el.append(ch_names[i])
                    choc_freq.append(1)
                    choc_dur.append(default_duration)

                # Look for stimulation with frequency above 1 Hz and below 50 Hz
                stim_freq_found = False
                if peaks_ind_j.size > 7:
                    # Compute PSD, remove very low frequency
                    window = signal.windows.hann(int(trace_j.size / 5))
                    n_overlap = int(window.size/2)
                    freqs, pxx = signal.welch(trace_j, srate, window=window, noverlap=n_overlap, nfft=nfft)
                    # freqs, pxx = signal.periodogram(trace_j, srate, nfft=nfft)
                    pxx_db = 10 * np.log10(pxx)
                    freqs, pxx_db = freqs[(0.3 < freqs) & (freqs < 100)], pxx_db[(0.3 < freqs) & (freqs < 100)]
                    # Compute PSD baseline and find peaks
                    f_max_ind = np.argmax(pxx_db)
                    f_max_hz = freqs[f_max_ind]
                    # Interpolate pxx_db at equally log-spaced frequencies
                    # freqs_log = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), 1000)
                    # pxx_db_log = np.interp(freqs_log, freqs, pxx_db)
                    pxx_db_base = utils_sigprocessing.get_signal_baseline(pxx_db, f_hzton(6, freqs[0], nfft, srate), 0.01, forder=5)
                    pxx_db_corrected = pxx_db - pxx_db_base
                    # pxx_db_log_base = utils_sigprocessing.get_signal_baseline(pxx_db_log, 20, 0.01, forder=5)
                    f_peaks_ind = peakutils.indexes(pxx_db - pxx_db_base)

                    # Look if frequency of max power is among the frequency of stimulation and more than 15dB above baseline
                    for f_stim_hz_i in f_stim_hz:
                        if f_stim_hz_i-0.5 < f_max_hz < f_stim_hz_i+0.5 and pxx_db_corrected[f_max_ind] > 15:
                            choc_times.append(time_j)
                            choc_el.append(ch_names[i])
                            choc_freq.append(f_stim_hz_i)
                            choc_dur.append(default_duration)
                            stim_freq_found = True
                            break

                    # Look if among the frequencies of stimulation there is one that is more than 20dB above the baseline
                    if not stim_freq_found:
                        for f_stim_hz_i in f_stim_hz:
                            f_stim_ind_i = f_hzton(f_stim_hz_i, freqs[0], nfft, srate)
                            if pxx_db_corrected[f_stim_ind_i] > 20:
                                choc_times.append(time_j)
                                choc_el.append(ch_names[i])
                                choc_freq.append(f_stim_hz_i)
                                choc_dur.append(default_duration)
                                stim_freq_found = True
                                break

                # 50 Hz stimulation
                if not stim_freq_found:
                    start_i = int((time_j+t_offset_50Hz)*srate)
                    trace_i = channel_i[start_i:int(start_i + artifact_duration_50Hz * srate)]
                    if detect_50Hz_artifact(trace_i, mean_artifact):
                        choc_times.append(time_j)
                        choc_el.append(ch_names[i])
                        choc_freq.append(50)
                        choc_dur.append(5)


choc_times, choc_el, choc_freq, choc_dur = np.array(choc_times), np.array(choc_el), np.array(choc_freq), np.array(choc_dur)
sort_vect = choc_times.argsort()
choc_times, choc_el, choc_freq, choc_dur = choc_times[sort_vect], choc_el[sort_vect], choc_freq[sort_vect], choc_dur[sort_vect]
choc_type = np.array(['stim {} Hz'.format(choc_freq_i) for choc_freq_i in choc_freq])


choc_chind = [ch_names.index(choc_el_i)+1 for choc_el_i in choc_el]

df_dict = {'type': choc_type, 'tpos': choc_times, 'channelname': choc_el, 'channelind': choc_chind, 'duration': choc_dur,
           'freq': choc_freq}
df = pd.DataFrame(df_dict)
# Remove events on the same channel and close in time
to_remove = []
for ch_name in df['channelname'].unique():
    df_i = df[df['channelname'] == ch_name]
    tpos_i = np.array(df_i.tpos)
    to_remove_i = np.hstack([False, (tpos_i[1:] - tpos_i[:-1]) < 4])
    to_remove.append(df_i.index[to_remove_i])
to_remove_arr = np.hstack(to_remove)
df = df.drop(to_remove_arr).reset_index(drop=True)


df.to_csv(csv_file_path, sep=',')

