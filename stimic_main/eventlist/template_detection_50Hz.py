"""
    STIMIC Project - Stim Event List Creation
          50 Hz mean Artefact template

Read time of stim onset, stim channel.
Extract data from the stim onset and for a fixed period of time (stim duration), equal for every stim of a patient
Smooth epochs to remove fast variations to keep only the trend of variation over seconds

See the doc here : https://tinmard.github.io/stimic-stimulation-listing.html

"""
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
import re
import os
import _pickle
import utils_sigprocessing
import peakutils

sns.set()
sns.set_context('paper')


stim_spreadsheet_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P53_CD25\p53_CD25_stimulations_all.xlsx'
stim_epoch_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P53_CD25\P53_CD25_stim_mne_epoch-resync-epo.fif.'
stim_duration_s = 5
# mean_50hz_artifact_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\50Hz_data\macro\mean_artifact_50Hz_ori.p'
mean_50hz_artifact_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\50Hz_data\macro\mean_50Hz_artefact_P51_SC23.p'
mean_50hz_artifact_dir = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\50Hz_data\macro'
# stim_desc = re.split('\\\\', stim_50Hz_csv_filepath)[-1]

# Load mean 50Hz artifact
with open(mean_50hz_artifact_path, 'rb') as f:
    mean_50hz_artifact = _pickle.load(f)

# Read csv file
df = pd.read_excel(stim_spreadsheet_path)
df_50Hz = df[df.freq == 50]

# Read Epoch struct
stim_epoch = mne.read_epochs(stim_epoch_path, preload=True)
srate, n_chan, ch_names = stim_epoch.info['sfreq'], stim_epoch.info['nchan'], stim_epoch.info['ch_names']
epoch_data_50Hz = stim_epoch.get_data()[df_50Hz.index, :, :]
n_stims, n_chan, n_pnts = epoch_data_50Hz.shape
stim_thresh = 3150

# Low pass filter design
f_cutoff = 1
b, a = signal.butter(6, 2*f_cutoff/srate, 'lowpass')
b_fir, a_fir = signal.firwin(200, 2*f_cutoff/srate), 1

# Select time point
t_offset = 0.1
time_sel_ind = ((stim_epoch.times + t_offset) > 0) & ((stim_epoch.times+t_offset) < stim_duration_s)
n_pnts = sum(time_sel_ind)

# Get stimulation data and filter it
artefacts, artefacts_smoothed = np.zeros((2, n_stims, n_pnts))
for i, chan_pos_i in enumerate(df_50Hz.channelind):
    artefacts[i, :] = epoch_data_50Hz[i, chan_pos_i, time_sel_ind]
    artefacts_smoothed[i, :] = signal.filtfilt(b, a, artefacts[i, :])


# Plot all artefact superposed (with smoothed version)
f = plt.figure()
t_vect = np.linspace(t_offset, t_offset+stim_duration_s, n_pnts)
ax = f.add_subplot(211)
ax.set(ylabel='Amp (uV)', xlabel='time (s)', title='Raw 50Hz Artefact on stimulation channels (monopolar)')
ax.plot(t_vect, artefacts.T, alpha=0.5)
ax2 = f.add_subplot(212, sharex=ax)
ax2.plot(t_vect, artefacts_smoothed.T, alpha=0.5)
ax2.set(ylabel='Amp (uV)', xlabel='time (s)', title='Smoothed 50Hz Artefact on stimulation channels (monopolar)')
ax2.autoscale(axis='x', tight=True)

# Compute spearman correlation between each pair of artefacts
corr_mat_spearman = np.zeros((n_stims, n_stims))
corr_mat_smoothed_spearman = np.zeros((n_stims, n_stims))
for i in range(n_stims):
    for j in range(n_stims):
        corr_mat_spearman[i, j] = stats.spearmanr(artefacts[i], artefacts[j])[0]
        corr_mat_smoothed_spearman[i, j] = stats.spearmanr(artefacts_smoothed[i], artefacts_smoothed[j])[0]

# Plot the absolute value of the Spearman correlation matrices (raw and smoothed)
f = plt.figure()
sns.heatmap(np.abs(corr_mat_spearman), vmin=0, vmax=1)
plt.title('Absolute value of Spearman correlation between artifact waveforms')
f = plt.figure()
sns.heatmap(np.abs(corr_mat_smoothed_spearman), vmin=0, vmax=1)
plt.title('Absolute value of Spearman correlation between artifact smoothed waveforms')

# Plot the histograms of the absolute value of the Spearman correlation matrices
f = plt.figure()
ax1 = f.add_subplot(121)
ax1.hist(np.abs(corr_mat_spearman[np.tril_indices(n=n_stims, k=-1)]), bins=30)
ax1.set(title='Absolute value of Spearman Correlation between raw artifact waveforms')
ax2 = f.add_subplot(122)
ax2.hist(np.abs(corr_mat_smoothed_spearman[np.tril_indices(n=n_stims, k=-1)]), bins=30)
ax2.set(title='Absolute value of Spearman Correlation between smoothed artifact waveforms')


# Compute a mean artifact waveform
# If the waveform is negative at the beginning, take the opossite
artefacts_smoothed_up = np.zeros(artefacts_smoothed.shape)
for i in range(n_stims):
    if artefacts_smoothed[i, 0] > 0:
        artefacts_smoothed_up[i] = artefacts_smoothed[i]
    else:
        artefacts_smoothed_up[i] = -artefacts_smoothed[i]

# Compute the mean artefact for this patient :
mean_patient_artefact = np.mean(artefacts_smoothed_up, axis=0)
# with open(os.path.join(mean_50hz_artifact_dir, 'mean_50Hz_artefact_P51_SC23.p'), 'wb') as f:
#     f.write(mean_patient_artefact)

# Plot all artefact superposed
f = plt.figure()
ax = f.add_subplot(111)
ax.plot(t_vect, artefacts_smoothed_up.T, alpha=0.5)
mean_line, = ax.plot(t_vect, mean_patient_artefact, c='k', lw=2, label='mean')
ax.legend(handles=[mean_line])
ax.set(ylabel='Amp (uV)', xlabel='time (s)', title='Upward Smoothed 50Hz Artefact on stimulation channels (monopolar)')
ax.autoscale(axis='x', tight=True)


# Compute correlation with the mean 50Hz artifact
corr_spearman_mean_artifact, corr_spearman_mean_patient_artefact = np.zeros((2, n_stims))
for i in range(n_stims):
    artefact_i = artefacts_smoothed[i] if artefacts_smoothed[i, 0] >= 0 else - artefacts_smoothed[i]
    corr_spearman_mean_artifact[i] = stats.spearmanr(mean_50hz_artifact, artefact_i)[0]
    corr_spearman_mean_patient_artefact[i] = stats.spearmanr(mean_patient_artefact, artefact_i)[0]

f = plt.figure()
ax = f.add_subplot(121)
n, bins, _ = ax.hist(np.abs(corr_spearman_mean_artifact), bins=20, alpha=0.6)
n_patient, bins, _ = ax.hist(np.abs(corr_spearman_mean_patient_artefact), bins=20, alpha=0.6)
ax.set(xlabel='Correlation', ylabel='Count', title='Spearman Correlation of smoothed waveforms')
ax.legend(['mean_50Hz_artefact', 'mean_patient_artefact'])
ax.autoscale(axis='x', tight=True)
ax2 = f.add_subplot(122)
ax2.hist(np.abs(corr_spearman_mean_artifact), bins=20, histtype='step', cumulative=1)
ax2.hist(np.abs(corr_spearman_mean_patient_artefact), bins=20, histtype='step', cumulative=1)
ax2.set(xlabel='Correlation', ylabel='Count', title='Cumulative Step Histogram')
ax2.legend(['mean_50Hz_artefact', 'mean_patient_artefact'])
ax2.autoscale(axis='x', tight=True)


# Zero crossing time, detect the time when the waveform crosses 0
zct = np.zeros(n_stims)
for i in range(n_stims):
    artefact_i = artefacts_smoothed[i]
    zc_pos = np.where(np.diff(np.sign(artefact_i)))[0]
    zct[i] = t_vect[zc_pos[0]]

plt.figure()
plt.hist(zct, bins=20)

# i = 13
# f = plt.figure()
# t_vect = np.linspace(t_offset, t_offset + stim_duration_s, n_pnts)
# ax = f.add_subplot(211)
# ax.plot(t_vect, artefacts[i])
# ax.plot(t_vect, mean_50hz_artifact, 'k', lw=2)
# # ax.text(t_vect[int(0.7 * n_pnts)], 3000, 'Spearman : {:.3f}'.format(corr_mat_spearman[i, j]))
# ax2 = f.add_subplot(212, sharex=ax)
# ax2.plot(t_vect, artefacts_smoothed[i])
# ax2.plot(t_vect, mean_50hz_artifact, 'k', lw=2)
# ax2.text(t_vect[int(0.7 * n_pnts)], 3000, 'Spearman : {:.3f}'.format(corr_spearman_mean_artifact[i]))
