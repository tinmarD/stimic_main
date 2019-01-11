# Read time of stim onset, stim channel
# Extract data from the stim onset and for a fixed period of time (stim duration), equal for every stim of a patient
# Smooth epochs to remove fast variations to keep only the trend of variation over seconds

import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
import re
import _pickle
import utils_sigprocessing
import peakutils

sns.set()
sns.set_context('paper')


if __name__ == '__main__':
    stim_50Hz_csv_filepath = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\50Hz_data\macro\P51_SC53_stim_1.csv'
    stim_eeg_filepath = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P51_SC23\SC23-stimulations-1.edf'
    # stim_50Hz_csv_filepath = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\50Hz_data\macro\P52_BA24_stim1.csv'
    # stim_eeg_filepath = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P52_BA24\BA24-STIMULATIONS1.edf'
    mean_artifact_50Hz_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\50Hz_data\macro\mean_artifact_50Hz.p'
    stim_duration_s = 5
    stim_desc = re.split('\\\\', stim_50Hz_csv_filepath)[-1]

    # Load mean 50Hz artifact
    with open(mean_artifact_50Hz_path, 'rb') as f:
        mean_50hz_artifact = _pickle.load(f)

    # Read csv file
    df = pd.read_csv(stim_50Hz_csv_filepath)
    stim_times, stim_chan_name, stim_chan_ind = np.array(df['tpos']), np.array(df['channelname']), np.array(df['channelind'])
    n_stims = stim_times.size

    # Read EEG file (edf format)
    edf_raw = mne.io.read_raw_edf(stim_eeg_filepath, preload=True)
    srate, n_chan, ch_names = edf_raw.info['sfreq'], edf_raw.info['nchan'], edf_raw.info['ch_names']
    stim_thresh = 3150

    # Low pass filter
    f_cutoff = 1
    b, a = signal.butter(6, 2*f_cutoff/srate, 'lowpass')
    b_fir, a_fir = signal.firwin(200, 2*f_cutoff/srate), 1
    n_pnts = int(stim_duration_s*srate)
    artefacts, artefacts_smoothed = np.zeros((2, n_stims, n_pnts))

    t_offset = 0.1
    for i in range(n_stims):
        i_start = int((stim_times[i]+t_offset)*srate)
        artefacts[i, :] = edf_raw._data[stim_chan_ind[i]-1, i_start:i_start+n_pnts] * 1E6
        # artefacts_smoothed[i, :] = signal.filtfilt(b, a, artefacts[i, :])
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

    # Plot all artefact superposed
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(t_vect, artefacts_smoothed_up.T, alpha=0.5)
    ax.plot(t_vect, np.mean(artefacts_smoothed_up, axis=0), c='k', lw=2)
    ax.set(ylabel='Amp (uV)', xlabel='time (s)', title='Upward Smoothed 50Hz Artefact on stimulation channels (monopolar) - {}'.format(stim_desc))
    ax.autoscale(axis='x', tight=True)

    # Compute correlation with the mean 50Hz artifact
    corr_spearman_mean_artifact = np.zeros(n_stims)
    for i in range(n_stims):
        artefact_i = artefacts_smoothed[i] if artefacts_smoothed[i, 0] >= 0 else - artefacts_smoothed[i]
        corr_spearman_mean_artifact[i] = stats.spearmanr(mean_50hz_artifact, artefact_i)[0]

    plt.figure()
    plt.hist(corr_spearman_mean_artifact, bins=20)

    # Zero crossing time, detect the time when the waveform crosses 0
    zct = np.zeros(n_stims)
    for i in range(n_stims):
        artefact_i = artefacts_smoothed[i]
        zc_pos = np.where(np.diff(np.sign(artefact_i)))[0]
        zct[i] = t_vect[zc_pos[0]]

    plt.figure()
    plt.hist(zct, bins=20)

    df = pd.DataFrame({'corr': corr_spearman_mean_artifact, 'zct': zct, 'time': df['tpos']})
    sns.jointplot('corr', 'zct', data=df)

    i = 13
    f = plt.figure()
    t_vect = np.linspace(t_offset, t_offset + stim_duration_s, n_pnts)
    ax = f.add_subplot(211)
    ax.plot(t_vect, artefacts[i])
    ax.plot(t_vect, mean_50hz_artifact, 'k', lw=2)
    # ax.text(t_vect[int(0.7 * n_pnts)], 3000, 'Spearman : {:.3f}'.format(corr_mat_spearman[i, j]))
    ax2 = f.add_subplot(212, sharex=ax)
    ax2.plot(t_vect, artefacts_smoothed[i])
    ax2.plot(t_vect, mean_50hz_artifact, 'k', lw=2)
    ax2.text(t_vect[int(0.7 * n_pnts)], 3000, 'Spearman : {:.3f}'.format(corr_spearman_mean_artifact[i]))

    # f = plt.figure()
    # ax = f.add_subplot(111)
    # ax.scatter(corr_spearman_mean_artifact, zct)
    # ax.set(xlabel='Spearman correlation', ylabel='Zero-crossing Time (s)', title='Artifact smoothed waveform')

