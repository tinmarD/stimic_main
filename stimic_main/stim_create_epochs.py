"""
    STIMIC Project - Stimulation Epoch Creation

Creates MNE epochs data structures containing EEG data around each stimulation.

See the doc here : https://tinmard.github.io/stimic-stimulation-listing.html

"""

import numpy as np
import pandas as pd
from scipy import signal
import re
import os
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from utils_eeg_io import io_eeg_macro_to_mne
from utils_stim import *
sns.set()
sns.set_context('paper')


stim_spreadsheet_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P51_SC23\Spreadsheets\p51_SC53_stimulations_all.xlsx'
stim_data_dir = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P51_SC23\data_clean'
epochs_out_dirpath = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P51_SC23'
patient_id = re.split('\\\\', stim_spreadsheet_path)[-2]
resync = True                   # If True, resynchronize the onset of the stimulation
inverse_neg_stim = True         # If True, inverse the negative stimulation so that all stimulation appears positive and the mean is not biased
select_all_channel = True       # If True select all the channel for creating the epoch, otherwise select only the stimulation channel

# Epochs parameters
t_pre, t_post = -5, 25
srate = 512
n_pnts_epoch = int((abs(t_pre)+abs(t_post))*srate)

# Load stim spreadsheet
df = pd.read_excel(stim_spreadsheet_path)
n_stims = df.shape[0]
if not select_all_channel:
    epoch_data = np.zeros((n_stims, n_pnts_epoch))
    last_filename, raw_eeg = '', []
else:
    # Read first eeg file from the DataFrame
    last_filename = df.file[0]
    last_filename = '{}.fif'.format(last_filename) if '.fif' not in last_filename else last_filename
    raw_eeg = io_eeg_macro_to_mne(os.path.join(stim_data_dir, last_filename))
    ch_names_eeg, n_chan = raw_eeg.ch_names, raw_eeg.info['nchan']
    epoch_data = np.zeros((n_stims, n_chan, n_pnts_epoch))

for i in range(n_stims):
    filename_i, time_i, chanind_i = df.file[i], df.time[i], df.channelind[i]
    filename_i = '{}.fif'.format(filename_i) if '.fif' not in filename_i else filename_i
    # Load macro file if not already loaded
    if not filename_i == last_filename:
        raw_eeg = io_eeg_macro_to_mne(os.path.join(stim_data_dir, filename_i))
        if select_all_channel:
            # Make sure that the channels are the same than in the previous file and in the same order
            raw_eeg = raw_eeg.pick_channels(ch_names_eeg)
            ch_names_i = np.array(raw_eeg.ch_names)
            if not raw_eeg.info['nchan'] == n_chan:
                raise ValueError('Number of channels vary across files. Do not forget to run the stim_clean_EDF_files.py first !')
            if not raw_eeg.ch_names == ch_names_eeg:
                raise ValueError('Channels are not the same or in the same order across files. Do not forget to run the stim_clean_EDF_files.py first !')
    srate_i = raw_eeg.info['sfreq']
    # Select the part around the i-th stim
    if not select_all_channel:
        if not srate_i == srate:
            data_temp = raw_eeg.get_data(picks=chanind_i, start=int(srate_i*(time_i+t_pre)),
                                           stop=int(srate_i*(time_i+t_post))).squeeze()*1E6
            epoch_data[i, :] = signal.resample(data_temp, num=n_pnts_epoch)
        else:
            epoch_data[i, :] = raw_eeg.get_data(picks=chanind_i, start=int(srate_i*(time_i+t_pre)),
                                                  stop=int(srate_i*(time_i+t_post))).squeeze()*1E6
    else:
        if not srate_i == srate:
            data_temp = raw_eeg.get_data(start=int(srate_i*(time_i+t_pre)),
                                           stop=int(srate_i*(time_i+t_post))).squeeze()*1E6
            epoch_data[i, :, :] = signal.resample(data_temp, num=n_pnts_epoch, axis=1)
        else:
            epoch_data[i, :, :] = raw_eeg.get_data(start=int(srate_i*(time_i+t_pre)),
                                                     stop=int(srate_i*(time_i+t_post))).squeeze()*1E6
    last_filename = filename_i

# Resync data if resync is True
if resync:
    time = np.linspace(-abs(t_pre), t_post, n_pnts_epoch)
    if select_all_channel:
        epoch_data = resync_traces_3d(epoch_data, time, df, inverse_neg_stim=inverse_neg_stim)
    else:
        epoch_data = resync_traces_2d(epoch_data, time, df, inverse_neg_stim=inverse_neg_stim)

# Create the mne-structure
info = mne.create_info(ch_names=ch_names_eeg, ch_types=['eeg']*n_chan, sfreq=srate)
mne_epochs = mne.EpochsArray(epoch_data, info, tmin=t_pre)

# Save it
if resync:
    stim_epoch_filepath, i = os.path.join(epochs_out_dirpath, '{}_stim_mne_epoch-resync-epo.fif'.format(patient_id)), 2
else:
    stim_epoch_filepath, i = os.path.join(epochs_out_dirpath, '{}_stim_mne_epoch-epo.fif'.format(patient_id)), 2

while os.path.exists(stim_epoch_filepath):
    stim_epoch_filepath = re.sub('-epo.fif$', '-{}-epo.fif'.format(i), stim_epoch_filepath)
    i += 1
mne_epochs.save(stim_epoch_filepath)

