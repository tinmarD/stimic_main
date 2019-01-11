"""
    STIMIC Project - Stim Event List Creation - step 1
                EDF File Cleaning

Open all the eeg data files (EDF files), remove the non-EEG channels, the bad channels (channel names with '...' in it).
Check that each file contains the same number of channels and that channels are ordered the same across each file.
If a channel is missing in some file, an NaN channel is added.
These clean files will allow to create epochs without problems and that the same index refers to the same channel
across each file
This script does not resample data if the sampling frequency is different across files, the resampling should be done
after epoching

See the doc here : https://tinmard.github.io/stimic-stimulation-listing.html

"""

import numpy as np
import mne
import os
import re
from utils_eeg_cleaning import *

patient_base_dir = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P56_AB28'
stim_data_dir = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P56_AB28\raw_data\Macro'
stim_data_dir_out = os.path.join(patient_base_dir, 'data_clean')
wrong_channel_patterns = ['y[ \d]+', 'Z', 'V', 'X', '\+', 'emg.*', 'eog.*', 'ecg.*', 'td', 'tg', 'og','od']

# Open data file and count the number of EDF files
filenames = np.array(os.listdir(stim_data_dir))
edf_files = filenames[['.edf' in file_i for file_i in filenames]]
n_files = edf_files.size
if n_files == 0:
    raise ValueError('Could not detect any EDF file. Check the stim_data_dir path.')
# Create output dir
if not os.path.exists(stim_data_dir_out):
    os.mkdir(stim_data_dir_out)

# Open all files
ch_names_clean_all = []
for i, edf_filename_i in enumerate(edf_files):
    raw_eeg = mne.io.read_raw_edf(os.path.join(stim_data_dir, edf_filename_i), preload=True)
    ch_names_clean_all_i, _ = get_clean_eeg_channelnames(raw_eeg.ch_names, wrong_channel_patterns)
    ch_names_clean_all.append(ch_names_clean_all_i)

# Get all the different channel names considering all the files (even if channel is present in only 1 file)
ch_names_all_files = np.hstack(ch_names_clean_all)
unique_index = np.unique(ch_names_all_files, return_index=True)[1]
ch_names_sel = ch_names_all_files[np.sort(unique_index)]
n_chan_sel = ch_names_sel.size

# Get the electrode names, numbers and channels number for each channel of the selected file
el_names, el_num, ch_num = get_electrode_info(ch_names_sel)

# Sort the channels, first by the electrode number, and then by the channel number
if ch_num[0].size == 1:     # Monopolar
    sort_ind = np.lexsort((ch_num, el_num))
else:   # Bipolar
    raise ValueError('TOOD for bipolar montage')

ch_names_sel_sorted = ch_names_sel[sort_ind]

# Re-open all the files and add the missing channels
for i, edf_filename_i in enumerate(edf_files):
    raw_eeg_i = mne.io.read_raw_edf(os.path.join(stim_data_dir, edf_filename_i), preload=True)
    ch_names_clean_i = ch_names_clean_all[i]
    # Select clean name EEG channels
    raw_eeg_i = raw_eeg_i.pick_channels(ch_names_clean_i)
    n_chan_i = raw_eeg_i.info['nchan']
    # If some channels are missing in file i
    if not n_chan_i == n_chan_sel:
        miss_channel_names = ch_names_sel[[chname_i not in ch_names_clean_i for chname_i in ch_names_sel]]
        print('Missing channels {} in file : {}. Add empty channels (NaN channel)'.format(miss_channel_names,
                                                                                          edf_filename_i))
        data_i = np.NaN * np.ones((n_chan_sel, raw_eeg_i.n_times))
        data_i[:n_chan_i, :] = raw_eeg_i.get_data()
        ch_names_i = list(np.hstack([ch_names_clean_i, miss_channel_names]))
    else:
        data_i, ch_names_i = raw_eeg_i.get_data(), raw_eeg_i.ch_names
    # Reorder data_i in the same way than the sorted selected file
    ordering_vect = np.array([ch_names_i.index(ch_i) for ch_i in ch_names_sel_sorted])
    data_i = data_i[ordering_vect, :]
    info = mne.create_info(ch_names=list(ch_names_sel_sorted), ch_types=['eeg']*n_chan_sel, sfreq=raw_eeg_i.info['sfreq'])
    new_raw_i = mne.io.RawArray(data_i, info)
    new_raw_i.save(os.path.join(stim_data_dir_out, '{}_clean_raw.fif'.format(edf_filename_i[:-4])), fmt='single')

