"""
    STIMIC Project - Stim Event List Creation - step 5
             Macro to micro correpondency

Use this script to create the stimulation spreadsheet for micro-file from the Macro-file stim spreadsheet.
It needs the time offset between macro and micro file and the channel correpondency. You can use the Resync Signals
tool in micMac for this : https://micmac.readthedocs.io/en/latest/signals/resynch_signals.html
To one Macro stimulation site, i.e. between 2 macro contacts, may correpond either no micro-electrodes or 2 or 3
tetrodes, so 8 to 12 micro channels

See the doc here : https://tinmard.github.io/stimic-stimulation-listing.html

"""

import pandas as pd
import numpy as np
import re
import os
import utils_eeg_io

## Parameters
spreadsheet_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P56_AB28\data_clean\AB28-stim4_clean_raw_mm.xlsx'
stim_macro_filepath = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P56_AB28\data_clean\AB28-stim4_clean_raw.fif'
stim_micro_filepath = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P56_AB28\raw_data\micro\AB28-stim4-20181114-170021\20181114-170021-001-s001-combined_5kHz.edf'
# Time offset between the macro file and the micro file
time_offset_macromicro = 94.2905


# If the file is a second part of a file (with fif file it add a -1 at the end of the filename), the duration from the
# first part must be added to the micro time (which file should not be divided)
if re.search('-\d', stim_macro_filepath):
    file_part_num = int(re.search('-\d', stim_macro_filepath)[0][1:])
    part_duration = []
    for i_part in range(file_part_num+1):
        if i_part == 0:
            stim_macro_part_i_path = re.sub('-\d', '', stim_macro_filepath)
            raw_part_i, _ = utils_eeg_io.io_eeg_to_mne(stim_macro_part_i_path, False)
            whole_duration = raw_part_i.n_times / raw_part_i.info['sfreq']
        else:
            stim_macro_part_i_path = re.sub('-\d', '-{}'.format(i_part), stim_macro_filepath)
            raw_part_i, _ = utils_eeg_io.io_eeg_to_mne(stim_macro_part_i_path, False)
            part_duration.append(raw_part_i.n_times / raw_part_i.info['sfreq'])
        if not os.path.isfile(stim_macro_part_i_path):
            raise ValueError('Could not the first part of the file which is supposed to be : {}'.format(stim_macro_filepath))

    time_offset_macromicro -= (whole_duration - sum(part_duration))


_, macro_chnames = utils_eeg_io.io_eeg_to_mne(stim_macro_filepath, False)
_, micro_chnames = utils_eeg_io.io_eeg_to_mne(stim_micro_filepath, False)
micro_filename = os.path.split(stim_micro_filepath)[1]

micro_chnames = [re.sub('EEG +','', micro_chname_i) for micro_chname_i in micro_chnames ]
micro_elname_all = [re.search('\D+', micro_chname_i)[0] for micro_chname_i in micro_chnames]
micro_elnames = np.unique(micro_elname_all)

df = pd.read_excel(spreadsheet_path)
n_stims = df.shape[0]

micro_elname_corr, micro_channelind = [], []
for i in range(n_stims):
    macro_chname_i = df['channelname'][i]
    macro_chname_i = re.sub('EEG', '', macro_chname_i)
    macro_chname_i = re.sub(' ', '', macro_chname_i)
    macro_elname_i = re.search('\D+', macro_chname_i)[0]
    # Try to find a correponding micro electrode name
    if macro_elname_i.lower() in micro_elnames:
        micro_elname_i = macro_elname_i.lower()
    else:
        macro_elname_i = re.sub('\'', 'p', macro_elname_i)
        if macro_elname_i.lower() in micro_elnames:
            micro_elname_i = macro_elname_i.lower()
        else:
            micro_elname_i = []
    micro_elname_corr.append(micro_elname_i)
    # If there is a correponding micro electrode, find the channel position
    if micro_elname_i:
        micro_channelind_i = np.where(np.array(micro_elname_all) == micro_elname_i)[0]
    else:
        micro_channelind_i = []
    micro_channelind.append(list(micro_channelind_i))

df['time-micro'] = df['time'] - time_offset_macromicro
df['channelname-micro'] = micro_elname_corr
df['channelind-micro'] = micro_channelind
df['file-micro'] = micro_filename

excel_writer = pd.ExcelWriter(spreadsheet_path)
df.to_excel(excel_writer, 'Sheet ', index_label=None)
excel_writer.save()

