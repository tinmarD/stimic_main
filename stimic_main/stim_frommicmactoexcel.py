"""
    STIMIC Project - Stim Event List Creation - step 4
                micMac file to Excel

This script takes as input the micMac csv file containing all the stimulation events (after visual check of the events
detected in step 2) and convert it into an excel file.

See the doc here : https://tinmard.github.io/stimic-stimulation-listing.html

"""

import pandas as pd
import numpy as np
import re
import os
import utils_eeg_io
import stim_spreadsheet_utils

## Change the following parameters lines :
micmac_stimevent_filepath = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P56_AB28\data_clean\AB28-stim4_clean_raw-1_mm.csv'
stim_filename = r'AB28-stim4_clean_raw-1'
patient_id = r'P56_AB28'
data_dir = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P56_AB28\data_clean'


################################################
df_in = pd.read_csv(micmac_stimevent_filepath)
df_out = df_in.copy()
df_out = df_out.drop(['id', 'sigid', 'rawparentid', 'centerfreq', 'color', 'tpos'], axis=1)
df_out['ID'] = patient_id
df_out['file'] = stim_filename
df_out['time'] = df_in['tpos']
df_out['freq'] = 1
df_out['intensity'] = 0
df_out['effect'] = 'RAS'
df_out['duration'] = np.round(df_out['duration'])
# Stim frequency
freq = np.zeros(df_out.shape[0], dtype=int)
for i in range(df_out.shape[0]):
    try:
        freq[i] = int(re.search('\d+', df_out['type'][i])[0])
    except:
        print('Could not detect frequency from event type')
        freq[i] = -1
df_out['freq'] = freq
# Bipolar channel - need to load data
# Get monopolar channel names
try:
    _, ch_names = utils_eeg_io.io_eeg_to_mne(os.path.join(data_dir, stim_filename), False)
except:
    _, ch_names = utils_eeg_io.io_eeg_to_mne(os.path.join(data_dir, '{}.fif'.format(stim_filename)), False)
df_out['channelname_bipolar'] = stim_spreadsheet_utils.get_stim_bipolar_channel(df_out['time'], df_out['channelind']-1, ch_names)

# Write excel sheet
excel_stimevent_filepath = re.sub('.csv$', '.xlsx', micmac_stimevent_filepath)
excel_writer = pd.ExcelWriter(excel_stimevent_filepath)
df_out.to_excel(excel_writer, 'Sheet ', index_label=None, columns=['ID', 'file', 'type', 'time', 'channelind', 'channelname',
                                                                   'channelname_bipolar', 'freq', 'intensity', 'duration', 'effect'])
excel_writer.save()

