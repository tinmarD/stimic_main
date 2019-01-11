import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils_eeg_cleaning
from utils_stimic import *
from stim_analysis_fun import *
sns.set()
sns.set_context('paper')


stim_epoch_path = r'C:\\Users\\deudon\\Desktop\\Stimic\\_Data\\STIMS\\P53_CD25\\P53_CD25_stim_mne_epoch-resync-epo.fif'
stim_spreadsheet_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P53_CD25\p53_CD25_stimulations_all.xlsx'

# Read mne epoch structure
stim_epoch = mne.read_epochs(stim_epoch_path)
print(stim_epoch)

# Load stim spreadsheet
df = pd.read_excel(stim_spreadsheet_path)


# df_sel = df[(df.freq==50) & (df.intensity==1.2)]
# plot_mean_artefact(stim_epoch, df, color_by='freq')
plot_mean_artefact(stim_epoch, df, color_by='intensity', freq=1)
# plot_mean_artefact(stim_epoch, df, freq=50, resync=False)

# plot_mean_artefact(stim_epoch, df, plot_traces=0, color_by='intensity', freq=1)
# plot_mean_artefact(stim_epoch, df, plot_traces=1, freq=1, intensity=2)
# plot_mean_artefact(stim_epoch, df, plot_traces=1, freq=1, intensity=3)
# plot_mean_artefact(stim_epoch, df, plot_traces=1, freq=50, intensity=1.2)
# plot_mean_artefact(stim_epoch, df, plot_traces=1, freq=50, intensity=1.4)
