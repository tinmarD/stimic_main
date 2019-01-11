import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
sns.set()
sns.set_context('paper')


stim_spreadsheet_path = r'C:\Users\deudon\Desktop\Stimic\_Data\STIMS\P51_SC23\Spreadsheets\p51_SC53_stimulations_all.xlsx'
patient_id = re.split('\\\\', stim_spreadsheet_path)[-3]

df = pd.read_excel(stim_spreadsheet_path)

f = plt.figure()
ax1 = f.add_subplot(131)
sns.countplot(df['freq'], ax=ax1)
ax2 = f.add_subplot(132)
sns.countplot(df['intensity'], ax=ax2)
ax2.set(title=patient_id)
ax3 = f.add_subplot(133)
sns.countplot(y='channelname', data=df, orient='h', ax=ax3)


f = plt.figure()
ax = f.add_subplot(111)
ax.set(title=patient_id)
sns.countplot(y='channelname_bipolar', hue='freq', data=df, orient='h', ax=ax)




