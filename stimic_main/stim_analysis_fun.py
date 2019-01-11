"""
    STIMIC Project - Analysis Functions

Functions for analyzing the stimulation artifacts

See the doc here : https://tinmard.github.io/stimic/index.html
"""

import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sns.set()
sns.set_context('paper')


def plot_artefact_ev_with_electrode_distance(epoch, df, **kwargs):
    df_sel = df
    if df['ID'].unique().size > 1:
        raise ValueError('Function designed for single patient analysis')
    for key, value in kwargs.items():
        df_sel = df_sel[df_sel[key] == value]
    epoch_data = epoch.get_data()[df_sel.index, :, :]
    n_stim, n_chan, n_pnts = epoch_data.shape
    stim_data_offset = {}
    chan_offset_all = np.arange(-3, 4)
    for chan_offset in chan_offset_all:
        stim_data = np.nan * np.ones((n_stim, n_pnts))
        for i, chan_name_i in enumerate(df_sel.channelname):
            chan_pos_offset_i, _ = utils_eeg_cleaning.get_neighbour_channel(epoch.ch_names, chan_name_i, chan_offset)
            if chan_pos_offset_i != -1:
                stim_data[i, :] = epoch_data[i, chan_pos_offset_i, :]
        stim_data_offset[chan_offset] = np.nanmean(stim_data, axis=0)
    f = plt.figure()
    ax = f.add_subplot(111)
    for chan_offset in chan_offset_all:
        lw = 3 if chan_offset == 0 else 1
        ax.plot(epoch.times, stim_data_offset[chan_offset], lw=lw)
    ax.legend(chan_offset_all)
    title_str = df['ID'][0]
    for key, value in kwargs.items():
        title_str += ' - {} = {}'.format(key, value)
    ax.set(xlabel='Time (ms)', ylabel='Amplitude (uV)', title=title_str)
    ax.autoscale(axis='x', tight=True)


def plot_mean_artefact(epoch, df, plot_traces=0, color_by=[], resync=False, **kwargs):
    df_sel = df
    if df['ID'].unique().size > 1:
        raise ValueError('Function designed for single patient analysis')
    for key, value in kwargs.items():
        df_sel = df_sel[df_sel[key] == value]
    epoch_data = epoch.get_data()[df_sel.index, :, :]
    n_trial, n_chan, n_pnts = epoch_data.shape
    stim_data = np.zeros((n_trial, n_pnts))
    for i, chan_pos_i in enumerate(df_sel.channelind):
        stim_data[i, :] = epoch_data[i, chan_pos_i]
    if resync:
        stim_data, _ = resync_traces_2d(stim_data, epoch.times)
    f = plt.figure()
    ax = f.add_subplot(111)
    if type(color_by) == str:
        cat_unique = np.unique(df_sel[color_by])
        n_colors = cat_unique.size
        if n_colors > 5:
            colors = sns.color_palette("cubehelix", n_colors=n_colors)
        else:
            colors = sns.color_palette(n_colors=n_colors)
        legend_str = []
        for i, cat_i in enumerate(cat_unique):
            if plot_traces:
                stim_data_i = stim_data[df_sel[color_by] == cat_i]
                alpha_val = 0.7 if n_trial < 10 else 0.2
                ax.plot(np.tile(epoch.times, (stim_data_i.shape[0], 1)), stim_data_i, color=colors[i], alpha=alpha_val)
                legend_str.append(cat_i)
            else:
                ax.plot(epoch.times, np.nanmean(stim_data[df_sel[color_by] == cat_i], axis=0), color=colors[i])
                legend_str.append('{} (N={})'.format(cat_i, sum(df_sel[color_by] == cat_i)))
        ax.legend(legend_str)
    else:
        if plot_traces:
            alpha_val = 0.7 if n_trial < 10 else 0.2
            ax.plot(np.tile(epoch.times, (n_trial, 1)).T, stim_data.T, alpha=alpha_val)
        else:
            ax.plot(epoch.times, np.nanmean(stim_data, axis=0))
    title_str = df['ID'][0]
    for key, value in kwargs.items():
        title_str += ' - {} = {}'.format(key, value)
    ax.set(xlabel='Time (ms)', ylabel='Amplitude (uV)', title=title_str)
    ax.autoscale(axis='x', tight=True)
