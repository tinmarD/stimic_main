import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils_eeg_cleaning
sns.set()
sns.set_context('paper')


def resync_traces_2d(stim_traces, time, threshold_init=3000, threshold_dec=200, threshold_min=1000, jitter_max_abs=0.1,
                     inverse_neg_stim=True):
    """ Synchronize the stimulation traces based on the first point where the absolute value of the trace reaches the
     threshold. The threshold is first set at threshold_init and if no points reaches it within the time interval around
     0 (defined by jitter_max_abs), the threshold is decremented by threshold_dec, until a threshold is found or
     threshold_min is reached.
     If inverse_neg_stim is set to True, the stimulation traces that start by a negative phase are turned upward. This
     is needed when averaging traces.

    Parameters
    ----------
    stim_traces : 2D array (n_stim, n_pnts)
        Stimulation data. Must be the stimulation channel.
    time : 1D array (n_pnts)
        Time vector (s)
    threshold_init : float
        Initial stimulation threshold (uV) - Default : 3000
    threshold_dec : float
        Decrement for the stimulation threshold (uV) - Default : 200
    threshold_min : float
        Minimum stimulation threshold (uV) - Default : 1000
    jitter_max_abs : float
        Maximum absolute jitter (s)
    inverse_neg_stim : bool
        If true, inverse the negative stimulation traces - Default (True)

    Returns
    -------
    sync_traces : 2D array (n_stim, n_pnts)
        Synchronized stimulation data
    offset_samples : 1D array (n_stim)
        Offset in sample for each stim

    """
    n_stim, n_pnts = stim_traces.shape
    fs, t_min = int(1/(time[1] - time[0])), time[0]
    zero_pnt = int(fs*abs(t_min))
    if not time.size == n_pnts:
        raise ValueError('stim_traces and time shapes do not match')
    threshold_init, jitter_max_abs = abs(threshold_init), abs(jitter_max_abs)
    sync_traces, offset_samples = np.zeros((n_stim, n_pnts)), np.zeros(n_stim, dtype=int)
    for i in range(n_stim):
        thresh_reached, threshold = False, threshold_init
        # Find the point where the threshold is reached
        while not thresh_reached:
            thresh_point_pos = np.where(stim_traces[i] > threshold)[0]
            thresh_point_neg = np.where(stim_traces[i] < -threshold)[0]
            # If both negative and positive threshold points are within range, select the one closest to 0 :
            if thresh_point_pos.size > 0 and -jitter_max_abs < time[thresh_point_pos[0]] < jitter_max_abs and \
               thresh_point_neg.size > 0 and -jitter_max_abs < time[thresh_point_neg[0]] < jitter_max_abs:
                if abs(time[thresh_point_pos[0]]) < abs(time[thresh_point_neg[0]]):
                    thresh_pnt, thresh_reached = thresh_point_pos[0], True
                else:
                    thresh_pnt, thresh_reached = thresh_point_neg[0], True
            # If only positive threshold is reached within time interval :
            elif thresh_point_pos.size > 0 and -jitter_max_abs < time[thresh_point_pos[0]] < jitter_max_abs:
                thresh_pnt, thresh_reached = thresh_point_pos[0], True
            # If only negative threshold is reached within time interval :
            elif thresh_point_neg.size > 0 and -jitter_max_abs < time[thresh_point_neg[0]] < jitter_max_abs:
                thresh_pnt, thresh_reached = thresh_point_neg[0], True
            else:
                if threshold < threshold_min:
                    break
                else:
                    threshold -= threshold_dec
        if not thresh_reached:
            print('Warning: Could not detect any point above threhsold')
            sync_traces[i, :] = stim_traces[i, :]
        else:
            # Re-align the trace
            offest_new_ori = thresh_pnt - zero_pnt
            # Threshold reached before time zero point, add points before, remove last points
            if offest_new_ori < 0:
                sync_traces[i, :] = np.hstack([np.ones(abs(offest_new_ori)) * stim_traces[i, 0],
                                              stim_traces[i, :offest_new_ori]])
            # Threshold reached after time zero point, remove first points, add points at the end
            elif offest_new_ori > 0:
                sync_traces[i, :] = np.hstack([stim_traces[i, offest_new_ori:],
                                               np.ones(abs(offest_new_ori)) * stim_traces[i, -offest_new_ori]])
            else:
                sync_traces[i, :] = stim_traces[i, :]
            offset_samples[i] = offest_new_ori
        # Turn the trace upward
        if inverse_neg_stim:
            half_pat_size = int(fs*0.05)
            pos_stim_pat = np.hstack([np.zeros(half_pat_size), threshold_init*np.ones(half_pat_size)])
            corr_coeff = np.corrcoef(pos_stim_pat, sync_traces[i, zero_pnt-half_pat_size:zero_pnt+half_pat_size])[0,1]
            sync_traces[i, :] = sync_traces[i, :] if corr_coeff > 0 else -sync_traces[i, :]
    return sync_traces, offset_samples


def resync_traces_3d(epoch_data, time, df, threshold_init=3000, threshold_dec=200, threshold_min=1000,
                     jitter_max_abs=0.1, inverse_neg_stim=True):
    n_stim, n_chan, n_pnts = epoch_data.shape
    epoch_data_sync = np.zeros((n_stim, n_chan, n_pnts))
    stim_data = np.zeros((n_stim, n_pnts))
    for i, chan_pos_i in enumerate(df.channelind):
        stim_data[i, :] = epoch_data[i, chan_pos_i]
    stim_data_sync, offset_samples = resync_traces_2d(stim_data, time, threshold_init, threshold_dec, threshold_min,
                                                      jitter_max_abs, inverse_neg_stim=True)
    for i in range(n_stim):
        if offset_samples[i] == 0:
            epoch_data_sync[i, :, :] = epoch_data[i, :, :]
        elif offset_samples[i] < 0:  # add points before, remove last points
            first_val = np.tile(np.atleast_2d(epoch_data[i, :, 0]).T, -offset_samples[i])
            epoch_data_sync[i, :, :] = np.hstack([first_val, epoch_data[i, :, :offset_samples[i]]])
        else:  # remove first points, add points at the end
            last_val = np.tile(np.atleast_2d(epoch_data[i, :, -1]).T, offset_samples[i])
            epoch_data_sync[i, :, :] = np.hstack([epoch_data[i, :, offset_samples[i]:], last_val])
        if inverse_neg_stim:
            epoch_data_sync[i, df.channelind[i], :] = stim_data_sync[i]

    return epoch_data_sync


