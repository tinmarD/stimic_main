"""
    STIMIC Project -  Stim Event List Creation
                Utils function

See the doc here : https://tinmard.github.io/stimic-stimulation-listing.html

"""
import numpy as np
import re


def get_stim_bipolar_channel(stim_times, stim_chanind, ch_names):
    """ Return the bipolar channel of stimulation given the monopolar channel and the times of each stimulation.
    The stim must occurs at the same time, more or less, and occurs on adjacent channels.

    Parameters
    ----------
    stim_times : array (float)
        time of each stim (s)
    stim_chanind : array (int)
        channel index of each stim in monopolar data
    ch_names : array (str)
        channel names of all channels, i.e. stim_channame = ch_names[stim_chanind[i]]

    Returns
    -------
    stim_channame_bi : array(str)
        bipolar channel name of each stim
    """
    time_limit = 0.5
    n_stim = stim_times.size
    stim_channame_bi = np.zeros(n_stim, dtype=object)
    for i in range(n_stim):
        if i == 0:
            if abs(stim_times[i+1]-stim_times[i]) < time_limit and abs(stim_chanind[i+1] - stim_chanind[i]) == 1:
                ch_name_a, ch_name_b = ch_names[stim_chanind[i]], ch_names[stim_chanind[i+1]]
                ch_num_a = int(re.search('\d+', ch_name_a)[0])
                ch_num_b = int(re.search('\d+', ch_name_b)[0])
                if ch_num_a < ch_num_b:
                    stim_channame_bi[i] = '{}-{}'.format(ch_name_a, ch_name_b)
                else:
                    stim_channame_bi[i] = '{}-{}'.format(ch_name_b, ch_name_a)
        elif i == n_stim-1:
            if abs(stim_times[i]-stim_times[i-1]) < time_limit and abs(stim_chanind[i] - stim_chanind[i-1]) == 1:
                ch_name_a, ch_name_b = ch_names[stim_chanind[i-1]], ch_names[stim_chanind[i]]
                ch_num_a = int(re.search('\d+', ch_name_a)[0])
                ch_num_b = int(re.search('\d+', ch_name_b)[0])
                if ch_num_a < ch_num_b:
                    stim_channame_bi[i] = '{}-{}'.format(ch_name_a, ch_name_b)
                else:
                    stim_channame_bi[i] = '{}-{}'.format(ch_name_b, ch_name_a)
        else:
            if abs(stim_times[i+1]-stim_times[i]) < time_limit and abs(stim_chanind[i+1] - stim_chanind[i]) == 1:
                ch_name_a, ch_name_b = ch_names[stim_chanind[i]], ch_names[stim_chanind[i+1]]
                ch_num_a = int(re.search('\d+', ch_name_a)[0])
                ch_num_b = int(re.search('\d+', ch_name_b)[0])
                if ch_num_a < ch_num_b:
                    stim_channame_bi[i] = '{}-{}'.format(ch_name_a, ch_name_b)
                else:
                    stim_channame_bi[i] = '{}-{}'.format(ch_name_b, ch_name_a)
            elif abs(stim_times[i]-stim_times[i-1]) < time_limit and abs(stim_chanind[i] - stim_chanind[i-1]) == 1:
                ch_name_a, ch_name_b = ch_names[stim_chanind[i-1]], ch_names[stim_chanind[i]]
                ch_num_a = int(re.search('\d+', ch_name_a)[0])
                ch_num_b = int(re.search('\d+', ch_name_b)[0])
                if ch_num_a < ch_num_b:
                    stim_channame_bi[i] = '{}-{}'.format(ch_name_a, ch_name_b)
                else:
                    stim_channame_bi[i] = '{}-{}'.format(ch_name_b, ch_name_a)
    return stim_channame_bi

