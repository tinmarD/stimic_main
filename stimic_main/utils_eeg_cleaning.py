import numpy as np
import re


def get_clean_eeg_channelnames(ch_names, wrong_channel_pattern=[]):
    """ From a list of channel names returns the channel names containing 'EEG' and remove the channel names
    containing '...'. Also return the indices of the correct EEG channels
    """
    wrong_channel_pattern = np.atleast_1d(wrong_channel_pattern)
    ch_names = np.array(ch_names)
    ch_eegclean_pos = np.where([('EEG' in ch_name_i) & ('...' not in ch_name_i) & ('EMG' not in ch_name_i)
                                for ch_name_i in ch_names])[0]
    ch_eegclean_names = ch_names[ch_eegclean_pos]
    # If wrong_channel_patterns are defined, detect wrong channels
    if wrong_channel_pattern.size > 0:
        wrong_channel_pos_all = np.zeros(ch_eegclean_names.size, dtype=bool)
        for pattern_i in wrong_channel_pattern:
            try:
                wrong_channel_pos_i = np.array([1 if re.search(pattern_i, channame_i, re.IGNORECASE) else 0
                                                for channame_i in ch_eegclean_names]).astype(bool)
            except:
                print('Could not detect wrong channel names for regex pattern : {}'.format(pattern_i))
                wrong_channel_pos_i = np.zeros(ch_eegclean_names.size, dtype=bool)
            wrong_channel_pos_all = wrong_channel_pos_all | wrong_channel_pos_i
    ch_eegclean_names = ch_eegclean_names[~wrong_channel_pos_all]

    return ch_eegclean_names, ch_eegclean_pos


def get_electrode_info(ch_names, montage=''):
    """  From a list of channel names return the electrode name, the electrode number and the channel number.

    Parameters
    ----------
    ch_names : list, array
        List of channel names
    montage : str (default '')
        Montage of EEG channels, must be 'mono', 'monopolar' for monopolar montage, or 'bi', 'bipolar' for bipolar
        montage, if not provided will try to detect automatically the montage

    Returns
    -------
    el_names : array
        Electrode name for each channel
    el_num : array
        Electrode unique number for each channel
    ch_num : array (n_chan) or (n_chan, 2)
        Channel number
    """
    ch_names = np.atleast_1d(ch_names)
    n_chan = ch_names.size
    el_names, unique_el_names = [], []
    el_num = np.zeros(n_chan, dtype=int)
    if not montage:
        # Try to detect montage with the first channel
        if len(re.findall('\d+', ch_names[0])) == 1:
            montage = 'mono'
        elif len(re.findall('\d+', ch_names[0])) == 2:
            montage = 'bi'
        else:
            raise ValueError('Could not detect montage, specify it in arguments')
    if montage in ['mono', 'monopolar']:
        ch_num = np.zeros(n_chan, dtype=int)
    elif montage in ['bi', 'bipolar']:
        ch_num = np.zeros((n_chan, 2), dtype=int)
    else:
        raise ValueError('Wrong montage argument : {}'.format(montage))
    rx_el = re.compile('[a-zA-Z\']+')
    rx_ch = re.compile('\d+')
    for i, ch_name_i in enumerate(ch_names):
        el_gps = rx_el.findall(ch_name_i)
        ch_gps = rx_ch.findall(ch_name_i)
        ch_gps = [int(c) for c in ch_gps]
        # Electrode name
        if not el_gps:
            print('Could not get electrode name of channel {}'.format(ch_name_i))
            el_name_i = ''
        elif el_gps[0].upper() == 'EEG' and len(el_gps) > 1:
            el_name_i = el_gps[1]
        elif el_gps[0].upper() == 'EEG' and len(el_gps) == 1:
            print('Could not get electrode name of EEG channel {}'.format(ch_name_i))
            el_name_i = ''
        else:
            el_name_i = el_gps[0]
        el_names.append(el_name_i)
        # Electrode number
        if el_name_i in unique_el_names:
            el_num[i] = unique_el_names.index(el_name_i)
        else:
            el_num[i] = len(unique_el_names)
            unique_el_names.append(el_name_i)
        # Channel numbers
        if montage in ['mono', 'monopolar']:
            if len(ch_gps) == 1:
                ch_num[i] = ch_gps[0]
            else:
                print('Could not determine channel number of channel {}'.format(ch_name_i))
                ch_num[i] = -1
        elif montage in ['bi', 'bipolar']:
            if len(ch_gps) == 2:
                ch_num[i] = ch_gps
            else:
                print('Could not determine channel number of channel {}'.format(ch_name_i))
                ch_num[i] = (-1, -1)
    return np.array(el_names), el_num, ch_num


def get_neighbour_channel(ch_names, chan_name_sel, offset):
    """ Given the list of channel names and, a selected channel name and an offset, returns the position and the name
    of the offset channel (a.k.a. the neighbour channel)
    For example, if chan_name_sel = "EEG CU2" and offset = 1, the function will return "EEG CU3" and its position.
    If the neighbour channel cannot be found, return -1 and ''

    Parameters
    ----------
    ch_names : array | list
        List of all channels' names
    chan_name_sel : str
        Selected channel name
    offset : int
        Offset from the selected channel

    Returns
    -------
    neighbour_chan_pos : int
        neighbour channel position in the list of channel
    neigbhour_chan_name : str
        neighbour channel name

    """
    neighbour_chan_pos, neighbour_chan_name = -1, ''
    n_chan = len(ch_names)
    if chan_name_sel not in ch_names:
        raise ValueError('{} is not on the channel list'.format(chan_name_sel))
    chan_pos = ch_names.index(chan_name_sel)
    el_name, el_num, ch_num = get_electrode_info(chan_name_sel)
    # MONOpolar montage
    if ch_num.size == 1:
        # Construct the supposed channel name with offset and see if it's on the channel list
        if 'EEG {}{}'.format(el_name[0], ch_num[0]+offset) in ch_names:
            neighbour_chan_name = 'EEG {}{}'.format(el_name[0], ch_num[0]+offset)
            neighbour_chan_pos = ch_names.index(neighbour_chan_name)
        # If not, look if the get the channel name from the channel list and see if it's on the same electrode
        else:
            if 0 <= chan_pos+offset < n_chan:
                el_name_off, el_num_off, ch_num_off = get_electrode_info(ch_names[chan_pos+offset])
                if el_name_off[0] == el_name[0]:
                    neighbour_chan_name = ch_names[chan_pos+offset]
                    neighbour_chan_pos = ch_names.index(neighbour_chan_name)

    # BIpolar montage
    elif ch_num.size == 2:
        # Construct the supposed channel name with offset and see if it's on the channel list
        if 'EEG {}{}-{}{}'.format(el_name[0], ch_num[0]+offset, el_name[0], ch_num[1]+offset) in ch_names:
            neighbour_chan_name = 'EEG {}{}-{}{}'.format(el_name[0], ch_num[0]+offset, el_name[0], ch_num[1]+offset)
            neighbour_chan_pos = ch_names.index(neighbour_chan_name)
        # If not, look if the get the channel name from the channel list and see if it's on the same electrode
        else:
            if 0 <= chan_pos + offset < n_chan:
                el_name_off, el_num_off, ch_num_off = get_electrode_info(ch_names[chan_pos + offset])
                if el_name_off[0] == el_name[0]:
                    neighbour_chan_name = ch_names[chan_pos + offset]
                    neighbour_chan_pos = ch_names.index(neighbour_chan_name)
    if neighbour_chan_pos == -1:
        print('Could not find the neighbour channel')

    return neighbour_chan_pos, neighbour_chan_name
