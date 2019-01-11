"""
    STIMIC Project - Stim Event List Creation - step 2
                Stim Detection Functions

See the doc here : https://tinmard.github.io/stimic-stimulation-listing.html

"""
import numpy as np
from scipy import signal, stats


def stim_threshold_crossing_points(x, fs, threshold, merge_dur_s=[], take_abs_value=False):
    """ Thresold x or abs(x) if take_abs_value is True, and get the threshold crossing points. If merge_dur_s is
    defined, delete threshold crossing points if a previous one occured less than merge_dur_s seconds ago.
    If threshold is negative select points inferior to the threshold

    Parameters
    ----------
    x : array (1D)
        Input signal
    fs : float
        Sampling frequency
    threshold : float
        Threshold
    merge_dur_s : float
        Events closer than this duration will be merged (i.e. : the second will be deleted)
    take_abs_value : bool (default: False)
        If True, the threshold is applied on the absolute value of the input signal x

    Returns
    -------
    thresh_crossed_times : array
        Times in seconds where the signal is above (below) the threshold if the threshold is positive (negative)

    thresh_crossed_pnts : array
        Indices in samples where the signal is above (below) the threshold if the threshold is positive (negative)

    """
    if take_abs_value:
        x = np.abs(x)
    thresh_crossed_pnts = np.where(x > threshold)[0] if threshold >= 0 else np.where(x < threshold)[0]
    if thresh_crossed_pnts.size > 0 and merge_dur_s:
        merge_dur_n = int(fs*merge_dur_s)
        thresh_crossed_pnts = thresh_crossed_pnts[np.hstack([True, np.array(thresh_crossed_pnts[1:] - thresh_crossed_pnts[:-1]) > merge_dur_n])]
    thresh_crossed_times = thresh_crossed_pnts / fs
    return thresh_crossed_times, thresh_crossed_pnts


def stim_get_threshold_peaks(x, fs, threshold, merge_peak_dur_s=[]):
    """ Count the number of peaks in the thresholded signal : trace > thresh. If merge_peak_s is defined, peaks are
    deleted if a previous one occured less than meark_peak_dur_s seconds ago.
    If threshold is negative select points inferior to the threshold"""
    if threshold >= 0:
        peaks_pnts = np.where(np.diff(x > threshold) & (x[1:] > threshold))[0]
    else:
        peaks_pnts = np.where(np.diff(x < threshold) & (x[1:] < threshold))[0]
    merge_peak_dur_n = int(fs*merge_peak_dur_s)
    # Delete peak if the previous one occured less than merge_peak_dur_n samples ago
    peaks_pnts = peaks_pnts[np.hstack([True, np.diff(peaks_pnts) > merge_peak_dur_n])]
    peaks_times = peaks_pnts / fs
    return peaks_times, peaks_pnts


def stim_50hz_correlation_with_mean_artifact(x, mean_artifact, duration_s=5, srate_mean=512, fcutoff=1, forder=6):
    # low pass filter the data
    t_vect_mean = np.linspace(0, duration_s, mean_artifact.size)
    t_vect_x = np.linspace(0, duration_s, x.size)
    if not x.size == mean_artifact.size:
        x = np.interp(t_vect_mean, t_vect_x, x)
    b, a = signal.butter(forder, 2 * fcutoff / srate_mean, 'lowpass')
    x_smoothed = signal.filtfilt(b, a, x)
    x_smoothed = x_smoothed if x_smoothed[0] >= 0 else -x_smoothed
    corr_spearman = stats.spearmanr(x_smoothed, mean_artifact)[0]
    return corr_spearman


def stim_50Hz_get_zero_crossing(x, srate, fcutoff=1, forder=6):
    b, a = signal.butter(forder, 2 * fcutoff / srate, 'lowpass')
    x_smoothed = signal.filtfilt(b, a, x)
    zc_pnts = np.where(np.diff(np.sign(x_smoothed)))[0]
    zc_times = zc_pnts / srate
    return zc_times, zc_pnts

