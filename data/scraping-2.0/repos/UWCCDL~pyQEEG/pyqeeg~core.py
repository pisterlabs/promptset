from collections import namedtuple
import numpy as np
from scipy.signal import coherence, find_peaks
from pyqeeg.utils import blink_ok, bounds_ok, detrend_data, qual_ok, get_bounds, longest_quality

SPIKE_CUTOFF = 200
Band = namedtuple("Band", ["name", "lower_bound", "upper_bound"])
Spectrum = namedtuple("Spectrum", ["good_samples", "power", "longest_quality_segment"])
Coherence = namedtuple("Coherence", ["good_samples", "coherence"])
IndividualizedAlphaFrequency = namedtuple("IndividualizedAlphaFrequency", ["power", "freq"])


def spectral_analysis(series, x, y, blink, quality, sampling=128, length=4, sliding=0.75, hamming=True):
    series = detrend_data(series, x, y)
    good_samples = 0
    shift = int(sampling * (length * sliding))
    window = int(sampling * length)
    spectrum_len = (sampling * length) // 2
    lower, upper = get_bounds(series)
    result = np.zeros(spectrum_len)

    for i in range(0, len(series) - window, shift):
        sub, bsub, qsub = series[i:i+window], blink[i:i+window], quality[i:i+window]
        maxmin = max(sub) - min(sub)
        if bounds_ok(sub, lower, upper) and blink_ok(bsub) and qual_ok(qsub) and maxmin < SPIKE_CUTOFF:
            good_samples += 1
            if hamming:
                sub = sub * np.hamming(window)
            partial = np.real(np.fft.fft(sub)) ** 2
            partial = partial[:spectrum_len]
            result = result + partial
    result = np.log(result / good_samples)

    return Spectrum(good_samples, result, longest_quality(quality, sampling))


def coherence_analysis(series1, series2, x, y, blink, quality1, quality2,
                       sampling=128, length=4, sliding=0.75, hamming=True):
    series1 = detrend_data(series1, x, y)
    series2 = detrend_data(series2, x, y)

    good_samples = 0
    shift = int(sampling * (length * sliding))
    window = int(sampling * length)
    spectrum_len = (sampling * length) // 2
    lower1, upper1 = get_bounds(series1)
    lower2, upper2 = get_bounds(series2)
    minlen = min(len(series1), len(series2))
    result = np.zeros(spectrum_len)

    for i in range(0, minlen - window, shift):
        sub1, sub2 = series1[i:i+window], series2[i:i+window]
        #print("sub1 = [" + ", ".join([str(v) for v in sub1]) + "]")
        #print("sub2 = [" + ", ".join([str(v) for v in sub2]) + "]")
        bsub, qsub1, qsub2 = blink[i:i+window], quality1[i:i+window], quality2[i:i+window]
        if bounds_ok(sub1, lower1, upper1) and bounds_ok(sub2, lower2, upper2) \
                and blink_ok(bsub) and qual_ok(qsub1) and qual_ok(qsub2):
            good_samples += 1
            if hamming:
                sub1 = sub1 * np.hamming(window)
                sub2 = sub2 * np.hamming(window)
            partial = coherence(sub1, sub2)[1][1:]
            result += partial
    result /= good_samples
    return Coherence(good_samples, result)


def find_iaf(power, freq, alpha_lower_bound=7, alpha_upper_bound=15):
    """
    Find individualized Alpha Frequency
    """
    max_peak_power, max_peak_freq = None, None
    alpha_power = power[(freq >= alpha_lower_bound) & (freq <= alpha_upper_bound)]
    alpha_freq = freq[(freq >= alpha_lower_bound) & (freq <= alpha_upper_bound)]
    peaks_coords, _ = find_peaks(alpha_power, prominence=0.2)
    if len(peaks_coords) > 0:
        max_peak_power = alpha_power[peaks_coords[0]]
        max_peak_freq = alpha_freq[peaks_coords[0]]
        for peak_coord in peaks_coords[1:]:
            if alpha_power[peak_coord] > max_peak_power:
                max_peak_power = alpha_power[peak_coord]
                max_peak_freq = alpha_freq[peak_coord]
    return IndividualizedAlphaFrequency(max_peak_power, max_peak_freq)


def draw_bands(band_method, whole_head_iaf=10):
    """
    Function to define the frequency bands according to the "band_method" argument.

    :param band_method: name of the definition method (IBIW or IBFW)
    :type band_method: str
    :param whole_head_iaf: whole head IAF (default 10)
    :type whole_head_iaf: float
    :return:
    """
    if band_method == "IBIW":
        return [Band("Delta", 0, whole_head_iaf * 0.4),
                Band("Theta", whole_head_iaf * 0.4, whole_head_iaf * 0.8),
                Band("Alpha", whole_head_iaf * 0.8, whole_head_iaf * 1.21),
                Band("Low_Beta", whole_head_iaf * 1.21, whole_head_iaf * 1.8),
                Band("High_Beta", whole_head_iaf * 1.8, whole_head_iaf * 3),
                Band("Gamma", whole_head_iaf * 3, 40.5)]
    elif band_method == "IBFW":
        return [Band("Delta", 0, whole_head_iaf - 6),
                Band("Theta", whole_head_iaf - 6, whole_head_iaf - 2),
                Band("Alpha", whole_head_iaf - 2, whole_head_iaf + 2.5),
                Band("Low_Beta", whole_head_iaf + 2.5, whole_head_iaf + 8),
                Band("High_Beta", whole_head_iaf + 8, whole_head_iaf + 20),
                Band("Gamma", whole_head_iaf + 20, 40.5)]
    elif band_method == "FBFW":
        return [Band("Delta", 0, 4),
                Band("Theta", 4, 8),
                Band("Alpha", 8, 12.5),
                Band("Low_Beta", 12.5, 18),
                Band("High_Beta", 18, 30),
                Band("Gamma", 30, 40.5)]
    else:
        print("Invalid band_method")
