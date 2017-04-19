"""Process electrophysiological recordings of fish behavior and trial structure"""


def chop_trials(signal, thr=2000):
    """for each unique value in the signal, 
       return the start and stop of each epoch corresponding to that value
    """
    from numpy import unique, where, concatenate, diff

    conditions = unique(signal)
    chopped = {}
    for c in conditions:
        tmp = where(signal == c)[0]
        offs = where(diff(tmp) > 1)[0]
        offs = concatenate((offs, [tmp.size-1]))
        ons = concatenate(([0], offs[0:-1] + 1))
        trLens = offs - ons        
        keep_trials = where(trLens > thr)
        offs = offs[keep_trials]
        ons = ons[keep_trials]
        chopped[c] = (tmp[ons], tmp[offs])
    
    return chopped


def estimate_onset(signal, threshold, duration):
    """
    Find indices in a vector when the values first cross a threshold. Useful when e.g. finding onset times for
    a ttl signal.


    Parameters
    ----------
    signal : numpy array, 1-dimensional
        Vector of values to be processed.

    threshold : instance of numeric data type contained in signal
        Onsets are counted as indices where the signal first crosses this value.

    duration : instance of numeric data type contained in signal
        Minimum distance between consecutive onsets.

    """

    from numpy import where, diff, concatenate

    # find indices where the signal is above threshold
    inits = where(signal > threshold)[0]

    # find indices where the threshold crossings are non-consecutive
    init_diffs = where(diff(inits) > 1)[0] + 1

    # add a 0 to count the first threshold crossing
    init_diffs = concatenate(([0], init_diffs))

    # index the threshold crossing indices with the non-consecutive indices
    inits = inits[init_diffs]

    keepers = concatenate((where(diff(inits) > duration)[0] + 1, [inits.size-1]))
    inits = inits[keepers]
    
    return inits


def estimate_swims(power, fs=6000):
    """ Estimate swim timing from ephys recording of motor neurons

    Parameters
    __________

    signal : numpy array, 1 dimensional

    fs : int
        sampling rate of the data

    """

    from numpy import zeros, where, diff, concatenate

    # set dead time between peaks, in seconds
    dead_time = .010 * fs

    # set minimum distance between swim bursts in seconds
    inter_swim_min = .12 * fs

    # estimate swim threshold
    thr = estimate_threshold(power, fs * 60)

    peaksT, peaksIndT = estimate_peaks(power, dead_time)

    burstIndT = peaksIndT[where(power[peaksIndT] > thr[peaksIndT])]
    burstT = zeros(power.shape)
    burstT[burstIndT] = 1

    interSwims = diff(burstIndT)
    swimEndIndB = where(interSwims > inter_swim_min)[0]
    swimEndIndB = concatenate((swimEndIndB, [burstIndT.size-1]))

    swimStartIndB = swimEndIndB[:-1] + 1
    swimStartIndB = concatenate(([0], swimStartIndB))
    nonShort = where(swimEndIndB != swimStartIndB)[0]
    swimStartIndB = swimStartIndB[nonShort]
    swimEndIndB = swimEndIndB[nonShort]

    bursts = zeros(power.size)
    starts = zeros(power.size)
    stops = zeros(power.size)
    bursts[burstIndT] = 1
    starts[burstIndT[swimStartIndB]] = 1
    stops[burstIndT[swimEndIndB]] = 1

    return starts, stops, thr


def windowed_variance(signal, kern_mean=None, kern_var=None, fs=6000):
    """
    Estimate smoothed sliding variance of the input signal

    signal : numpy array

    kern_mean : numpy array
        kernel to use for estimating baseline

    kern_var : numpy array
        kernel to use for estimating variance

    fs : int
        sampling rate of the data
    """
    from scipy.signal import gaussian, fftconvolve

    # set the width of the kernels to use for smoothing
    kw = int(.04 * fs)

    if kern_mean is None:
        kern_mean = gaussian(kw, kw // 10)
        kern_mean /= kern_mean.sum()

    if kern_var is None:
        kern_var = gaussian(kw, kw // 10)
        kern_var /= kern_var.sum()

    mean_estimate = fftconvolve(signal, kern_mean, 'same')
    var_estimate = (signal - mean_estimate) ** 2
    fltch = fftconvolve(var_estimate, kern_var, 'same')

    return fltch, var_estimate, mean_estimate


def estimate_peaks(signal, dead_time):
    """
    Estimate peak times in a signal, with a minimum distance between estimated peaks.

    Parameters
    __________

    signal : numpy array, 1-dimensional

    dead_time : int
        minimum number of sample between estimated peaks

    """

    from numpy import diff, where, zeros

    aa = diff(signal)
    peaks = (aa[:-1] > 0) * (aa[1:] < 0)
    inds = where(peaks)[0]

    # take the difference between consecutive indices
    d_inds = diff(inds)
                    
    # find differences greater than deadtime
    to_keep = (d_inds > dead_time)
    
    # only keep the indices corresponding to differences greater than deadT 
    inds[1:] = inds[1:] * to_keep
    inds = inds[inds.nonzero()]
    
    peaks = zeros(signal.shape[0])
    peaks[inds] = 1
    
    return peaks, inds


def load(in_file, num_channels=10):
    """Load multichannel binary data from disk, return as a [channels,samples] sized numpy array
    """
    from numpy import fromfile, float32

    fd = open(in_file, 'rb')
    data = fromfile(file=fd, dtype=float32)
    trim = data.size % num_channels
    # transpose to make dimensions [channels, time]
    data = data[:(data.size - trim)].reshape(data.size // num_channels, num_channels).T
    if trim > 0:
        print('Data needed to be truncated!')

    return data


def estimate_threshold(signal, window=180000, scaling=1.6, lower_percentile=.01):
    """
    Return non-sliding windowed threshold of input ndarray vec.

    Parameters
    ----------

    signal : ndarray, 1-dimensional
        Input array from which to estimate a threshold

    window : int
        Step size / window length for the resulting threshold.

    scaling : float or int
        scaling factor applied to estimated spread of the noise distribution of vec. Sets magnitude of threshold
        relative to the estimated upper bound of the noise distribution.

    lower_percentile : float
        Percentile of signal to use when estimating the lower bound of the noise distribution.
    """
    from numpy import zeros, percentile, arange, median

    th = zeros(signal.shape)
    for t in arange(0, signal.size - window, window):
        plr = arange(t, min(t + window, signal.size))
        sig = signal[plr]
        med = median(sig)
        bottom = percentile(sig, lower_percentile)
        th[t:] = (med + scaling * (med - bottom))

    return th
