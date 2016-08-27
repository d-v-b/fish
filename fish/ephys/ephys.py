"""Process electrophysiological recordings of fish behavior and trial structure"""


def chopTrials(signal, trialThr=2000):
    """for each unique value in the signal, 
       return the start and stop of each epoch corresponding to that value
    """
    from numpy import unique, where, concatenate

    conditions = unique(signal)
    chopped = {}
    for c in conditions:
        tmp = where(signal == c)[0]
        offs = where(np.diff(tmp) > 1)[0]
        offs = concatenate((offs, [tmp.size-1]))
        ons = concatenate(([0], offs[0:-1] + 1))
        trLens = offs - ons        
        keepTrials = where(trLens > trialThr)
        offs = offs[keepTrials]
        ons = ons[keepTrials]
        chopped[c] = (tmp[ons], tmp[offs])
    
    return chopped


def stack_inits(signal, thr=3.8, dur=10):
    """
        Find indices in ephys time corresponding to the onset of each stack in image time
    """

    from numpy import where, diff, concatenate

    # find indices where the signal is above threshold
    inits = where(signal > thr)[0]

    # find indices where the threshold crossings are non-consecutive
    init_diffs = where(diff(inits) > 1)[0]

    # add a 0 to count the first threshold crossing
    init_diffs = concatenate(([0], init_diffs+1))

    # index the threshold crossing indices with the non-consecutive indices
    inits = inits[init_diffs]

    keepers = concatenate((where(diff(inits) > dur)[0], [inits.size-1]))
    inits = inits[keepers]
    
    return inits


def getSwims(ch):
    """ Estimate swim timing from ephys recording of motor neurons
    """
    
    from numpy import zeros, where, diff, concatenate

    # set dead time, in samples
    deadT = 80
    
    fltch = windowed_variance(ch)
    peaksT, peaksIndT = getPeaks(fltch, deadT)
    thr = getThreshold(fltch, 2600000)
    burstIndT = peaksIndT[where(fltch[peaksIndT] > thr[peaksIndT])]
    burstT = zeros(fltch.shape)
    burstT[burstIndT] = 1
    
    interSwims = diff(burstIndT)
    swimEndIndB = where(interSwims > 800)[0]
    swimEndIndB = concatenate((swimEndIndB, [burstIndT.size-1]))

    swimStartIndB = swimEndIndB[0:-1] + 1
    swimStartIndB = concatenate(([0], swimStartIndB))
    nonShort = where(swimEndIndB != swimStartIndB)[0]
    swimStartIndB = swimStartIndB[nonShort]
    swimEndIndB = swimEndIndB[nonShort]

    bursts = zeros(fltch.size)
    starts = zeros(fltch.size)
    stops = zeros(fltch.size)
    bursts[burstIndT] = 1
    starts[burstIndT[swimStartIndB]] = 1
    stops[burstIndT[swimEndIndB]] = 1
    
    return starts, stops, thr


# filter signal, extract power
def windowed_variance(signal, kern_mean=None, kern_var=None):
    """
    Estimate sliding variance of the input signal
    :param signal:
    :param kern_mean:
    :return kern_var:
    """
    from scipy.signal import gaussian, fftconvolve

    if kern_mean is None:
        kern_mean = gaussian(121, 20)
        kern_mean /= kern_mean.sum()

    if kern_var is None:
        kern_var = gaussian(121, 20)
        kern_var /= kern_var.sum()

    mean_estimate = fftconvolve(signal, kern_mean, 'same')
    power = (signal - mean_estimate)**2
    fltch = fftconvolve(power, kern_var, 'same')
    return fltch


# get peaks
def getPeaks(fltch, deadTime=80):
    
    aa = np.diff(fltch)
    peaks = (aa[0:-1] > 0) * (aa[1:] < 0)
    inds = np.where(peaks)[0]    

    # take the difference between consecutive indices
    dInds = np.diff(inds)
                    
    # find differences greater than deadtime
    toKeep = (dInds > deadTime)    
    
    # only keep the indices corresponding to differences greater than deadT 
    inds[1::] = inds[1::] * toKeep
    inds = inds[inds.nonzero()]
    
    peaks = np.zeros(fltch.size)
    peaks[inds] = 1
    
    return peaks, inds


# find threshold
def getThreshold(fltch, wind=180000, shiftScale=1.6):
    
    th = np.zeros(fltch.shape)
    
    for t in np.arange(0, fltch.size-wind, wind):

        interval = np.arange(t, t+wind)
        sqrFltch = fltch ** .5            
        hist, bins = np.histogram(sqrFltch[interval], 1000)
        mx = np.min(np.where(hist == np.max(hist)))
        mn = np.max(np.where(hist[0:mx] < hist[mx]/200.0))        
        th[t:] = (bins[mx] + shiftScale * (bins[mx] - bins[mn]))**2.0

    return th


def load(in_file):
    """Load 10chFlt data from disk, return as a [channels,samples] sized numpy array
    """
    from numpy import fromfile, float32

    fd = open(in_file, 'rb')
    data = fromfile(file=fd, dtype=float32)
    n_chan = 10
    trim = len(data) % n_chan
    # transpose to make dimensions [channels, time]
    data = data[:(data.size - trim)].reshape(data.size // n_chan, n_chan).T
    if trim > 0:
        print('Data needed to be truncated!')

    return data
