"""
Collection of functions and routines to compute vertical coherence profiles.

Content:

vertical_coherence_comp: driver script for vertical coherence calculations

coher_sig_dist: compute significant value of coherence from coherence distribution

cross_phase_2d: Compute phase angled from averaged cross-spectral components

"""
import numpy as np
import xarray as xr
from tropical_diagnostics import spacetime as st

def vertical_coherence_comp(data1, data2, levels, nDayWin, nDaySkip, spd, siglevel):
    """
     Main driver to compute vertical coherence profile. This can be called from
     a script that reads in filtered data and level data.
    :param data1: single level filtered precipitation input data
    :param data2: multi-level dynamical variable, dimension 1 needs to match the levels
    given in levels
    :param levels: vertical levels to compute coherence at
    :param nDayWin: number of time steps per window
    :param nDaySkip: number of time steps to overlap per window
    :param spd: number of time steps per day
    :param siglevel: significance level
    :return: CohAvg, CohMask, CohMat: Vertical profile, masked cross-spectra at all levels,
    full cross-spectra at all levels
    """
    symmetries = ['symm', 'asymm']
    lat = data2['lat']
    # compute coherence - loop through levels
    for ll in np.arange(0, len(levels), 1):
        print('processing level = '+str(levels[ll]))
        for symm in symmetries:
            y = st.get_symmasymm(data2[:, ll, :, :], lat, symm)
            x = st.get_symmasymm(data1, lat, symm)
            # compute coherence
            result = st.mjo_cross(x, y, nDayWin, nDaySkip)
            tmp = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
            try:
                CrossMat
            except NameError:
                # initialize cross-spectral array
                freq = result['freq']
                freq = freq * spd
                wnum = result['wave']
                dims = tmp.shape
                CrossMat = xr.DataArray(np.empty([len(levels), dims[0]*2, dims[1], dims[2]]),
                                  dims=['level', 'cross', 'freq', 'wave'],
                                  coords={'level': levels, 'cross': np.arange(0, 16, 1), 'freq': freq, 'wave': wnum})

            # write cross-spectral components to array
            if symm == 'symm':
                CrossMat[ll, 0::2, :, :] = tmp
            elif symm == 'asymm':
                CrossMat[ll, 1::2, :, :] = tmp

    # compute significant value of coherence based on distribution
    sigval = coher_sig_dist(CrossMat[:, 8:10, :, :].values, siglevel)
    print(str(siglevel*100)+"% significance coherence value: "+str(sigval))

    # mask cross-spectra where coherence < siglevel
    MaskArray = CrossMat[:, 8:9, :, :]
    MaskArray = np.where(MaskArray <= sigval, np.nan, 1)
    MaskAll = np.empty(CrossMat.shape)
    for i in np.arange(0,8,1):
        MaskAll[:, i*2:i*2+1, :, :] = MaskArray
    CrossMask = CrossMat * MaskAll

    # average coherence across significant values
    CrossAvg = np.nanmean(CrossMask.sel(freq=slice(0, 1), wave=slice(-20, 20)), axis=(2, 3))

    # recompute phase angles of averaged cross-spectra
    CrossAvg = cross_phase_2d(CrossAvg)
    CrossAvg = xr.DataArray(CrossAvg, dims=['level', 'cross'], coords={'level': levels, 'cross': np.arange(0, 16, 1)})

    # return output
    return CrossAvg, CrossMask, CrossMat

def vertical_coherence_comp_bgcoh(data1, data2, levels, nDayWin, nDaySkip, spd, siglevel, N = 100, opt = 0):
    """
     Main driver to compute vertical coherence profile. This can be called from
     a script that reads in filtered data and level data. This function differs from the
     vertical_coherence_comp function in that it estimates the significant level of coh2
     by running one of the time series backwards an adding coh2 from 2 random normal time
     series.
    :param data1: single level filtered precipitation input data
    :param data2: multi-level dynamical variable, dimension 1 needs to match the levels
    given in levels
    :param levels: vertical levels to compute coherence at
    :param nDayWin: number of time steps per window
    :param nDaySkip: number of time steps to overlap per window
    :param spd: number of time steps per day
    :param siglevel: significance level
    :param N: number of random samples for the background coherence estimation
    :return: CohAvg, CohMask, CohMat: Vertical profile, masked cross-spectra at all levels,
    full cross-spectra at all levels
    """
    symmetries = ['symm', 'asymm']
    lat = data2['lat']
    # compute coherence - loop through levels
    for ll in np.arange(0, len(levels), 1):
        print('processing level = '+str(levels[ll]))
        for symm in symmetries:
            y = st.get_symmasymm(data2[:, ll, :, :], lat, symm)
            x = st.get_symmasymm(data1, lat, symm)
            # compute coherence
            result = st.mjo_cross(x, y, nDayWin, nDaySkip)
            tmp = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
            resultBG = st.mjo_cross(x[::-1,:,:], y, nDayWin, nDaySkip)
            tmpBG = resultBG['STC']
            try:
                CrossMat
            except NameError:
                # initialize cross-spectral array
                freq = result['freq']
                freq = freq * spd
                wnum = result['wave']
                dims = tmp.shape
                CrossMat = xr.DataArray(np.empty([len(levels), dims[0]*2, dims[1], dims[2]]),
                                  dims=['level', 'cross', 'freq', 'wave'],
                                  coords={'level': levels, 'cross': np.arange(0, 16, 1), 'freq': freq, 'wave': wnum})
                CohBGMat = xr.DataArray(np.empty([len(levels), 2, dims[1], dims[2]]),
                                        dims=['level', 'coh', 'freq', 'wave'],
                                        coords={'level': levels, 'coh': np.arange(0, 2, 1), 'freq': freq,
                                                'wave': wnum})

            # write cross-spectral components to array
            if symm == 'symm':
                CrossMat[ll, 0::2, :, :] = tmp
                CohBGMat[ll, 0, :, :] = tmpBG[4]
            elif symm == 'asymm':
                CrossMat[ll, 1::2, :, :] = tmp
                CohBGMat[ll, 1, :, :] = tmpBG[4]

    # compute significance levels
    if opt == 0:
        CohSig = coher_sig_bg(data1, data2, levels, nDayWin, nDaySkip, CohBGMat, spd, siglevel, N)
    elif opt == 1:
        CohSig = coher_sig_bg_lev(data1, data2, levels, nDayWin, nDaySkip, CohBGMat, spd, siglevel, N)

    # mask cross-spectra where coherence < siglevel
    MaskArray = CrossMat[:, 8:9, :, :]
    MaskArray = np.where(MaskArray <= CohSig, np.nan, 1)
    MaskAll = np.empty(CrossMat.shape)
    for i in np.arange(0, 8, 1):
        MaskAll[:, i*2:i*2+2, :, :] = MaskArray
    CrossMask = CrossMat * MaskAll

    # average coherence across significant values
    CrossAvg = np.nanmean(CrossMask.sel(freq=slice(0, 1), wave=slice(-20, 20)), axis=(2, 3))

    # recompute phase angles of averaged cross-spectra
    CrossAvg = cross_phase_2d(CrossAvg)
    CrossAvg = xr.DataArray(CrossAvg, dims=['level', 'cross'], coords={'level': levels, 'cross': np.arange(0, 16, 1)})

    # return output
    return CrossAvg, CrossMask, CrossMat


def coher_sig_dist(Coher, siglevel):
    """
    Compute the significant coherence level based on the distribution of coherence2.
    Sorts the coherence2 values by size and picks the value corresponding to the siglevel
    percentile. E.g. for a siglevel or 0.95 it picks the value of coherence2 larger than
    95% of all the input values. Based on testing this is a stronger requirement (i.e.
    gives larger coh2 values) than estimating the background coherence2 by running one
    time series backwards and adding values of coh2 for 2 random normal time series.
    :param Coher: numpy array containing all coherence2 values
    :return: sigval
    """
    # make a 1d array
    coher = Coher.flatten()
    # find all valid values
    coher = coher[~np.isnan(coher)]
    coher = coher[(0 <= coher) & (coher <= 1)]

    # sort array
    coher = np.sort(coher)
    nvals = len(coher)
    # find index of significant level
    isig = int(np.floor(nvals*siglevel))
    # read significant value
    sigval = coher[isig]

    return sigval


def coher_sig_bg(data1, data2, levels, nDayWin, nDaySkip, CohBGMat, spd, siglevel, N):
    """
    Compute random noise coh2 N times and add to background coh2. Sort and
    find siglevel values. Return significant coh2 values at each level and
    for both symmetric and anti-symmetric parts.
    :param data1: single level filtered precipitation input data
    :param data2: multi-level dynamical variable, dimension 1 needs to match the levels
    given in levels
    :param levels: vertical levels to compute coherence at
    :param nDayWin: number of time steps per window
    :param nDaySkip: number of time steps to overlap per window
    :param CohBGMat: background coherence matrix (lev x 2 x freq x wave)
    :param spd: number of time steps per day
    :param siglevel: significance level
    :param N: number of random samples for the background coherence estimation
    :return:
    """
    # compute standard deviation
    xstd = np.std(data1, axis=0)
    ystd = np.std(data2, axis=(0, 1))

    # generate random samples and compute coherence spectra
    for rr in np.arange(0, N, 1):
        xrand = np.random.normal(0, xstd, data1.shape)
        yrand = np.random.normal(0, ystd, data1.shape)
        resultR = st.mjo_cross(xrand, yrand, nDayWin, nDaySkip)
        tmpR = resultR['STC']
        try:
            CohR
        except NameError:
            freq = resultR['freq']
            freq = freq * spd
            wnum = resultR['wave']
            dims = tmpR.shape
            CohR = xr.DataArray(np.empty([N, dims[1], dims[2]]),
                                dims=['sample', 'freq', 'wave'],
                                coords={'sample': np.arange(0, N, 1), 'freq': freq, 'wave': wnum})
        CohR[rr, :, :] = tmpR[4]

    # expand dimensions of random sample coherence
    tmpC = CohR.values
    tmpR = tmpC[:, np.newaxis, np.newaxis, :, :]
    tmpR = np.tile(tmpR, (1, len(levels), 2, 1, 1))

    # add background and random sample coherence
    CohDist = np.broadcast_to(CohBGMat, (N, len(levels), 2, dims[1], dims[2])) + tmpR

    # sort coherence distribution
    CohDist = np.sort(CohDist, axis=0)

    # find siglevel index and get significant coherence
    isig = int(np.floor(N * siglevel))
    CohSig = CohDist[isig, :, :, :, :]

    return CohSig


def cross_phase_2d(Cross):
    """
    Compute phase angles from cross spectra.
    :param Cross: 2d array with the cross-spectra components in dim=1
    :return: Cross with replaced phase angles
    """

    # read co- and quadrature-spectral components
    cxys = Cross[:, 4]
    cxya = Cross[:, 5]
    qxys = Cross[:, 6]
    qxya = Cross[:, 7]

    # compute phase angles
    pha_s = np.arctan2(qxys, cxys)
    pha_a = np.arctan2(qxya, cxya)

    # compute phase vectors
    v1s = -qxys / np.sqrt(np.square(qxys) + np.square(cxys))
    v2s = cxys / np.sqrt(np.square(qxys) + np.square(cxys))
    v1a = -qxya / np.sqrt(np.square(qxya) + np.square(cxya))
    v2a = cxya / np.sqrt(np.square(qxya) + np.square(cxya))

    # write phase angles and vectors to array
    Cross[:, 10] = pha_s
    Cross[:, 11] = pha_a
    Cross[:, 12] = v1s
    Cross[:, 13] = v1a
    Cross[:, 14] = v2s
    Cross[:, 15] = v2a

    return Cross
