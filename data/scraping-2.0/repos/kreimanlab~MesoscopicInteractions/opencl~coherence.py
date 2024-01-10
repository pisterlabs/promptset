# Jiarui "Jerry" Wang :: jwang04@g.harvard.edu
# Kreiman Lab :: klab.tch.harvard.edu

import random
import h5py
import numpy
import scipy
from scipy import signal
from scipy import stats
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import multiprocessing as mp
import reikna
from reikna.cluda import dtypes, any_api
from reikna.fft import FFT
from reikna.core import Annotation, Type, Transformation, Parameter
from timeit import time
import math
                                                       
HG_S = 70
HG_E = 200

def get_complex_trf(arr):
    try:
        complex_dtype = dtypes.complex_for(arr.dtype)
    except KeyError:
        complex_dtype = 'complex64'
    return Transformation(
        [Parameter('output', Annotation(Type(complex_dtype, arr.shape), 'o')),
        Parameter('input', Annotation(arr, 'i'))],
        """
            ${output.store_same}(
            COMPLEX_CTR(${output.ctype})(
                ${input.load_same},
                0));
        """)

def coherence_gpu(v, nperseg, fs, thr, pli):
    #
    #   GPU implementation of magnitude square coherence estimate using Welch's
    #   method based on scipy.signal.coherence. Reports the pair-wise coherence
    #   of a multivariate input averaged over defined frequency bands.
    #
    #   Inputs:
    #       v           2-D matrix where each column is a variable and each row
    #                   is a time sample
    #
    #       nperseg     number of time samples per window segment
    #
    #       fs          sampling rate in Hz
    #
    #       thr         GPU thread object (reikna.cluda.api.Thread)
    #
    #                       Example thread generating code:
    #
    #                       import reikna
    #                       dev_n = 0 # GPU device index, this varies by machine
    #                       plat_n = 0 # platform index, this is usually 0
    #                       api = reikna.cluda.any_api()
    #                       plat = api.get_platforms()[plat_n]
    #                       p0_dev = plat.get_devices()
    #                       dev = p0_dev[dev_n] 
    #                       thr = api.Thread(dev)
    #
    #       pli         frequency of power line, in Hz
    #
    #   Outputs:
    #       R           N by M matrix where N is the number of frequency bands
    #                   defined below, and M is the number of columns of input
    #                   matrix v (in other words number of variables) choose 2.
    #
    #                   These pairs can be indexed by:
    #
    #                   for i in range(1, n_col):
    #                       for j in range(i+1, n_col+1):
    #                           v1_index = i-1
    #                           v2_index = j-1
    #
    #
    #   Testing:
    #       95 pct difference from scipy.signal.coherence: 0.000727467, n=15336
    #

    # --------------------------------------------------------------------------
    # Frequency boundaries in Hz
    #
    #   Any edits here requires also editing the frequency segmentation code
    #   immediately following coherence estimate (see below)
    #
    PLI_S = pli-4 #56      # Power line interference frequency
    PLI_E = pli+4 #64
    PL2_S = (fs-(3*pli))-2     # Power line interference frequency second band
    PL2_E = (fs-(3*pli))+2
    PL3_S = (2*pli)-3 #117      # Power line interference frequency third band
    PL3_E = (2*pli)+3 #123
    #
    THZ_S = 17
    THZ_E = 23
    #
    DEL_S = 0.5     # Delta wave
    DEL_E = 3
    THE_S = 3       # Theta wave
    THE_E = 8
    ALP_S = 8       # Alpha wave
    ALP_E = 12
    BET_S = 12      # Beta wave
    BET_E = 30
    GAM_S = 30      # Gamma wave
    GAM_E = 100
    BRO_S = 0.5     # Broadband
    BRO_E = 125
    N_BANDS = 6     # Number of frequency bands (not including power line)
    # --------------------------------------------------------------------------

    n_samples,n_chan = v.shape

    # Segmentation
    noverlap = int(nperseg * 0.8) # rounded
    stride = nperseg - noverlap
    n_seg = (n_samples - noverlap) // stride
    

    # -=- Zero-pad end of segment MATLAB-style -=-
    #nfft = int(numpy.max([256,2**(numpy.ceil(numpy.log2(nperseg)))]))

    # -=- Don't zero-pad Scipy-style -=-
    nfft = nperseg
    
    # Init segmented matrix
    V = numpy.zeros((nfft,n_chan,n_seg))
    f = numpy.linspace(0,(fs/2),math.ceil(nfft/2)+1)

    # frequency masks
    mask_del = ((f > DEL_S) & (f < DEL_E))
    mask_the = ((f >= THE_S) & (f < THE_E))
    mask_alp = ((f >= ALP_S) & (f < ALP_E))
    mask_bet = ((f >= BET_S) & (f < THZ_S)) | ((f > THZ_E) & (f < BET_E))
    mask_gam = ((f >= GAM_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < GAM_E))
    mask_bro = ((f > BRO_S) & (f < THZ_S)) | ((f > THZ_E) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < PL3_S)) | ((f > PL3_E) & (f < BRO_E))

    # -=- Hann window (Scipy default) -=-
    #win = numpy.hanning(nperseg)
    # -=- Hamming window (MATLAB default), more robust to PLI -=-
    win = numpy.hamming(nperseg)

    # Used to normalize out the effect of the window in the final power
    # (this does not affect the coherence)
    #win_s = numpy.sum(win ** 2) 

    # Loop through segments
    for i in range(n_seg):
        start_i = i*stride
        end_i = start_i + nperseg
        #pad = 0

        # Apply window
        for j in range(n_chan):
            V[0:(nperseg+0),j,i] = numpy.multiply(v[start_i:end_i,j],win)

    # FFT
    arr = V.astype(numpy.float32);
    trf = get_complex_trf(arr)
    fft = FFT(trf.output, axes=(0,))
    fft.parameter.input.connect(trf, trf.output, new_input=trf.input)
    cfft = fft.compile(thr)
    arr_dev = thr.to_device(arr)
    res_dev = thr.array(arr.shape, numpy.complex64)
    cfft(res_dev, arr_dev, 0)
    Vf = res_dev.get()

    #print(Vf.shape)
    Vf = Vf[0:(math.ceil(nfft/2)+1),:,:]

    # Coherence
    n_comb = int(0.5*n_chan*(n_chan-1))
    r = numpy.zeros((N_BANDS,n_comb))
    rf = numpy.zeros((N_BANDS,n_comb,math.ceil(nfft/2)+1))

    # Coherence phase 
    ph = numpy.zeros((N_BANDS,n_comb))
    phf = numpy.zeros((N_BANDS,n_comb,math.ceil(nfft/2)+1))

    Vf_ms = numpy.sqrt(numpy.sum(numpy.square(numpy.abs(Vf)),axis=2))
    Vf_c = numpy.conj(Vf)
    c_comb = 0
    for i in range(1, n_chan):
        for j in range(i+1, n_chan+1):
            Xii = Vf_ms[:,(i-1)]
            Xjj = Vf_ms[:,(j-1)]
            Xij = numpy.sum(numpy.multiply(Vf[:,(i-1),:],Vf_c[:,(j-1),:]), axis=1)
            CSD = numpy.divide(Xij,numpy.multiply(Xii,Xjj))
            #C2 = numpy.divide(numpy.abs(Xij),numpy.multiply(Xii,Xjj))
            #C2 = numpy.square(numpy.abs(CSD))
            C2 = numpy.abs(CSD)
            #C2 = C2[0:(math.ceil(nfft/2)+1)]

            # Plot phase
            #if ((i == 1) and (j == 7)):
            if (False):
                PHI = numpy.arctan2(numpy.imag(CSD),numpy.real(CSD))
                PHI = PHI[0:(math.ceil(nfft/2)+1)]
                fig = plt.figure()

                ax = plt.axes()
                rect = patches.Rectangle((PLI_S,0),(PLI_E-PLI_S),1,edgecolor='none',facecolor='gray')
                ax.add_patch(rect)
                rect = patches.Rectangle((PL2_S,0),(PL2_E-PL2_S),1,edgecolor='none',facecolor='gray')
                ax.add_patch(rect)
                rect = patches.Rectangle((PL3_S,0),(PL3_E-PL3_S),1,edgecolor='none',facecolor='gray')
                ax.add_patch(rect)
                rect = patches.Rectangle((THZ_S,0),(THZ_E-THZ_S),1,edgecolor='none',facecolor='gray')
                ax.add_patch(rect)
                ax.plot(f,C2,'black')
                #ax.plot([PLI_S,PLI_S],[0,1],'r:')
                #ax.plot([PLI_E,PLI_E],[0,1],'r:')
                #ax.plot([PL2_S,PL2_S],[0,1],'r:')
                #ax.plot([PL2_E,PL2_E],[0,1],'r:')
                #ax.plot([PL3_S,PL3_S],[0,1],'r:')
                #ax.plot([PL3_E,PL3_E],[0,1],'r:')
                #ax.plot([THZ_S,THZ_S],[0,1],'r:')
                #ax.plot([THZ_E,THZ_E],[0,1],'r:')
                ax.plot([DEL_S,DEL_S],[0,1],'b:')
                ax.plot([THE_S,THE_S],[0,1],'y:')
                ax.plot([ALP_S,ALP_S],[0,1],'c:')
                ax.plot([BET_S,BET_S],[0,1],'k:')
                ax.plot([GAM_S,GAM_S],[0,1],'m:')
                ax.plot([GAM_E,GAM_E],[0,1],'m:')
                ax.plot([BRO_S,BRO_S],[0,1],'g:')
                ax.plot([BRO_E,BRO_E],[0,1],'g:')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Coherence Magnitude')
                ax.set_title('BIP{0} - BIP{1}'.format(i,j))
                ax.set_xlim([min(f),max(f)])
                ax.set_ylim([0,1])
                fig.show()

                fig = plt.figure()
                ax = plt.axes()
                ax.plot(f,PHI,'black')
                ax.plot([PLI_S,PLI_S],[-math.pi,math.pi],'r:')
                ax.plot([PLI_E,PLI_E],[-math.pi,math.pi],'r:')
                ax.plot([PL2_S,PL2_S],[-math.pi,math.pi],'r:')
                ax.plot([PL2_E,PL2_E],[-math.pi,math.pi],'r:')
                ax.plot([PL3_S,PL3_S],[-math.pi,math.pi],'r:')
                ax.plot([PL3_E,PL3_E],[-math.pi,math.pi],'r:')
                ax.plot([THZ_S,THZ_S],[-math.pi,math.pi],'r:')
                ax.plot([THZ_E,THZ_E],[-math.pi,math.pi],'r:')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Coherence Phase (radians)')
                ax.set_title('BIP{0} - BIP{1}'.format(i,j))
                ax.set_xlim([min(f),max(f)])
                ax.set_ylim([-math.pi,math.pi])
                fig.show()
                a = input('enter to close.')
                exit()

            # ------------------------------------------------------------------
            # Frequency band segmentation
            #r[0,c_comb] = numpy.mean(C2[mask_del])
            #r[1,c_comb] = numpy.mean(C2[mask_the])
            #r[2,c_comb] = numpy.mean(C2[mask_alp])
            #r[3,c_comb] = numpy.mean(C2[mask_bet])
            #r[4,c_comb] = numpy.mean(C2[mask_gam])
            #r[5,c_comb] = numpy.mean(C2[mask_bro])
            # ------------------------------------------------------------------
            # Phase
            #ph[0,c_comb] = numpy.mean(PHI[mask_del])
            #ph[1,c_comb] = numpy.mean(PHI[mask_the])
            #ph[2,c_comb] = numpy.mean(PHI[mask_alp])
            #ph[3,c_comb] = numpy.mean(PHI[mask_bet])
            #ph[4,c_comb] = numpy.mean(PHI[mask_gam])
            #ph[5,c_comb] = numpy.mean(PHI[mask_bro])

            rf[:,c_comb,:] = C2;
            phf[:,c_comb,:] = numpy.arctan2(numpy.imag(CSD),numpy.real(CSD))
            # --------------------------------- freq seg --- 0.16672945022583008

            c_comb = c_comb + 1
            
    #tv = numpy.mean(rf[0,:,mask_del],axis=0)
    #print(tv.shape)
    r[0,:] = numpy.mean(rf[0,:,mask_del],axis=0)
    r[1,:] = numpy.mean(rf[1,:,mask_the],axis=0)
    r[2,:] = numpy.mean(rf[2,:,mask_alp],axis=0)
    r[3,:] = numpy.mean(rf[3,:,mask_bet],axis=0)
    r[4,:] = numpy.mean(rf[4,:,mask_gam],axis=0)
    r[5,:] = numpy.mean(rf[5,:,mask_bro],axis=0)
 
    ph[0,:] = numpy.mean(phf[0,:,mask_del],axis=0)
    ph[1,:] = numpy.mean(phf[1,:,mask_the],axis=0)
    ph[2,:] = numpy.mean(phf[2,:,mask_alp],axis=0)
    ph[3,:] = numpy.mean(phf[3,:,mask_bet],axis=0)
    ph[4,:] = numpy.mean(phf[4,:,mask_gam],axis=0)
    ph[5,:] = numpy.mean(phf[5,:,mask_bro],axis=0)

    return r,ph


def coherence2_gpu(v, v2, nperseg, fs, thr, pli):
    #
    #   GPU implementation of magnitude square coherence estimate using Welch's
    #   method based on scipy.signal.coherence. Reports the pair-wise coherence
    #   of a multivariate input averaged over defined frequency bands.
    #
    #   Differs from coherence_gpu in that it reports the coherence between two
    #   multivariate matrices v and v2. For example, v2 could be v randomly 
    #   shifted in time, then coherence2_gpu would report the null pairwise
    #   coherence
    #
    #   Inputs:
    #       v           2-D matrix where each column is a variable and each row
    #                   is a time sample
    #
    #       v2          2-D matrix where each column is a variable and each row
    #                   is a time sample. Must be same size as v.
    #
    #       nperseg     number of time samples per window segment
    #
    #       fs          sampling rate in Hz
    #
    #       thr         GPU thread object (reikna.cluda.api.Thread)
    #
    #                       Example thread generating code:
    #
    #                       import reikna
    #                       dev_n = 0 # GPU device index, this varies by machine
    #                       plat_n = 0 # platform index, this is usually 0
    #                       api = reikna.cluda.any_api()
    #                       plat = api.get_platforms()[plat_n]
    #                       p0_dev = plat.get_devices()
    #                       dev = p0_dev[dev_n] 
    #                       thr = api.Thread(dev)
    #
    #       pli         frequency of power line, in Hz
    #
    #   Outputs:
    #       R           N by M matrix where N is the number of frequency bands
    #                   defined below, and M is the number of columns of input
    #                   matrix v (in other words number of variables) choose 2.
    #
    #                   These pairs can be indexed by:
    #
    #                   for i in range(1, n_col):
    #                       for j in range(i+1, n_col+1):
    #                           v1_index = i-1
    #                           v2_index = j-1
    #
    #
    #   Testing:
    #       95 pct difference from scipy.signal.coherence: 0.000727467, n=15336
    #

    # --------------------------------------------------------------------------
    # Frequency boundaries in Hz
    #
    #   Any edits here requires also editing the frequency segmentation code
    #   immediately following coherence estimate (see below)
    #
    PLI_S = pli-4 #56      # Power line interference frequency
    PLI_E = pli+4 #64
    PL2_S = (fs-(3*pli))-2     # Power line interference frequency second band
    PL2_E = (fs-(3*pli))+2
    PL3_S = (2*pli)-3 #117      # Power line interference frequency third band
    PL3_E = (2*pli)+3 #123
    #
    THZ_S = 17
    THZ_E = 23
    #
    DEL_S = 0.5     # Delta wave
    DEL_E = 3
    THE_S = 3       # Theta wave
    THE_E = 8
    ALP_S = 8       # Alpha wave
    ALP_E = 12
    BET_S = 12      # Beta wave
    BET_E = 30
    GAM_S = 30      # Gamma wave
    GAM_E = 100
    BRO_S = 0.5     # Broadband
    BRO_E = 125
    N_BANDS = 6     # Number of frequency bands (not including power line)
    # --------------------------------------------------------------------------

    if (not (v.shape == v2.shape)):
        print('E: first two input matrices to coherence2_gpu must be same size.')

    n_samples,n_chan = v.shape

    # Segmentation
    noverlap = int(nperseg * 0.8) # rounded
    stride = nperseg - noverlap
    n_seg = (n_samples - noverlap) // stride
    
    # -=- Zero-pad end of segment MATLAB-style -=-
    #nfft = int(numpy.max([256,2**(numpy.ceil(numpy.log2(nperseg)))]))

    # -=- Don't zero-pad Scipy-style -=-
    nfft = nperseg
 
    # Init segmented matrix
    V = numpy.zeros((nfft,n_chan,n_seg))
    V2 = numpy.zeros((nfft,n_chan,n_seg))
    f = numpy.linspace(0,(fs/2),math.ceil(nfft/2)+1)

    # frequency masks
    mask_del = ((f > DEL_S) & (f < DEL_E))
    mask_the = ((f >= THE_S) & (f < THE_E))
    mask_alp = ((f >= ALP_S) & (f < ALP_E))
    #mask_bet = ((f >= BET_S) & (f < BET_E))
    #mask_gam = ((f >= GAM_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < GAM_E))
    #mask_bro = ((f > BRO_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < PL3_S)) | ((f > PL3_E) & (f < BRO_E))
    mask_bet = ((f >= BET_S) & (f < THZ_S)) | ((f > THZ_E) & (f < BET_E))
    mask_gam = ((f >= GAM_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < GAM_E))
    mask_bro = ((f > BRO_S) & (f < THZ_S)) | ((f > THZ_E) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < PL3_S)) | ((f > PL3_E) & (f < BRO_E))

    n_del = numpy.sum(mask_del)
    n_the = numpy.sum(mask_the)
    n_alp = numpy.sum(mask_alp)
    n_bet = numpy.sum(mask_bet)
    n_gam = numpy.sum(mask_gam)
    n_bro = numpy.sum(mask_bro)

    # === 0.00017261505126953125

    # -=- Hann window (Scipy default) -=-
    #win = numpy.hanning(nperseg)
    # -=- Hamming window (MATLAB default), more robust to PLI -=-
    win = numpy.hamming(nperseg)

    # Used to normalize out the effect of the window in the final power
    # (this does not affect the coherence)
    #win_s = numpy.sum(win ** 2) 

    # Loop through segments
    for i in range(n_seg):
        start_i = i*stride
        end_i = start_i + nperseg
        #pad = 0

        # Apply window
        for j in range(n_chan):
            V[0:nperseg,j,i] = numpy.multiply(v[start_i:end_i,j],win)
            V2[0:nperseg,j,i] = numpy.multiply(v2[start_i:end_i,j],win)

    # === 0.13275456428527832
    
    # FFT
    V12 = numpy.concatenate((V,V2),axis=1)
    arr = V12.astype(numpy.float32);
    trf = get_complex_trf(arr)
    fft = FFT(trf.output, axes=(0,))
    fft.parameter.input.connect(trf, trf.output, new_input=trf.input)
    # === 0.04143881797790527 
    cfft = fft.compile(thr)
    # === 0.015437126159667969
    arr_dev = thr.to_device(arr)
    # === 0.0001125335693359375
    res_dev = thr.array(arr.shape, numpy.complex64)
    cfft(res_dev, arr_dev, 0)
    # === 0.0003275871276855469
    Vf = res_dev.get()
    # === 0.04879593849182129

    Vf = Vf[0:(math.ceil(nfft/2)+1),:,:]

    # Coherence
    n_comb = int(0.5*n_chan*(n_chan-1))
    r = numpy.zeros((N_BANDS,n_comb))
    rf = numpy.zeros((N_BANDS,n_comb,math.ceil(nfft/2)+1))
    c_comb = 0

    # Coherence phase 
    ph = numpy.zeros((N_BANDS,n_comb))
    phf = numpy.zeros((N_BANDS,n_comb,math.ceil(nfft/2)+1))

    Vf_ms = numpy.sqrt(numpy.sum(numpy.square(numpy.abs(Vf)),axis=2))
    #Vf_ms = numpy.sum(numpy.square(numpy.abs(Vf)),axis=2)
    # === 0.07235550880432129
    Vf_c = numpy.conj(Vf)
    # === 0.02622389793395996
    #print('SHAPE\n',Vf.shape,Vf_ms.shape) (257, 308, 59) (257, 308)
    #t0 = time.time()
    #t1 = 0
    for i in range(1, n_chan):
        for j in range(i+1, n_chan+1):
            # ------------------------------------- coh --- start
            Xii = Vf_ms[:,(i-1)]
            # ------------------------------------- coh --- 0.001316070556640625
            Xjj = Vf_ms[:,(j-1+n_chan)]
            # ------------------------------------- coh --- 0.001050233840942383
            #ta = Vf[:,(i-1),:]
            # --- 0.006984233856201172
            #tb = Vf_c[:,(j-1+n_chan),:]
            #print('--- SHAPE:\n',ta.shape,tb.shape) # (512, 59) (512, 59)
            # --- 0.006033420562744141
            #Vfcp = numpy.multiply(ta, tb)
            # --- 0.9781203269958496
            Vfcp = numpy.multiply(Vf[:,(i-1),:], Vf_c[:,(j-1+n_chan),:])
            # ------------------------------------- coh --- 0.21906280517578125
            Xij = numpy.sum(Vfcp, axis=1)
            # ------------------------------------- coh --- 0.06380510330200195
            #CSD = numpy.divide(Xij,numpy.multiply(numpy.sqrt(Xii),numpy.sqrt(Xjj)))
            #CSD = numpy.divide(Xij,numpy.multiply(Xii,Xjj))
            # Xii and Xjj are already square-rooted outside of for loop
            C2 = numpy.divide(numpy.abs(Xij),numpy.multiply(Xii,Xjj))
            # ------------------------------------- coh --- 0.02692723274230957
            #C2 = numpy.square(numpy.abs(CSD))
            #C2 = numpy.abs(CSD)
            # ------------------------------------- coh --- 0.013168811798095703
            #C2 = C2[0:(math.ceil(nfft/2)+1)]
            # ------------------------------------- coh --- 0.027124404907226562
            # Frequency band segmentation
            # --------------------------------- freq seg --- start
            #C2del = C2[mask_del]
            #C2the = C2[mask_the]
            #C2alp = C2[mask_alp]
            #C2bet = C2[mask_bet]
            #C2gam = C2[mask_gam]
            #C2bro = C2[mask_bro]
            # --- 0.039420127868652344
            #C2del = numpy.mean(C2del)
            # - 0.14268064498901367
            #C2the = numpy.mean(C2the)
            # - 0.1090383529663086
            #C2alp = numpy.mean(C2alp)
            #C2bet = numpy.mean(C2bet)
            #C2gam = numpy.mean(C2gam)
            #t0 = time.time()
            #C2bro = numpy.sum(C2bro)/n_bro # - 0.07562899589538574
            #t1 = t1 + (time.time() - t0)
            # - 0.10579109191894531
            # --- 0.6355729103088379
            #r[0,c_comb] = C2del
            #r[1,c_comb] = C2the
            #r[2,c_comb] = C2alp
            #r[3,c_comb] = C2bet
            #r[4,c_comb] = C2gam
            #r[5,c_comb] = C2bro
            # --- 0.012447595596313477
            #r[0,c_comb] = numpy.sum(C2[mask_del])/n_del
            #r[1,c_comb] = numpy.sum(C2[mask_the])/n_the
            #r[2,c_comb] = numpy.sum(C2[mask_alp])/n_alp
            #r[3,c_comb] = numpy.sum(C2[mask_bet])/n_bet
            #r[4,c_comb] = numpy.sum(C2[mask_gam])/n_gam
            #r[5,c_comb] = numpy.sum(C2[mask_bro])/n_bro
            #r[0,c_comb] = numpy.mean(C2[mask_del])
            #r[1,c_comb] = numpy.mean(C2[mask_the])
            #r[2,c_comb] = numpy.mean(C2[mask_alp])
            #r[3,c_comb] = numpy.mean(C2[mask_bet])
            #r[4,c_comb] = numpy.mean(C2[mask_gam])
            #r[5,c_comb] = numpy.mean(C2[mask_bro])

            rf[:,c_comb,:] = C2;
            # --------------------------------- freq seg --- 0.16672945022583008

            # coherence phase
            # Xii and Xjj are already square-rooted outside of for loop
            CSD = numpy.divide(Xij,numpy.multiply(Xii,Xjj))
            phf[:,c_comb,:] = numpy.arctan2(numpy.imag(CSD),numpy.real(CSD))
            #PHI = numpy.arctan2(numpy.imag(CSD),numpy.real(CSD))
            #PHI = PHI[0:(math.ceil(nfft/2)+1)]

            c_comb = c_comb + 1
            
    #tv = numpy.mean(rf[0,:,mask_del],axis=0)
    #print(tv.shape)
    r[0,:] = numpy.mean(rf[0,:,mask_del],axis=0)
    r[1,:] = numpy.mean(rf[1,:,mask_the],axis=0)
    r[2,:] = numpy.mean(rf[2,:,mask_alp],axis=0)
    r[3,:] = numpy.mean(rf[3,:,mask_bet],axis=0)
    r[4,:] = numpy.mean(rf[4,:,mask_gam],axis=0)
    r[5,:] = numpy.mean(rf[5,:,mask_bro],axis=0)

    ph[0,:] = numpy.mean(phf[0,:,mask_del],axis=0)
    ph[1,:] = numpy.mean(phf[1,:,mask_the],axis=0)
    ph[2,:] = numpy.mean(phf[2,:,mask_alp],axis=0)
    ph[3,:] = numpy.mean(phf[3,:,mask_bet],axis=0)
    ph[4,:] = numpy.mean(phf[4,:,mask_gam],axis=0)
    ph[5,:] = numpy.mean(phf[5,:,mask_bro],axis=0)

    #print(r.shape)
    #print(c_comb-1)
    # --- total for loop time --- 1.4920814037322998
    #print('===== PROFILE =====\n',time.time() - t0,'\n===================')
    #print('===== PROFILE =====\n',t1,'\n===================')

    return r,ph


def coherence_gpu_test(v, nperseg, fs, thr):

    # Frequency boundaries in Hz
    PLI_S = 56      # Power line interference frequency
    PLI_E = 64
    PL2_S = (fs-180)-2     # Power line interference frequency second band
    PL2_E = (fs-180)+2
    PL3_S = 117      # Power line interference frequency third band
    PL3_E = 123
    #
    THZ_S = 17
    THZ_E = 23
    #
    DEL_S = 0.5
    DEL_E = 3
    THE_S = 3
    THE_E = 8
    ALP_S = 8
    ALP_E = 12
    BET_S = 12
    BET_E = 25
    GAM_S = 25
    GAM_E = 100
    BRO_S = 0.5
    BRO_E = 125
    N_BANDS = 6

    n_samples,n_chan = v.shape

    # Segmentation
    noverlap = int(nperseg * 0.8) # rounded
    stride = nperseg - noverlap
    n_seg = (n_samples - noverlap) // stride
    
    # Init segmented matrix
    nfft = nperseg + 1
    V = numpy.zeros((nfft,n_chan,n_seg))
    f = numpy.linspace(0,(fs/2),math.ceil(nfft/2))

    # Get window and normalization coefficient
    #win = numpy.hanning(nfft)
    win = numpy.hanning(nperseg)
    win_s = numpy.sum(win ** 2)
    #win_s = numpy.sum(win ** 2) * ((2 * math.pi)**2 )

    # Loop through segments
    for i in range(n_seg):
        start_i = i*stride
        #end_i = start_i + nfft
        end_i = start_i + nperseg
        pad = 0

        # Apply window
        for j in range(n_chan):

            #print('nfft: ',nfft)
            #print('v[start_i:end_i,j].shape: ',v[start_i:end_i,j].shape)
            #print('V[0:nperseg,j,i].shape: ',V[0:nperseg,j,i].shape)
            V[0:nperseg,j,i] = numpy.multiply(v[start_i:end_i,j],win)
            #V[nperseg,j,i] = pad
            V[0,j,i] = pad
         
    print('\nV:')
    print(V[0:5,0:4,0])
    print(V[0:5,0:4,1],'\n...')
    print(V[0:5,0:4,-1],V.shape)

    # FFT
    #Vp = numpy.square(abs(numpy.fft.fft(V, axis=0))) / win_s
    #print('\nVp:')
    #print(Vp[0:5,0:4,0])
    #print(Vp[0:5,0:4,1],'\n...')
    #print(Vp[0:5,0:4,-1],Vp.shape)

    arr = V.astype(numpy.float32);
    trf = get_complex_trf(arr)
    fft = FFT(trf.output, axes=(0,))
    fft.parameter.input.connect(trf, trf.output, new_input=trf.input)
    cfft = fft.compile(thr)
    arr_dev = thr.to_device(arr)
    res_dev = thr.array(arr.shape, numpy.complex64)
    cfft(res_dev, arr_dev, 0)
    Vf = res_dev.get()
    Vp_gpu = numpy.square(abs(Vf)) / win_s

    print('\nVp_gpu:')
    print(Vp_gpu[0:5,0:4,0])
    print(Vp_gpu[0:5,0:4,1],'\n...')
    print(Vp_gpu[0:5,0:4,-1],Vp_gpu.shape)

    # Coherence
    n_comb = int(0.5*n_chan*(n_chan-1))
    r = numpy.zeros((N_BANDS,n_comb))
    c_comb = 0

    # normalization
    #Vf = Vf / win_s
    #Vf2 = numpy.zeros((math.ceil(nfft/2),n_chan,n_seg),numpy.complex64)

    for i in range(1, n_chan):
        for j in range(i+1, n_chan+1):
            k = 0
            Xii = numpy.abs(numpy.square(Vf[:,(i-1),k]))
            Xjj = numpy.abs(numpy.square(Vf[:,(j-1),k]))
            Xij = numpy.multiply(Vf[:,i-1,k],numpy.conj(Vf[:,(j-1),k]))
            #print('k: ',k)
            #print('\nXii: \n',Xii[0:5],Xii.shape)
            #print('\nXjj: \n',Xjj[0:5],Xjj.shape)
            #print('\nXij: \n',Xij[0:5],Xij.shape)
            for k in range(n_seg-1):
                #print('k: ',k+1)
                Xii = Xii + numpy.abs(numpy.square(Vf[:,(i-1),k+1]))
                Xjj = Xjj + numpy.abs(numpy.square(Vf[:,(j-1),k+1]))
                Xij = Xij + numpy.multiply(Vf[:,(i-1),k+1],numpy.conj(Vf[:,(j-1),k+1]))
            CSD = numpy.divide(Xij,numpy.multiply(numpy.sqrt(Xii),numpy.sqrt(Xjj)))
            #C2 = numpy.square(numpy.abs(CSD))
            C2 = numpy.abs(CSD)
            C2 = C2[0:(math.ceil(nfft/2))]
            #C2 = numpy.divide(numpy.square(abs(Xij)),numpy.multiply(Xii,Xjj))
            #print('\nCSD: \n',CSD[0:8],CSD.shape)
            #print('\nC2: \n',C2[0:8],C2.shape)
            #f, C2 = signal.coherence(v[:,i-1],v[:,j-1],fs=fs, nperseg=fs*2, detrend=False)
            mask_del = ((f > DEL_S) & (f < DEL_E))
            mask_the = ((f >= THE_S) & (f < THE_E))
            mask_alp = ((f >= ALP_S) & (f < ALP_E))
            #mask_bet = ((f >= BET_S) & (f < BET_E))
            #mask_gam = ((f >= GAM_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < GAM_E))
            #mask_bro = ((f > BRO_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < PL3_S)) | ((f > PL3_E) & (f < BRO_E))
            mask_bet = ((f >= BET_S) & (f < THZ_S)) | ((f > THZ_E) & (f < BET_E))
            mask_gam = ((f >= GAM_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < GAM_E))
            mask_bro = ((f > BRO_S) & (f < THZ_S)) | ((f > THZ_E) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < PL3_S)) | ((f > PL3_E) & (f < BRO_E))

            r[0,c_comb] = numpy.mean(C2[mask_del])
            r[1,c_comb] = numpy.mean(C2[mask_the])
            r[2,c_comb] = numpy.mean(C2[mask_alp])
            r[3,c_comb] = numpy.mean(C2[mask_bet])
            r[4,c_comb] = numpy.mean(C2[mask_gam])
            r[5,c_comb] = numpy.mean(C2[mask_bro])

            #print(r[:,c_comb])
            c_comb = c_comb + 1


    # Average across segments
    #Vp_gpu = Vp_gpu.sum(axis=2) / n_seg
    Vp_gpu = numpy.sum(Vp_gpu, axis=2) / n_seg
    print('\nVp_gpu avg:')
    print(Vp_gpu[0:5,0:4],Vp_gpu.shape)

    # Normalize accordingly to pwelch
    Vp_gpu = Vp_gpu / ((2*math.pi)**3)
    #Vp_gpu = Vp_gpu / (2*math.pi)

    # Chop off second half
    Vp_gpu = Vp_gpu[0:(math.ceil(nfft/2)),:]

    # Adjust for chopping off second half
    Vp_gpu[1:,:] = 2 * Vp_gpu[1:,:]

    # Build frequency vector
    f_gpu = numpy.linspace(0,(fs/2),math.ceil(nfft/2))
    print('\nVp_gpu pxx:')
    print(Vp_gpu[0:5,0:4],Vp_gpu.shape)
    print('\nVp_gpu f:')
    print(f_gpu[0:5],f_gpu[-1],f_gpu.shape)

    # Compare to scipy
    f_sp, Vp_sp = scipy.signal.welch(v, fs=fs, window='hamming', nperseg=nperseg, noverlap=round(0.8*nperseg), detrend=False, axis=0)
    print('\nVp_sp pxx:')
    print(Vp_sp[0:5,0:4],Vp_sp.shape)
    print('\nVp_sp f:')
    print(f_sp[0:5],f_sp[-1],f_sp.shape)
    


    #plt.plot(f, C2)
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Magnitude Squared Coherence')
    #plt.show()

    return r



def coherence(v, fs, npseg, pli):

    # Frequency boundaries in Hz
    #PLI_S = 56      # Power line interference frequency
    #PLI_E = 64
    #PL2_S = (fs-180)-2     # Power line interference frequency second band
    #PL2_E = (fs-180)+2
    #PL3_S = 117      # Power line interference frequency third band
    #PL3_E = 123
    
    PLI_S = pli-4 #56      # Power line interference frequency
    PLI_E = pli+4 #64
    PL2_S = (fs-(3*pli))-2     # Power line interference frequency second band
    PL2_E = (fs-(3*pli))+2
    PL3_S = (2*pli)-3 #117      # Power line interference frequency third band
    PL3_E = (2*pli)+3 #123
    #
    THZ_S = 17
    THZ_E = 23
    #
    DEL_S = 0.5
    DEL_E = 3
    THE_S = 3
    THE_E = 8
    ALP_S = 8
    ALP_E = 12
    BET_S = 12
    BET_E = 25
    GAM_S = 25
    GAM_E = 100
    BRO_S = 0.5
    BRO_E = 125
    N_BANDS = 6

    n_samples,n_chan = v.shape
    n_comb = int(0.5*n_chan*(n_chan-1))
    r = numpy.zeros((N_BANDS,n_comb))

    c_comb = 0
    for i in range(1, n_chan):
        for j in range(i+1, n_chan+1):
            f, C2 = signal.coherence(v[:,i-1],v[:,j-1],fs=fs, nperseg=npseg, noverlap=round(0.8*npseg), detrend=False, window='hamming')
            C2 = numpy.sqrt(C2)

            mask_del = ((f > DEL_S) & (f < DEL_E))
            mask_the = ((f >= THE_S) & (f < THE_E))
            mask_alp = ((f >= ALP_S) & (f < ALP_E))
            #mask_bet = ((f >= BET_S) & (f < BET_E))
            #mask_gam = ((f >= GAM_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < GAM_E))
            #mask_bro = ((f > BRO_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < PL3_S)) | ((f > PL3_E) & (f < BRO_E))
            mask_bet = ((f >= BET_S) & (f < THZ_S)) | ((f > THZ_E) & (f < BET_E))
            mask_gam = ((f >= GAM_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < GAM_E))
            mask_bro = ((f > BRO_S) & (f < THZ_S)) | ((f > THZ_E) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < PL3_S)) | ((f > PL3_E) & (f < BRO_E))
            r[0,c_comb] = numpy.mean(C2[mask_del])
            r[1,c_comb] = numpy.mean(C2[mask_the])
            r[2,c_comb] = numpy.mean(C2[mask_alp])
            r[3,c_comb] = numpy.mean(C2[mask_bet])
            r[4,c_comb] = numpy.mean(C2[mask_gam])
            r[5,c_comb] = numpy.mean(C2[mask_bro])

            #print(r[:,c_comb])
            c_comb = c_comb + 1

    #plt.plot(f, C2)
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Magnitude Squared Coherence')
    #plt.show()

    return r


def coherence2(v, w, npseg, fs, pli):

    # Frequency boundaries in Hz
    #PLI_S = 56      # Power line interference frequency
    #PLI_E = 64
    #PL2_S = (fs-180)-2     # Power line interference frequency second band
    #PL2_E = (fs-180)+2
    #PL3_S = 117      # Power line interference frequency third band
    #PL3_E = 123
    PLI_S = pli-4 #56      # Power line interference frequency
    PLI_E = pli+4 #64
    PL2_S = (fs-(3*pli))-2     # Power line interference frequency second band
    PL2_E = (fs-(3*pli))+2
    PL3_S = (2*pli)-3 #117      # Power line interference frequency third band
    PL3_E = (2*pli)+3 #123
    #
    THZ_S = 17
    THZ_E = 23
    #
    DEL_S = 0.5
    DEL_E = 3
    THE_S = 3
    THE_E = 8
    ALP_S = 8
    ALP_E = 12
    BET_S = 12
    BET_E = 25
    GAM_S = 25
    GAM_E = 100
    BRO_S = 0.5
    BRO_E = 125
    N_BANDS = 6

    n_samples,n_chan = v.shape
    n_comb = int(0.5*n_chan*(n_chan-1))
    r = numpy.zeros((N_BANDS,n_comb))

    c_comb = 0
    for i in range(1, n_chan):
        for j in range(i+1, n_chan+1):
            f, C2 = signal.coherence(v[:,i-1],w[:,j-1],fs=fs, nperseg=npseg, noverlap=round(0.8*npseg), detrend=False, window='hamming')
            C2 = numpy.sqrt(C2)
            mask_del = ((f > DEL_S) & (f < DEL_E))
            mask_the = ((f >= THE_S) & (f < THE_E))
            mask_alp = ((f >= ALP_S) & (f < ALP_E))
            #mask_bet = ((f >= BET_S) & (f < BET_E))
            #mask_gam = ((f >= GAM_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < GAM_E))
            #mask_bro = ((f > BRO_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < PL3_S)) | ((f > PL3_E) & (f < BRO_E))
            mask_bet = ((f >= BET_S) & (f < THZ_S)) | ((f > THZ_E) & (f < BET_E))
            mask_gam = ((f >= GAM_S) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < GAM_E))
            mask_bro = ((f > BRO_S) & (f < THZ_S)) | ((f > THZ_E) & (f < PLI_S)) | ((f > PLI_E) & (f < PL2_S)) | ((f > PL2_E) & (f < PL3_S)) | ((f > PL3_E) & (f < BRO_E))
            r[0,c_comb] = numpy.mean(C2[mask_del])
            r[1,c_comb] = numpy.mean(C2[mask_the])
            r[2,c_comb] = numpy.mean(C2[mask_alp])
            r[3,c_comb] = numpy.mean(C2[mask_bet])
            r[4,c_comb] = numpy.mean(C2[mask_gam])
            r[5,c_comb] = numpy.mean(C2[mask_bro])

            #print(r[:,c_comb])
            c_comb = c_comb + 1

    #plt.plot(f, C2)
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Magnitude Squared Coherence')
    #plt.show()

    return r

def main():
    h5fname = '/media/jerry/internal/data/h5_notch20/sub3.h5'
    #h5fname = '/media/jerry/KLAB101/h5_notch20/sub40.h5'
    #h5fname = '/mnt/cuenap/data/h5_notch20/sub1.h5'
    #h5fname = '/mnt/cuenap2/scripts/synth/out/sub40.h5'
    #h5fname = '/mnt/cuenap2/scripts/synth/out/sub2.h5'
    #h5fname = '/mnt/cuenap2/scripts/synth/out/sub1.h5'
    h5f = h5py.File(h5fname,'r')
    fs = int(round(h5f['/h5eeg/eeg'].attrs['rate'][0]))
    print('fs:',fs)
    n_chan = int(round(h5f['/h5eeg/eeg'].attrs['n_chan'][0]))
    n_samples = int(round(h5f['/h5eeg/eeg'].attrs['n_samples'][0]))
    bip = h5f['/h5eeg/eeg'].attrs['bip']
    e1 = numpy.array(bip[0,:], numpy.float32)
    e2 = numpy.array(bip[1,:], numpy.float32)

    w = 10
    r_idx = random.randint(0,(n_samples-w*round(fs)))
    r_rows = int(w*round(fs))
    X = h5f['/h5eeg/eeg'][r_idx:(r_idx+r_rows),0:(0+n_chan)]
    Xbip = numpy.zeros((r_rows,len(e1)))
    for i in range(len(e1)):
        Xbip[:,i] = (X[:,int(e1[i]-1)] - X[:,int(e2[i]-1)])

    #for i in range(n_chan):
    #    X[:,i] = stats.rankdata(X[:,i])

    r_idx = random.randint(0,(n_samples-w*round(fs)))
    Y = h5f['/h5eeg/eeg'][r_idx:(r_idx+r_rows),0:(0+n_chan)]
    Ybip = numpy.zeros((r_rows,len(e1)))
    for i in range(len(e1)):
        Ybip[:,i] = (Y[:,int(e1[i]-1)] - Y[:,int(e2[i]-1)])

    #for i in range(n_chan):
    #    Y[:,i] = stats.rankdata(Y[:,i])


    
    # GPU
    dev_n = 0
    plat_n = 0
    api = any_api()
    plat = api.get_platforms()[plat_n]
    p0_dev = plat.get_devices()
    dev = p0_dev[dev_n] 
    thr = api.Thread(dev)

    fs = round(1 * fs)
    
    t0 = time.time()
    r_coha,ph = coherence_gpu(Xbip, round(fs), fs, thr, 60)
    t_gpu = time.time() - t0
    print('gpu coherence: \n',r_coha[5,0:10],'...',r_coha[5,-1])
    print('gpu time: ',t_gpu)

    if (True):
        t0 = time.time()
        r_coh = coherence(Xbip, fs, round(fs), 60)
        t_cpu = time.time() - t0
        print('scipy coherence: \n',r_coh[5,0:10],'...',r_coh[5,-1])
        print('cpu time: ',t_cpu)
        print('gpu speedup: ',t_cpu/t_gpu)

        #print('GPU\n',numpy.sqrt(r_coha))
        #print('CPU\n',numpy.sqrt(r_coh))
        d = numpy.abs(numpy.sqrt(r_coha[5,:]) - numpy.sqrt(r_coh[5,:])).ravel()
        #print(d,d.shape)
        d = sorted(d)
        d = [x for x in d if str(x) != 'nan']
        #d_mu = numpy.mean(numpy.abs(r_coha - r_coh))
        #d_sig = numpy.std(numpy.abs(r_coha - r_coh))
        #print('avg abs difference: ', d_mu)
        #print('std abs difference: ', d_sig)
        print('mean difference: ', numpy.mean(d))
        print('std difference: ', numpy.std(d))
        print('95 pct difference: ', d[int(0.95*len(d))])
        print('max difference: ', max(d),'\n')
        print('n: ', len(d))

    t0 = time.time()
    r_coh2a,ph = coherence2_gpu(Xbip, Ybip, round(fs), fs, thr, 60)
    t_gpu = time.time() - t0
    print('gpu time: ',t_gpu)

    #exit()

    if (True):
        t0 = time.time()
        r_coh2 = coherence2(Xbip, Ybip, round(fs), fs, 60)
        t_cpu = time.time() - t0
        print('cpu time: ',t_cpu)
        print('gpu speedup: ',t_cpu/t_gpu)

        d = numpy.abs(numpy.sqrt(r_coh2a[5,:]) - numpy.sqrt(r_coh2[5,:])).ravel()
        d = sorted(d)
        d = [x for x in d if str(x) != 'nan']
        #d_mu = numpy.mean(numpy.abs(r_coha - r_coh))
        #d_sig = numpy.std(numpy.abs(r_coha - r_coh))
        print('mean difference: ', numpy.mean(d))
        print('std difference: ', numpy.std(d))
        print('coh2 95 pct difference: ', d[int(0.95*len(d))])
        print('coh2 max difference: ', max(d))
        print('n: ', len(d))

if __name__ == '__main__':
    main()
