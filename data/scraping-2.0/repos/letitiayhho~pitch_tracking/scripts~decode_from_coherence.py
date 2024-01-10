#!/usr/bin/env python3

#SBATCH --time=00:10:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=24G
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/decode_from_coherence_%j.log

import sys
import numpy as np
import pickle
from mne_bids import BIDSPath, read_raw_bids
from bids import BIDSLayout
from util.io.coherence import *
from util.io.iter_BIDSPaths import *
from mne_connectivity import spectral_connectivity_time, check_indices
from scipy.signal import coherence

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import SlidingEstimator, cross_val_multiscore, Scaler, Vectorizer

def main(SUB, TASK, RUN):
    FPATH = f'/project2/hcn1/pitch_tracking/data/bids/derivatives/preprocessing/sub-{SUB}/sub-{SUB}_task-{TASK}_run-{RUN}_res-hi_desc-clean_epo.fif.gz'

    BIDS_ROOT = '../data/bids'
    DECODING_ROOT = '../data/bids/derivatives/decoding'
    FIGS_ROOT = '../figs'

    DERIV_ROOT = '../data/bids/derivatives'
    METHOD = 'coh'
    FS = 5000
    RAW_TMIN = -0.2
    RAW_TMAX = 0.5
    TMIN = 0
    TMAX = 0.25
    N_CHANS = 62
    CONDS = ['50', '100', '150', '200', '250']
    FREQS = [50, 100, 150, 200, 250]
    
    # Load epoched data
    epochs = mne.read_epochs(FPATH, preload = True)
    events = epochs.events
    n_epochs = len(events)
    
    # Use a different sub for generating stim channels if sub has bad Aux channel
    STIM_SUB, STIM_RUN = get_stim_sub(SUB, RUN)
    
    # Create epochs from raw data to create simulated stim channels
    raw_epochs = get_raw_epochs(BIDS_ROOT, STIM_SUB, TASK, STIM_RUN)
    stim_epochs_array = create_stim_epochs_array(raw_epochs, n_epochs, CONDS)
    simulated_epochs = create_stim_epochs_object(stim_epochs_array, events, CONDS, FS, RAW_TMIN)
    
    # Crop data so both epoch objects have same windowing
    # epochs is (4801, 62, 1251) which is (epochs, channels, time points)
    # simulated epochs is (5, 1251) which is (freqs, time points)

    simulated_epochs = simulated_epochs.crop(tmin = TMIN, tmax = TMAX)
    epochs = epochs.crop(tmin = TMIN)

    # Keep only one version of stim signals
    stim_array = simulated_epochs.get_data()[0, :, :]

    # Create target array
    labels = pd.Series(events[:, 2])
    y = labels.replace({10001 : 0, 10002 : 1, 10003 : 2, 10004 : 3, 10005 : 4})
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # What shape does it need to be for decoding?? Keep this in mind!
    # Shape of Zxx for stft is n_epochs, n_freqs*n_chans, n_windows
    # Shape of Zxx in general is a trials, features, time array
    # Shape of Zxx in this case is n_epochs, n_chans, n_stim, n_freqs

    # Compute coherence
    epoch_array = epochs.get_data()
    n_epochs = np.shape(epoch_array)[0]
    n_chans = np.shape(epoch_array)[1]
    n_freq = 5

    X = np.zeros((n_epochs, n_chans, n_freq))

    for epoch in range(n_epochs):
        epoch_X = []
        for channel in range(n_chans):

            # get signal for current epoch
            current_epoch = epoch_array[epoch, channel, :]

            # get signal for stim for current epoch
            stim_index = y[epoch]
            current_stim = stim_array[stim_index, :]

            # compute coherence
            f, Cxy = coherence(current_epoch, current_stim, FS)

            # get coherences for condition frequencies only
            Cxy = extract_coherence_for_condition_frequencies(FREQS, f, Cxy) 

            # add to array
            X[epoch, channel, ] = Cxy

    save_fp = f'../data/bids/derivatives/coherence/sub-{SUB}_task-{TASK}_run-{RUN}_coh-for-each-stim-by-epoch.pkl'
    np.save(save_fp, X)
    
    
    # Decode
    n_stimuli = 5
    metric = 'accuracy'

    clf = make_pipeline(Vectorizer(),
                        LogisticRegression(solver='liblinear'))

    print("Creating sliding estimators")
    time_decod = SlidingEstimator(clf)

    print("Fit estimators")
    scores = cross_val_multiscore(
        time_decod,
        X, # a trials x features x freq
        y, # an (n_trials,) array of integer condition labels
        cv = 5, # use stratified 5-fold cross-validation
        n_jobs = -1, # use all available CPU cores
    )
    scores = np.mean(scores, axis = 0) # average across cv splits
    
    save_fp = f'{DECODING_ROOT}/sub-{SUB}/sub-{SUB}_task-{TASK}_run-{RUN}_desc-coh-for-each-stim_scores.npy'
    print(f'Saving scores to: {save_fp}')
    np.save(save_fp, scores)
    
    # Plot
    fig, ax = plt.subplots()
    ax.plot(range(len(scores)), scores, label = 'score')
    ax.axhline(1/n_stimuli, color = 'k', linestyle = '--', label = 'chance')
    ax.set_xlabel('Stim')
    ax.set_ylabel(metric)  # Area Under the Curve
    ax.legend()
    ax.set_title('Sensor space decoding')
    
    # Save plot
    fig_fpath = FIGS_ROOT + '/sub-' + SUB + '_' + 'task-pitch_' + 'run-' + RUN + '_coh-for-each-stim' + '.png'
    print('Saving figure to: ' + fig_fpath)
    plt.savefig(fig_fpath)
    
__doc__ = "Usage: ./decode_from_coherence.py <sub> <task> <run>"

if __name__ == "__main__":
    print(len(sys.argv))
    print("Argument List:", str(sys.argv))
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit("Incorrect call to script")
    print("Reading args")
    SUB = sys.argv[1]
    TASK = sys.argv[2]
    RUN = sys.argv[3]
    main(SUB, TASK, RUN)
