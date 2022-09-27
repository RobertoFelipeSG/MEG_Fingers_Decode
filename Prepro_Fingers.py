import os
import time
import numpy as np
import scipy as sp
import mne
import autoreject
import matplotlib.pyplot as plt
from scipy import stats

def Prepro_Fingers (path, sub):

    start_time = time.time()

    raw = mne.io.read_raw_ctf(path, preload=True, system_clock='ignore')
    if 'UPPT001' not in raw.ch_names:
        print('No triggers in', path, sub)

    ### Filtering
    raw.filter(0.1, 150, method='iir')

    ### Find events of interest
    events = mne.find_events(raw, stim_channel='UPPT001', min_duration=1/raw.info['sfreq'], shortest_event=1)

    event_dict = {'little/on':1, 'ring/on':2, 'middle/on':3, 'index/on':4,
             'little/off':5, 'ring/off':6, 'middle/off':7, 'index/off':8}

    ### Set MEG layout
    layout = mne.channels.read_layout('CTF275.lay')
    layout.names = [name+'-1609' for name in layout.names]
    print(layout.names[1])

    ### Make montage out of the layout
    mont = dict(zip(layout.names,layout.pos[:,:3]))
    montage = mne.channels.make_dig_montage(ch_pos=mont, coord_frame='ctf_meg')

    raw.drop_channels = list((set(raw.ch_names)^set(montage.ch_names)))
    print(len(raw.ch_names))

    raw.set_montage(montage, on_missing='ignore')

    ### Pick only the Mag and drop the other channels
    picks = mne.pick_types(raw.info, meg=True, eeg=False, misc=False, ref_meg=False)
    print(len(picks))

    ### Epoching 
    epochs_1 = mne.Epochs(raw, events=events, preload=True, picks=picks, event_id=event_dict, baseline=None) 

    ### Resampling
    epochs = epochs_1.copy().resample(200, npad='auto', n_jobs=-1)
    print('New sampling rate:', epochs.info['sfreq'], 'Hz')
    
    ### Artifact rejection
    ### Autoreject must be applied after filtering to not lose data
    ar = autoreject.AutoReject(random_state=11, n_jobs=1)
    ar.fit(epochs) #All epochs @ once?
    ep_ar, reject_log = ar.transform(epochs, return_log=True)

    ### ICA
    ica = mne.preprocessing.ICA(method="fastica", random_state=99, n_components=90)
    ica.fit(ep_ar)
    ica.plot_sources(ep_ar, block=True)
    ica.apply(ep_ar)

    print("fatto")

    preprocessed = ep_ar
    preprocessed.save('FingId_%s_prepro_epo.fif' %sub, overwrite=True)

    print("---%s seconds ---" %(time.time() - start_time))

        #Return output
    return preprocessed