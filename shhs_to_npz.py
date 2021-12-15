import numpy as np
import glob
import pyedflib
import xml.etree.ElementTree as ET
import random
import argparse
import resampy
import os
import tqdm
from sklearn.preprocessing import StandardScaler

def load_from_edf(path, n_to_load=False, resampling=True):
    # all_edf_name = sorted(glob.glob('../shhs_data/edf/shhs1-200*.edf'))[:n_to_load]
    all_edf_name = sorted(glob.glob(os.path.join(path, 'edf/shhs1-200*.edf')))[:n_to_load]
    random.seed(0)
    random.shuffle(all_edf_name) # shuffle for imbalance
    
    len_epoch = 3600 if resampling else 3750
    eeg_train = np.empty((0,len_epoch))
    eeg_val = np.empty((0, len_epoch))
    eeg_test = np.empty((0, len_epoch))
    
    # all_prof_name = sorted(glob.glob('../shhs_data/profusion/shhs1-200*-profusion.xml'))[:n_to_load]
    all_prof_name = sorted(glob.glob(os.path.join(path, 'profusion/shhs1-200*-profusion.xml')))[:n_to_load]
    random.seed(0)
    random.shuffle(all_prof_name) # shuffle for imbalance
    stage_train = np.empty((0,))
    stage_val = np.empty((0,))
    stage_test = np.empty((0,))
    
    n_data = 0

    bar_load = tqdm.tqdm(total=n_to_load, desc='loading   ')
    if resampling: bar_resample = tqdm.tqdm(total=n_to_load, desc='resampling')
    for i, (edf_name, prof_name) in enumerate(zip(all_edf_name, all_prof_name)):
        bar_load.update(1)
        bar_load.set_postfix({'edf': os.path.basename(edf_name), 'prof': os.path.basename(prof_name)})
        edf = pyedflib.EdfReader(edf_name)
        eeg = edf.readSignal(7)

        # standardization by record
        eeg = eeg.reshape((-1, 1))
        scaler = StandardScaler()
        eeg = scaler.fit_transform(eeg)
        
        eeg = eeg.reshape((-1, 3750))

        if resampling:
            eeg = resampy.resample(eeg, 125, 120)
            bar_resample.update(1)
            bar_resample.set_postfix({'edf': os.path.basename(edf_name), 'prof': os.path.basename(prof_name)})
        
        tree = ET.parse(prof_name)
        root = tree.getroot()
        stage = np.array([int(e.text) for e in root.findall('SleepStages/SleepStage')])
        stage[stage==4] = 3
        stage[stage>4] = 4
        
        n_data += eeg.shape[0]
        if i < 0.6*len(all_edf_name):
            eeg_train = np.concatenate([eeg_train, eeg])
            stage_train = np.concatenate([stage_train, stage])
        elif i < 0.8*len(all_edf_name):
            eeg_val = np.concatenate([eeg_val, eeg])
            stage_val = np.concatenate([stage_val, stage])
        else:
            eeg_test = np.concatenate([eeg_test, eeg])
            stage_test = np.concatenate([stage_test, stage])
        edf.close()
    bar_load.close()
    if resampling: bar_resample.close()

    print('  =>   eeg:', eeg_train.shape, eeg_val.shape, eeg_test.shape)
    print('  => stage:', stage_train.shape, stage_val.shape, stage_test.shape)
    print('Data ratio:', eeg_train.shape[0]/n_data, eeg_val.shape[0]/n_data, eeg_test.shape[0]/n_data)

    return eeg_train, eeg_val, eeg_test, stage_train, stage_val, stage_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load eeg and label from SHHS files.')
    parser.add_argument('-path', default='./', help='path to the files')
    parser.add_argument('-n', default=5, help='the number to load')
    parser.add_argument('--resampling', action='store_true', help='whether resample to 120 Hz or not')
    parser.add_argument('--save', action='store_true', help='save the results to .npz file')
    args = parser.parse_args()
    
    n_to_load = int(args.n)
    eeg_train, eeg_val, eeg_test, stage_train, stage_val, stage_test = load_from_edf(args.path, n_to_load=n_to_load, resampling=args.resampling)

    Fs = 120 if args.resampling else 125
    if args.save:
        np.savez(f'eeg_n{n_to_load}_Fs{Fs}', train=eeg_train, val=eeg_val, test=eeg_test)
        np.savez(f'stage_n{n_to_load}', train=stage_train, val=stage_val, test=stage_test)