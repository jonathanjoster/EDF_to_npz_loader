# EDF loader to save to .npz file

Get .npz file for eeg and labels.

## Usage

```
python shhs_to_npz.py -path [path_to_dir] -n [num_to_load] --resampling --save
```

Note that specified path contains `edf/` and `profusion/` directories.
If resampling flag is set, it takes much time to calculate.

## Recommeded file composition

```
EDF_to_npz_loader
 ├ edf/
 ├ profusion/
 └ shhs_to_npz.py
 ```