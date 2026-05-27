# Supplementary Data: PDZ2 Conformational Selection

This repository contains the explicit featurized datasets used to generate the Markov State Models (MSMs), Random Forest Classifications and HDBSCAN Clusters presented in the manuscript. In addition, th
folder "Hdbscan Medoid Structures" contains the centroid structures of the Apo and Bound clusters respectively.

**Requirements:** Python 3.8+, `numpy`, `pickle`

## General Usage
All data files are serialized Python dictionaries (`.pkl`). You can load and inspect any file using the following standard template:

```python
import pickle
import numpy as np

# Replace with the target file name
file_name = 'data1.pkl'

with open(file_name, 'rb') as f:
    data = pickle.load(f)

# View available data matrices
print(data.keys())

## File Index

### 1. `data1.pkl` (Feature Indices)
Explicit C_alpha-C_alpha residue contact pairs used to featurize the unstrided molecular dynamics trajectories prior to Time-lagged Independent Component Analysis (TICA).
* `apo_features`: (196, 2) array of integer residue indices for the apo ensemble.
* `bound_features`: (199, 2) array of integer residue indices for the bound ensemble.
* `unified_features`: (219, 2) array representing the mathematical union of both sets, used for Random Forest classification.

### 2. `data2.pkl` (Apo Trajectory Features)
Time-series evaluations of the 196 apo-specific C_alpha-C_alpha distance features calculated across all 49 unstrided apo molecular dynamics trajectories.
* `apo_traj_features`: A Python list containing 49 `numpy.ndarray` objects. Each array corresponds to a single trajectory and has the shape `(n_frames, 196)`.

### 2. `data3.pkl` (Bound Trajectory Features)
Time-series evaluations of the 199 bound-specific C_alpha-C_alpha distance features calculated across all 49 unstrided bound molecular dynamics trajectories.
* `bound_traj_features`: A Python list containing 49 `numpy.ndarray` objects. Each array corresponds to a single trajectory and has the shape `(n_frames, 199)`.

