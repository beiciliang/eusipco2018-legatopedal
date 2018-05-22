# eusipco2018-legatopedal

Companion code for the paper:

Beici Liang, György Fazekas, Mark Sandler. "Piano Legato-Pedal Onset Detection Based on a Sympathetic Resonance Measure​", in Proceedings of the 26th European Signal Processing Conference (EUSIPCO), 2018. (accepted)

## Index

* [`get_features.py`](eusipco2018-legatopedal/get_features.py): 
Main python script to extract features from the example audio file in [`input/chopin`](eusipco2018-legatopedal/input/chopin). 
Save the features as a `.npz` file in [`features`](eusipco2018-legatopedal/features).

* [`evaluate.py`](eusipco2018-legatopedal/evaluate.py): 
Python script to run the task of `aic` or `logr` based on `features/chopin_features.npz` (extracted features) and `input/chopin/chopin_gt.npy` (ground truth).
If set the task as `logr_eusipco`, return the performance matrix of logistic regression model using the data in [`eusipco-data`](eusipco2018-legatopedal/eusipco-data). 
The matrix corresponds to the table of experiment result in the paper.

* [`input`](eusipco2018-legatopedal/input):
The folder contains `.csv` files of f0 and inharmonicity coefficient computed from 88 piano notes (check this [repository](https://github.com/beiciliang/estimate-f0-inharmonicity) for calculation details). 
The sub-folder [`input/chopin`](eusipco2018-legatopedal/input/chopin) includes example audio file `chopin.wav`, ground truth `chopin_gt.npy` obtained from `chopin.mid`, and transcription result `chopin_transcription.npy` (check this [repository](https://github.com/beiciliang/modelAttackDecay-for-piano-transcription) for transcription details).

## Requirements

Codes are based on the following softwares and their corresponding versions. 

Software | Version
------------ | -------------
Python | 2.7.14 64bit
IPython | 5.3.0
OS | Darwin 15.6.0 x86_64 i386 64bit

Other dependencies can be installed by
```
pip install -r requirements.txt
```