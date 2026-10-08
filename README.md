# ECG Arrhythmia Classification Using CNN-LSTM

Deep learning project for beat-level ECG arrhythmia classification using a CNN
for local waveform feature extraction and an LSTM for temporal feature modeling.
The project uses the open MIT-BIH Arrhythmia Database from PhysioNet.

## Project overview

This project implements and evaluates five neural architectures (Simple RNN,
LSTM, 1D CNN, CNN-LSTM, and Transformer) for classifying heartbeats into the
four primary AAMI classes (N, S, V, F) using the standard inter-patient DS1/DS2
protocol from de Chazal et al.

### Key features

- **Inter-patient DS1/DS2 split**: 22 training records, 22 disjoint test records,
  4 paced records excluded — no data leakage between patients.
- **RR-interval context**: Eight timing features (previous/next RR intervals,
  local rhythm statistics, local/global ratios) supplement the 360-sample
  morphology window.
- **Class imbalance handling**: Four loss/sampling strategies compared on DS1
  validation only. Square-root class-weighted cross-entropy was selected by
  validation macro-F1 for the final benchmark.
- **Post-hoc calibration**: Temperature scaling and per-class threshold
  calibration learned on DS1 validation to improve minority-class recall.
- **5-model ensemble**: All five architectures combined with temperature-scaled
  probability averaging and jointly calibrated thresholds.

### Notebooks

- [`notebooks/01_initial_eda_preprocessing.ipynb`](notebooks/01_initial_eda_preprocessing.ipynb)
  — EDA, signal processing, and preprocessing pipeline.
- [`cse427_final_submission.ipynb`](cse427_final_submission.ipynb)
  — **Final submission notebook**: complete pipeline from data loading through
  five-model training, calibration, ensemble construction, DS2 evaluation, and
  the final figures. The committed copy includes the executed outputs.

### Final DS2 results

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Simple RNN | 0.8554 | 0.4319 | 0.8711 |
| LSTM | 0.8381 | 0.4229 | 0.8576 |
| 1D CNN | 0.8972 | **0.5304** | 0.9131 |
| CNN-LSTM | 0.8911 | 0.5193 | 0.9062 |
| Transformer | **0.9296** | 0.5097 | **0.9226** |
| 5-model ensemble | 0.9255 | 0.4821 | 0.9161 |

## Running the notebook in Google Colab

1. Open `cse427_final_submission.ipynb` in Google Colab.
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU).
3. Run the first cell to install `wfdb`.
4. Select **Runtime > Run all** (approximately 20–40 GPU minutes).

The notebook reads records directly from PhysioNet with `pn_dir="mitdb"`, so a
manual dataset upload is not required.

## Dataset policy

Do not commit the raw MIT-BIH database to this repository. It is approximately
104 MB, is already publicly hosted by PhysioNet, and would unnecessarily enlarge
every clone and the permanent Git history.

If a local copy is needed, place it under one of these ignored paths:

```text
data/mitdb/
datasets/mitdb/
mitdb/
```

The `.gitignore` also excludes WFDB waveform/annotation files, processed NumPy
arrays, model checkpoints, and temporary outputs. Commit source code, notebooks,
small figures, reports, configuration, and documentation instead.

Dataset source: [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/)
