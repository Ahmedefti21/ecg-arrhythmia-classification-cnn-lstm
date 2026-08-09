# ECG Arrhythmia Classification Using CNN-LSTM

Deep learning project for beat-level ECG arrhythmia classification using a CNN
for local waveform feature extraction and an LSTM for temporal feature modeling.
The project uses the open MIT-BIH Arrhythmia Database from PhysioNet.

## Current project update

- [`notebooks/01_initial_eda_preprocessing.ipynb`](notebooks/01_initial_eda_preprocessing.ipynb)
  is the Colab-ready Week 4 notebook. It audits all 48 record headers and
  annotations, visualizes the data, maps beat symbols into five preliminary
  AAMI-style groups, and validates the initial preprocessing pipeline on five
  representative records.
- [`reports/Project_Update_1_Report.docx`](reports/Project_Update_1_Report.docx)
  contains the two-page project introduction, objectives, eight-paper
  literature review, and dataset description.
- `scripts/` contains reproducible builders for the notebook and report.

No CNN-LSTM model has been trained in this update. Model implementation and
evaluation follow after the preprocessing and record-wise split protocol are
finalized.

## Running the notebook in Google Colab

1. Open the notebook from the `notebooks/` directory in Google Colab.
2. Run the first installation cell to install `wfdb`.
3. Select **Runtime > Run all**.

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
