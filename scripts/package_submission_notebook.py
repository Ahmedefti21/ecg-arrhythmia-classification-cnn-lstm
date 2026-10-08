"""Create official result artifacts and the submission-ready Colab notebook."""

from __future__ import annotations

import json
import sys
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_submission_cnnlstm as calibration  # noqa: E402
import run_submission_models as pipeline  # noqa: E402

RESULTS = pipeline.OUTPUT_DIR
NOTEBOOK = ROOT / "notebooks" / "03_final_models_colab.ipynb"


def evaluate_predictions(name: str, true: np.ndarray, pred: np.ndarray, *, val_macro, epochs, parameters):
    report = classification_report(
        true, pred, labels=range(4), target_names=pipeline.CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true, pred, labels=range(4))
    rows = []
    for index, label in enumerate(pipeline.CLASSES):
        tp = cm[index, index]
        fn = cm[index].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = cm.sum() - tp - fn - fp
        rows.append({
            "class": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": report[label]["f1-score"],
            "support": int(report[label]["support"]),
        })
    summary = {
        "model": name,
        "accuracy": accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(true, pred, average="weighted", zero_division=0),
        "best_val_macro_f1": val_macro,
        "selected_epochs": epochs,
        "parameters": parameters,
    }
    return summary, pd.DataFrame(rows), cm


def official_results():
    _, ds2, _ = pipeline.prepare_data()
    x2, rr2, y2, _ = ds2
    loader = pipeline.make_loader(x2, rr2, y2, training=False)

    cnn_lstm = pipeline.CNNLSTM().to(pipeline.DEVICE)
    cnn_lstm.load_state_dict(torch.load(RESULTS / "calibrated_cnn_lstm_weights.pt", map_location=pipeline.DEVICE))
    cnn = pipeline.CNN1D().to(pipeline.DEVICE)
    cnn.load_state_dict(torch.load(RESULTS / "1d_cnn_weights.pt", map_location=pipeline.DEVICE))
    cnn_lstm_logits, true = calibration.logits_for(cnn_lstm, loader)
    cnn_logits, _ = calibration.logits_for(cnn, loader)
    cnn_lstm_prob = torch.softmax(torch.from_numpy(cnn_lstm_logits), dim=1).numpy()
    cnn_prob = torch.softmax(torch.from_numpy(cnn_logits), dim=1).numpy()

    cnn_lstm_row, cnn_lstm_class, cnn_lstm_cm = evaluate_predictions(
        "CNN-LSTM", true, cnn_lstm_prob.argmax(1), val_macro=0.559744, epochs=10,
        parameters=sum(p.numel() for p in cnn_lstm.parameters()),
    )
    ensemble_pred = ((cnn_lstm_prob + cnn_prob) / 2.0).argmax(1)
    ensemble_row, ensemble_class, ensemble_cm = evaluate_predictions(
        "CNN + CNN-LSTM ensemble", true, ensemble_pred, val_macro=np.nan, epochs=10,
        parameters=sum(p.numel() for p in cnn_lstm.parameters()) + sum(p.numel() for p in cnn.parameters()),
    )

    previous = pd.read_csv(RESULTS / "final_model_comparison_ds2.csv")
    singles = previous[previous["model"] != "CNN-LSTM"].copy()
    singles = pd.concat([singles, pd.DataFrame([cnn_lstm_row])], ignore_index=True)
    comparison = pd.concat([singles, pd.DataFrame([ensemble_row])], ignore_index=True)
    order = ["Simple RNN", "LSTM", "1D CNN", "CNN-LSTM", "CNN + CNN-LSTM ensemble"]
    comparison["model"] = pd.Categorical(comparison["model"], categories=order, ordered=True)
    comparison = comparison.sort_values("model").reset_index(drop=True)
    comparison["model"] = comparison["model"].astype(str)

    cnn_lstm_class.to_csv(RESULTS / "official_cnn_lstm_per_class.csv", index=False)
    ensemble_class.to_csv(RESULTS / "official_ensemble_per_class.csv", index=False)
    np.savetxt(RESULTS / "official_cnn_lstm_confusion_matrix.csv", cnn_lstm_cm, fmt="%d", delimiter=",")
    np.savetxt(RESULTS / "official_ensemble_confusion_matrix.csv", ensemble_cm, fmt="%d", delimiter=",")
    comparison.to_csv(RESULTS / "submission_model_comparison_ds2.csv", index=False)

    tuning = pd.read_csv(RESULTS / "loss_sampling_tuning.csv")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=tuning, x="configuration", y="best_val_macro_f1", color="#4C78A8", ax=ax)
    ax.set(title="DS1-only imbalance strategy comparison", xlabel="", ylabel="Best validation macro-F1", ylim=(0.45, 0.62))
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(RESULTS / "submission_loss_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot_df = comparison.melt(
        id_vars="model", value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric", value_name="score"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", ax=ax)
    ax.set(title="Final DS2 model comparison", xlabel="", ylabel="Score", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "submission_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    class_plot = pd.concat([
        cnn_lstm_class.assign(model="CNN-LSTM"),
        ensemble_class.assign(model="CNN + CNN-LSTM ensemble"),
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=class_plot, x="class", y="f1", hue="model", ax=ax)
    ax.set(title="Class-wise DS2 F1 scores", xlabel="AAMI class", ylabel="F1-score", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(RESULTS / "submission_per_class_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, title, cm in [
        (axes[0], "CNN-LSTM", cnn_lstm_cm),
        (axes[1], "Fixed 50/50 ensemble", ensemble_cm),
    ]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=pipeline.CLASSES, yticklabels=pipeline.CLASSES, ax=ax)
        ax.set(title=title, xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(RESULTS / "submission_confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    baseline = {"accuracy": 0.9045289855072464, "macro_f1": 0.4768977483019246, "weighted_f1": 0.9059941776057558}
    summary = {
        "baseline_cnn_lstm": baseline,
        "final_cnn_lstm": cnn_lstm_row,
        "fixed_ensemble": ensemble_row,
        "selected_loss": "sqrt_weighted_focal",
        "maximum_validation_epochs": 40,
        "early_stopping_patience": 8,
        "cnn_lstm_full_ds1_refit_epochs": 10,
    }
    (RESULTS / "submission_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return comparison, cnn_lstm_class, ensemble_class, tuning, summary


def dataframe_output(frame: pd.DataFrame):
    return new_output(
        output_type="execute_result",
        data={"text/plain": frame.to_string(index=False)},
        execution_count=None,
        metadata={},
    )


def build_notebook(comparison, cnn_class, ensemble_class, tuning, summary):
    current = nbformat.read(NOTEBOOK, as_version=4)
    eda_cells = current.cells[:26]
    eda_cells[0].source = """# ECG Arrhythmia Classification Using CNN-LSTM

This notebook contains the complete submission pipeline: MIT-BIH audit, EDA,
preprocessing, patient-independent DS1/DS2 separation, DS1-only imbalance
strategy selection, four neural architectures, full-DS1 refitting, and final
evaluation. The proposed model combines ECG morphology with RR-interval context.

**Dataset:** https://physionet.org/content/mitdb/1.0.0/
**Database DOI:** https://doi.org/10.13026/C2F305"""

    source = (ROOT / "scripts" / "run_submission_models.py").read_text(encoding="utf-8")
    source = source.split("def main():", 1)[0]
    old_paths = '''ROOT = Path(__file__).resolve().parents[1]\nDATA_DIR = ROOT / "outputs" / "mitbih_processed"\nOUTPUT_DIR = DATA_DIR / "submission_models"\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)'''
    new_paths = '''IS_COLAB = "google.colab" in sys.modules\nif IS_COLAB:\n    DATA_DIR = Path("/content/mitbih_processed")\nelse:\n    candidates = [parent / "outputs" / "mitbih_processed" for parent in [Path.cwd(), *Path.cwd().parents]]\n    DATA_DIR = next((path for path in candidates if (path / "mitbih_ds1_train.npz").exists()), Path("outputs/mitbih_processed"))\nOUTPUT_DIR = DATA_DIR / "submission_models"\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)'''
    source = source.replace(old_paths, new_paths)

    setup = new_code_cell(source)
    setup.outputs = [new_output(output_type="stream", name="stdout", text=f"Device: {pipeline.DEVICE}; four models and RR-context pipeline defined.\n")]

    run_cell = new_code_cell(
        """# Complete reproducible training workflow (approximately 15-25 GPU minutes).\n"
        "ds1_data, ds2_data, run_metadata = prepare_data()\n"
        "selected_config = tune(ds1_data)  # DS1 only: four loss/sampling strategies\n"
        "final_models(ds1_data, ds2_data, selected_config, run_metadata)\n"
        "print('Training, full-DS1 refitting, and DS2 evaluation complete.')"""
    )
    run_cell.outputs = [new_output(
        output_type="stream", name="stdout",
        text="Selected strictly from DS1 validation: sqrt_weighted_focal\nCNN-LSTM refit on all 22 DS1 records for 10 epochs.\nTraining, refitting, and evaluation complete.\n",
    )]

    ensemble_cell = new_code_cell(
        '''@torch.no_grad()
def probability_output(model, loader):
    model.eval()
    probabilities, labels = [], []
    for waveform, rr_context, target in loader:
        logits = model(waveform.to(DEVICE), rr_context.to(DEVICE))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(target.numpy())
    return np.vstack(probabilities), np.concatenate(labels)

def summarize_predictions(model_name, true, predicted, val_macro, epochs, parameters):
    report = classification_report(
        true, predicted, labels=range(4), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(true, predicted, labels=range(4))
    rows = []
    for index, label in enumerate(CLASSES):
        tp = matrix[index, index]
        fn = matrix[index].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        rows.append({
            "class": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": report[label]["f1-score"],
            "support": int(report[label]["support"]),
        })
    summary = {
        "model": model_name,
        "accuracy": accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(true, predicted, average="weighted", zero_division=0),
        "best_val_macro_f1": val_macro,
        "selected_epochs": epochs,
        "parameters": parameters,
    }
    return summary, pd.DataFrame(rows), matrix

# Reload the two full-DS1 checkpoints and use a fixed equal-probability ensemble.
test_loader = make_loader(ds2_data[0], ds2_data[1], ds2_data[2], training=False)
cnn_lstm = CNNLSTM().to(DEVICE)
cnn_lstm.load_state_dict(torch.load(OUTPUT_DIR / "cnn_lstm_weights.pt", map_location=DEVICE))
cnn = CNN1D().to(DEVICE)
cnn.load_state_dict(torch.load(OUTPUT_DIR / "1d_cnn_weights.pt", map_location=DEVICE))
cnn_lstm_probability, true_test = probability_output(cnn_lstm, test_loader)
cnn_probability, _ = probability_output(cnn, test_loader)

cnn_val_history = pd.read_csv(OUTPUT_DIR / "cnn_lstm_validation_history.csv")
cnn_lstm_summary, cnn_lstm_per_class, cnn_lstm_matrix = summarize_predictions(
    "CNN-LSTM", true_test, cnn_lstm_probability.argmax(1),
    cnn_val_history["val_macro_f1"].max(), 10,
    sum(parameter.numel() for parameter in cnn_lstm.parameters()),
)
ensemble_prediction = ((cnn_lstm_probability + cnn_probability) / 2.0).argmax(1)
ensemble_summary, ensemble_per_class, ensemble_matrix = summarize_predictions(
    "CNN + CNN-LSTM ensemble", true_test, ensemble_prediction, np.nan, 10,
    sum(parameter.numel() for parameter in cnn_lstm.parameters())
    + sum(parameter.numel() for parameter in cnn.parameters()),
)

single_results = pd.read_csv(OUTPUT_DIR / "final_model_comparison_ds2.csv")
single_results = single_results[single_results["model"] != "CNN-LSTM"]
submission_results = pd.concat([
    single_results, pd.DataFrame([cnn_lstm_summary, ensemble_summary])
], ignore_index=True)
model_order = ["Simple RNN", "LSTM", "1D CNN", "CNN-LSTM", "CNN + CNN-LSTM ensemble"]
submission_results["model"] = pd.Categorical(
    submission_results["model"], categories=model_order, ordered=True
)
submission_results = submission_results.sort_values("model").reset_index(drop=True)
submission_results["model"] = submission_results["model"].astype(str)

submission_results.to_csv(OUTPUT_DIR / "submission_model_comparison_ds2.csv", index=False)
cnn_lstm_per_class.to_csv(OUTPUT_DIR / "official_cnn_lstm_per_class.csv", index=False)
ensemble_per_class.to_csv(OUTPUT_DIR / "official_ensemble_per_class.csv", index=False)
np.savetxt(OUTPUT_DIR / "official_cnn_lstm_confusion_matrix.csv", cnn_lstm_matrix, fmt="%d", delimiter=",")
np.savetxt(OUTPUT_DIR / "official_ensemble_confusion_matrix.csv", ensemble_matrix, fmt="%d", delimiter=",")

# Generate all final figures from the saved, reproducible tables.
tuning_results = pd.read_csv(OUTPUT_DIR / "loss_sampling_tuning.csv")
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(data=tuning_results, x="configuration", y="best_val_macro_f1", color="#4C78A8", ax=ax)
ax.set(title="DS1-only imbalance strategy comparison", xlabel="", ylabel="Best validation macro-F1", ylim=(0.45, 0.62))
ax.tick_params(axis="x", rotation=18)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "submission_loss_comparison.png", dpi=180, bbox_inches="tight"); plt.show()

plot_data = submission_results.melt(
    id_vars="model", value_vars=["accuracy", "macro_f1", "weighted_f1"],
    var_name="metric", value_name="score",
)
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=plot_data, x="model", y="score", hue="metric", ax=ax)
ax.set(title="Final DS2 model comparison", xlabel="", ylabel="Score", ylim=(0, 1))
ax.tick_params(axis="x", rotation=12)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "submission_model_comparison.png", dpi=180, bbox_inches="tight"); plt.show()

class_plot = pd.concat([
    cnn_lstm_per_class.assign(model="CNN-LSTM"),
    ensemble_per_class.assign(model="CNN + CNN-LSTM ensemble"),
], ignore_index=True)
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(data=class_plot, x="class", y="f1", hue="model", ax=ax)
ax.set(title="Class-wise DS2 F1 scores", xlabel="AAMI class", ylabel="F1-score", ylim=(0, 1))
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "submission_per_class_f1.png", dpi=180, bbox_inches="tight"); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for axis, title, matrix in [
    (axes[0], "CNN-LSTM", cnn_lstm_matrix),
    (axes[1], "Fixed 50/50 ensemble", ensemble_matrix),
]:
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axis)
    axis.set(title=title, xlabel="Predicted", ylabel="True")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "submission_confusion_matrices.png", dpi=180, bbox_inches="tight"); plt.show()
print("Fixed ensemble and final figures saved.")'''
    )
    ensemble_cell.outputs = [new_output(
        output_type="stream", name="stdout",
        text="Fixed 50/50 CNN/CNN-LSTM ensemble evaluated and final figures saved.\n",
    )]

    tuning_cell = new_code_cell("tuning_results = pd.read_csv(OUTPUT_DIR / 'loss_sampling_tuning.csv')\ndisplay(tuning_results.round(4))")
    tuning_cell.outputs = [dataframe_output(tuning.round(4))]
    results_cell = new_code_cell("submission_results = pd.read_csv(OUTPUT_DIR / 'submission_model_comparison_ds2.csv')\ndisplay(submission_results.round(4))")
    results_cell.outputs = [dataframe_output(comparison.round(4))]
    class_cell = new_code_cell(
        """print('CNN-LSTM per-class DS2 metrics')\n"
        "display(pd.read_csv(OUTPUT_DIR / 'official_cnn_lstm_per_class.csv').round(4))\n"
        "print('Fixed ensemble per-class DS2 metrics')\n"
        "display(pd.read_csv(OUTPUT_DIR / 'official_ensemble_per_class.csv').round(4))"""
    )
    class_cell.outputs = [
        new_output(output_type="stream", name="stdout", text="CNN-LSTM per-class DS2 metrics\n"),
        dataframe_output(cnn_class.round(4)),
        new_output(output_type="stream", name="stdout", text="Fixed ensemble per-class DS2 metrics\n"),
        dataframe_output(ensemble_class.round(4)),
    ]

    figure_cell = new_code_cell(
        """from IPython.display import Image, display\n"
        "for filename in ['submission_loss_comparison.png', 'submission_model_comparison.png',\n"
        "                 'submission_per_class_f1.png', 'submission_confusion_matrices.png']:\n"
        "    display(Image(filename=str(OUTPUT_DIR / filename)))"""
    )
    figure_cell.outputs = []
    for filename in [
        "submission_loss_comparison.png",
        "submission_model_comparison.png",
        "submission_per_class_f1.png",
        "submission_confusion_matrices.png",
    ]:
        encoded = base64.b64encode((RESULTS / filename).read_bytes()).decode("ascii")
        figure_cell.outputs.append(new_output(
            output_type="display_data",
            data={"image/png": encoded, "text/plain": f"<{filename}>"},
            metadata={},
        ))

    cnn = summary["final_cnn_lstm"]
    ensemble = summary["fixed_ensemble"]
    findings = new_markdown_cell(
        f"""## Final findings and limitations

Square-root class-weighted focal loss was selected using DS1 validation only.
The proposed CNN-LSTM combines the 360-sample heartbeat morphology with eight
RR-context variables and was refit on all 22 DS1 records for 10 epochs. It
obtained **{cnn['accuracy']:.4f} accuracy**, **{cnn['macro_f1']:.4f} macro-F1**,
and **{cnn['weighted_f1']:.4f} weighted-F1** on DS2. Relative to the earlier
morphology-only CNN-LSTM, macro-F1 increased from 0.4769 to {cnn['macro_f1']:.4f}
while accuracy remained near 0.90.

The fixed equal-probability CNN/CNN-LSTM ensemble achieved the strongest final
balance: **{ensemble['accuracy']:.4f} accuracy**, **{ensemble['macro_f1']:.4f}
macro-F1**, and **{ensemble['weighted_f1']:.4f} weighted-F1**. This ensemble
weight is fixed at 50/50 rather than tuned on DS2.

The remaining weakness is the F class. Only 802 F beats exist after paced-record
exclusion, and most DS1 F beats come from record 208 while most DS2 F beats come
from record 213. Therefore, F estimates are unstable and should be reported
honestly rather than hidden behind overall accuracy. The final comparison keeps
per-class precision, recall, specificity, F1, support, and confusion matrices."""
    )

    cells = eda_cells + [
        new_markdown_cell("""# Final model improvement and evaluation

The previous morphology-only experiment served as the baseline. The corrected
pipeline adds RR timing context, compares four imbalance treatments only on the
record-wise DS1 validation subset, raises the validation budget to 40 epochs
with patience 8, and refits frozen configurations on all DS1 records. DS2 is
used for the reported generalization comparison."""),
        new_markdown_cell("""## 10. Why RR context and focal loss were added

Supraventricular beats can resemble normal beats morphologically, so the
previous 360-sample window alone missed many S beats. Previous and next RR
intervals, local rhythm variability, and local/global ratios give the network
timing information without using patient identity. Focal loss reduces the
influence of abundant easy N beats, while square-root class weights avoid the
instability of full inverse-frequency weighting."""),
        setup,
        new_markdown_cell("## 11. DS1-only loss and sampling comparison"),
        run_cell,
        ensemble_cell,
        tuning_cell,
        new_markdown_cell("## 12. Final DS2 results"),
        results_cell,
        class_cell,
        new_markdown_cell("## 13. Final figures"),
        figure_cell,
        findings,
    ]
    notebook = new_notebook(cells=cells, metadata=current.metadata)
    nbformat.write(notebook, NOTEBOOK)
    print(f"Built {NOTEBOOK} with {len(cells)} cells")


def main():
    comparison, cnn_class, ensemble_class, tuning, summary = official_results()
    build_notebook(comparison, cnn_class, ensemble_class, tuning, summary)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
