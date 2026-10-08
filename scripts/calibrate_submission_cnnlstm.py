"""Build the calibrated final CNN-LSTM using DS1-only decisions."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import run_submission_models as pipeline


@torch.no_grad()
def logits_for(model, loader):
    model.eval()
    logits, labels = [], []
    for x, rr, y in loader:
        logits.append(model(x.to(pipeline.DEVICE), rr.to(pipeline.DEVICE)).cpu().numpy())
        labels.append(y.numpy())
    return np.vstack(logits), np.concatenate(labels)


def select_biases(logits: np.ndarray, labels: np.ndarray):
    """Select additive class-logit biases on DS1 validation only."""
    best = None
    # N is the reference class. A moderately wide search corrects focal-loss
    # probability distortion without changing learned ECG representations.
    values = np.arange(-1.5, 1.51, 0.15)
    for s_bias in values:
        for v_bias in values:
            for f_bias in values:
                bias = np.asarray([0.0, s_bias, v_bias, f_bias], dtype=np.float32)
                prediction = (logits + bias).argmax(1)
                accuracy = accuracy_score(labels, prediction)
                if accuracy < 0.92:
                    continue
                macro = f1_score(labels, prediction, average="macro", zero_division=0)
                score = macro + 0.10 * accuracy
                candidate = (score, macro, accuracy, bias)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        raise RuntimeError("No calibration candidate met the validation accuracy guard")
    return best[3], best[1], best[2]


def train_fixed(model, config, data, epochs):
    x, rr, y = data
    loader = pipeline.make_loader(x, rr, y, training=True, sampler_power=config.sampler_power)
    model = model.to(pipeline.DEVICE)
    criterion = pipeline.make_criterion(config, y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    rows = []
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for x_batch, rr_batch, y_batch in loader:
            x_batch = x_batch.to(pipeline.DEVICE)
            rr_batch = rr_batch.to(pipeline.DEVICE)
            y_batch = y_batch.to(pipeline.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch, rr_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(y_batch)
            seen += len(y_batch)
        scheduler.step()
        rows.append({"epoch": epoch, "train_loss": total / seen, "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"full DS1 epoch {epoch:02d}/{epochs}: loss={total / seen:.5f}")
        best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def main():
    ds1, ds2, metadata = pipeline.prepare_data()
    x1, rr1, y1, records1, train_mask, val_mask = ds1
    x2, rr2, y2, records2 = ds2
    selected = json.loads((pipeline.OUTPUT_DIR / "selected_config.json").read_text(encoding="utf-8"))
    config = pipeline.TrainingConfig(
        selected["name"], selected["loss"], selected["weighted_loss"], selected["sampler_power"]
    )

    # Reproduce the DS1 loss-tuning seed whose selected epoch was 10.
    pipeline.reset_seeds(1)
    validation_loader = pipeline.make_loader(x1[val_mask], rr1[val_mask], y1[val_mask], training=False)
    validation_model, validation_history, best_epoch = pipeline.fit_with_validation(
        pipeline.CNNLSTM(),
        config,
        (x1[train_mask], rr1[train_mask], y1[train_mask]),
        validation_loader,
        max_epochs=40,
        patience=8,
    )
    validation_logits, validation_labels = logits_for(validation_model, validation_loader)
    biases, calibrated_val_macro, calibrated_val_accuracy = select_biases(validation_logits, validation_labels)
    print(
        f"DS1 calibration: biases={biases.tolist()}, val_macro_f1={calibrated_val_macro:.4f}, "
        f"val_accuracy={calibrated_val_accuracy:.4f}"
    )

    # The tuning experiment selected epoch 10. Refit all DS1 records for that
    # fixed budget, rather than deriving a noisy epoch from a second seed.
    final_epochs = 10
    pipeline.reset_seeds(1)
    final_model, refit_history = train_fixed(
        pipeline.CNNLSTM(), config, (x1, rr1, y1), final_epochs
    )
    test_loader = pipeline.make_loader(x2, rr2, y2, training=False)
    test_logits, test_labels = logits_for(final_model, test_loader)
    test_prediction = (test_logits + biases).argmax(1)

    report = classification_report(
        test_labels,
        test_prediction,
        labels=range(4),
        target_names=pipeline.CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_labels, test_prediction, labels=range(4))
    class_rows = []
    for index, label in enumerate(pipeline.CLASSES):
        tp = cm[index, index]
        fn = cm[index].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = cm.sum() - tp - fn - fp
        class_rows.append({
            "class": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": report[label]["f1-score"],
            "support": int(report[label]["support"]),
        })
    class_df = pd.DataFrame(class_rows)
    summary = pd.DataFrame([{
        "model": "Calibrated CNN-LSTM",
        "accuracy": accuracy_score(test_labels, test_prediction),
        "macro_f1": f1_score(test_labels, test_prediction, average="macro", zero_division=0),
        "weighted_f1": f1_score(test_labels, test_prediction, average="weighted", zero_division=0),
        "best_val_macro_f1_before_calibration": validation_history["val_macro_f1"].max(),
        "calibrated_val_macro_f1": calibrated_val_macro,
        "calibrated_val_accuracy": calibrated_val_accuracy,
        "selected_epochs": final_epochs,
        "parameters": sum(p.numel() for p in final_model.parameters()),
    }])

    safe = "calibrated_cnn_lstm"
    torch.save(final_model.state_dict(), pipeline.OUTPUT_DIR / f"{safe}_weights.pt")
    np.save(pipeline.OUTPUT_DIR / f"{safe}_logit_biases.npy", biases)
    np.savetxt(pipeline.OUTPUT_DIR / f"{safe}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    validation_history.to_csv(pipeline.OUTPUT_DIR / f"{safe}_validation_history.csv", index=False)
    refit_history.to_csv(pipeline.OUTPUT_DIR / f"{safe}_full_ds1_refit_history.csv", index=False)
    class_df.to_csv(pipeline.OUTPUT_DIR / f"{safe}_per_class.csv", index=False)
    summary.to_csv(pipeline.OUTPUT_DIR / f"{safe}_ds2_summary.csv", index=False)
    with open(pipeline.OUTPUT_DIR / f"{safe}_metadata.json", "w", encoding="utf-8") as file:
        json.dump({
            **metadata,
            "loss": config.name,
            "selected_epochs": final_epochs,
            "early_stopping_max_epochs": 40,
            "early_stopping_patience": 8,
            "logit_biases": biases.tolist(),
            "calibration_accuracy_guard": 0.92,
            "calibration_source": "DS1 validation only",
        }, file, indent=2)
    print("\nCALIBRATED FINAL RESULT")
    print(summary.to_string(index=False))
    print(class_df.to_string(index=False))


if __name__ == "__main__":
    main()
