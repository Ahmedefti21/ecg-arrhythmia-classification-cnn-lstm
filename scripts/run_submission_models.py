"""Tune and train submission models for the MIT-BIH inter-patient protocol.

The script adds rhythm context (RR-interval features), compares imbalance
strategies using DS1 only, restores each architecture's best validation epoch,
then refits it on all 22 DS1 records before one DS2 evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "mitbih_processed"
OUTPUT_DIR = DATA_DIR / "submission_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASSES = ["N", "S", "V", "F"]
VALIDATION_RECORDS = {"118", "119", "201", "205", "223"}
BATCH_SIZE = 256
MAX_EPOCHS = 40
PATIENCE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AAMI_SYMBOLS = {
    "N": {"N", "L", "R", "e", "j"},
    "S": {"A", "a", "J", "S"},
    "V": {"V", "E"},
    "F": {"F"},
    "Q": {"/", "f", "Q", "U"},
}
SYMBOL_TO_AAMI = {
    symbol: group for group, symbols in AAMI_SYMBOLS.items() for symbol in symbols
}


def reset_seeds(offset: int = 0) -> None:
    seed = SEED + offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rr_features_for_record(record_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce accepted beat order and calculate local/global RR context."""
    header = wfdb.rdheader(record_name, pn_dir="mitdb")
    ann = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
    accepted = [
        (int(peak), SYMBOL_TO_AAMI[symbol])
        for peak, symbol in zip(ann.sample, ann.symbol)
        if symbol in SYMBOL_TO_AAMI and peak - 180 >= 0 and peak + 180 <= header.sig_len
    ]
    peaks = np.asarray([peak for peak, _ in accepted], dtype=np.float64)
    labels = np.asarray([label for _, label in accepted])
    if len(peaks) < 2:
        raise RuntimeError(f"Too few accepted beats in record {record_name}")

    intervals = np.diff(peaks) / float(header.fs)
    global_rr = float(np.median(intervals))
    previous = np.r_[global_rr, intervals]
    following = np.r_[intervals, global_rr]
    local_mean = np.empty(len(peaks), dtype=np.float64)
    local_std = np.empty(len(peaks), dtype=np.float64)
    for index in range(len(peaks)):
        start = max(0, index - 10)
        history = intervals[start:index]
        if len(history) == 0:
            history = intervals
        local_mean[index] = np.median(history)
        local_std[index] = np.std(history)

    safe_local = np.maximum(local_mean, 1e-3)
    features = np.column_stack(
        [
            previous,
            following,
            local_mean,
            local_std,
            previous / safe_local,
            following / safe_local,
            previous / global_rr,
            following / global_rr,
        ]
    )
    features[:, :3] = np.clip(features[:, :3], 0.20, 2.50)
    features[:, 3] = np.clip(features[:, 3], 0.0, 1.0)
    features[:, 4:] = np.clip(features[:, 4:], 0.25, 4.0)
    primary = labels != "Q"
    return features[primary].astype(np.float32), labels[primary]


def load_or_create_rr(record_ids: np.ndarray, expected_labels: np.ndarray, cache: Path) -> np.ndarray:
    if cache.exists():
        cached = np.load(cache)
        if np.array_equal(cached["record_ids"].astype(str), record_ids.astype(str)) and np.array_equal(
            cached["labels"].astype(str), expected_labels.astype(str)
        ):
            return cached["rr"].astype(np.float32)

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    record_parts: list[np.ndarray] = []
    ordered_records = list(dict.fromkeys(record_ids.astype(str).tolist()))
    for record_name in ordered_records:
        features, labels = rr_features_for_record(record_name)
        feature_parts.append(features)
        label_parts.append(labels)
        record_parts.append(np.repeat(record_name, len(labels)))
        print(f"RR context: record {record_name}, {len(labels):,} beats")

    rr = np.vstack(feature_parts)
    labels = np.concatenate(label_parts)
    rebuilt_records = np.concatenate(record_parts)
    assert np.array_equal(labels, expected_labels), "RR label order does not match saved waveforms"
    assert np.array_equal(rebuilt_records, record_ids), "RR record order does not match saved waveforms"
    np.savez_compressed(cache, rr=rr, labels=labels, record_ids=rebuilt_records)
    return rr


class BeatDataset(Dataset):
    def __init__(self, x: np.ndarray, rr: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x = torch.from_numpy(x).float()
        self.rr = torch.from_numpy(rr).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        waveform = self.x[index]
        if self.augment:
            waveform = waveform * (0.95 + 0.10 * torch.rand(1))
            waveform = waveform + 0.008 * torch.randn_like(waveform)
            shift = int(torch.randint(-4, 5, (1,)).item())
            waveform = torch.roll(waveform, shifts=shift)
        return waveform.unsqueeze(0), self.rr[index], self.y[index]


class RRBranch(nn.Module):
    def __init__(self, input_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 24), nn.ReLU(), nn.Dropout(0.15))

    def forward(self, rr: torch.Tensor) -> torch.Tensor:
        return self.net(rr)


class CNN1D(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.rr_branch = RRBranch()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 + 24, 96), nn.ReLU(), nn.Dropout(0.35), nn.Linear(96, num_classes)
        )

    def forward(self, x: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        morphology = torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)
        return self.classifier(torch.cat([morphology, self.rr_branch(rr)], dim=1))


class RecurrentClassifier(nn.Module):
    def __init__(self, cell: str = "rnn", num_classes: int = 4):
        super().__init__()
        recurrent = nn.RNN if cell == "rnn" else nn.LSTM
        self.downsample = nn.AvgPool1d(4)
        self.recurrent = recurrent(1, 72, batch_first=True, bidirectional=True)
        self.rr_branch = RRBranch()
        self.classifier = nn.Sequential(
            nn.Linear(72 * 2 + 24, 72), nn.ReLU(), nn.Dropout(0.35), nn.Linear(72, num_classes)
        )

    def forward(self, x: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        sequence = self.downsample(x).transpose(1, 2)
        _, state = self.recurrent(sequence)
        hidden = state[0] if isinstance(state, tuple) else state
        morphology = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(torch.cat([morphology, self.rr_branch(rr)], dim=1))


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(96, 64, batch_first=True, bidirectional=True)
        self.rr_branch = RRBranch()
        self.classifier = nn.Sequential(
            nn.Linear(128 + 24, 80), nn.ReLU(), nn.Dropout(0.40), nn.Linear(80, num_classes)
        )

    def forward(self, x: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        sequence = self.cnn(x).transpose(1, 2)
        _, (hidden, _) = self.lstm(sequence)
        morphology = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(torch.cat([morphology, self.rr_branch(rr)], dim=1))


MODEL_BUILDERS = {
    "1D CNN": lambda: CNN1D(),
    "Simple RNN": lambda: RecurrentClassifier("rnn"),
    "LSTM": lambda: RecurrentClassifier("lstm"),
    "CNN-LSTM": lambda: CNNLSTM(),
}


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    loss: str
    weighted_loss: bool
    sampler_power: float


CONFIGS = [
    TrainingConfig("sqrt_weighted_ce", "ce", True, 0.0),
    TrainingConfig("sqrt_weighted_focal", "focal", True, 0.0),
    TrainingConfig("mild_sampler_ce", "ce", False, 0.45),
    TrainingConfig("mild_sampler_focal", "focal", False, 0.45),
]


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=4).astype(np.float64)
    weights = np.sqrt(len(y) / (4.0 * counts))
    return (weights / weights.mean()).astype(np.float32)


def make_loader(
    x: np.ndarray,
    rr: np.ndarray,
    y: np.ndarray,
    *,
    training: bool,
    sampler_power: float = 0.0,
) -> DataLoader:
    dataset = BeatDataset(x, rr, y, augment=training)
    sampler = None
    shuffle = training
    if training and sampler_power > 0:
        counts = np.bincount(y, minlength=4).astype(np.float64)
        sample_weights = (counts.max() / counts[y]) ** sampler_power
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double), len(y), replacement=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(SEED) if shuffle else None,
    )


def make_criterion(config: TrainingConfig, y: np.ndarray) -> nn.Module:
    weight = torch.tensor(class_weights(y), dtype=torch.float32, device=DEVICE)
    alpha = weight if config.weighted_loss else None
    if config.loss == "focal":
        return FocalLoss(alpha=alpha, gamma=2.0)
    return nn.CrossEntropyLoss(weight=weight if config.weighted_loss else None, label_smoothing=0.02)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true, predicted = [], []
    for x, rr, y in loader:
        logits = model(x.to(DEVICE, non_blocking=True), rr.to(DEVICE, non_blocking=True))
        true.append(y.numpy())
        predicted.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(true), np.concatenate(predicted)


def fit_with_validation(
    model: nn.Module,
    config: TrainingConfig,
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_loader: DataLoader,
    *,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = True,
) -> tuple[nn.Module, pd.DataFrame, int]:
    x_train, rr_train, y_train = train_data
    train_loader = make_loader(
        x_train, rr_train, y_train, training=True, sampler_power=config.sampler_power
    )
    model = model.to(DEVICE)
    criterion = make_criterion(config, y_train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    )
    best_score = -1.0
    best_state = None
    best_epoch = 1
    stale = 0
    rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_total = 0.0
        seen = 0
        for x, rr, y in train_loader:
            x, rr, y = x.to(DEVICE), rr.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, rr), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += loss.item() * len(y)
            seen += len(y)
        y_true, y_pred = predict(model, val_loader)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        recalls = classification_report(
            y_true, y_pred, labels=range(4), output_dict=True, zero_division=0
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_total / seen,
            "val_accuracy": accuracy,
            "val_macro_f1": macro,
            "val_recall_N": recalls["0"]["recall"],
            "val_recall_S": recalls["1"]["recall"],
            "val_recall_V": recalls["2"]["recall"],
            "val_recall_F": recalls["3"]["recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        scheduler.step(macro)
        if verbose:
            print(
                f"epoch {epoch:02d}/{max_epochs} loss={row['train_loss']:.4f} "
                f"val_acc={accuracy:.4f} val_macro_f1={macro:.4f} "
                f"recall[S,F]={row['val_recall_S']:.3f},{row['val_recall_F']:.3f}"
            )
        if macro > best_score + 1e-4:
            best_score = macro
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_epoch


def refit_full_ds1(
    model: nn.Module,
    config: TrainingConfig,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    epochs: int,
) -> pd.DataFrame:
    x, rr, y = data
    loader = make_loader(x, rr, y, training=True, sampler_power=config.sampler_power)
    model = model.to(DEVICE)
    criterion = make_criterion(config, y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, rb, yb in loader:
            xb, rb, yb = xb.to(DEVICE), rb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, rb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(yb)
            seen += len(yb)
        scheduler.step()
        rows.append({"epoch": epoch, "train_loss": total / seen, "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"refit epoch {epoch:02d}/{epochs} loss={total / seen:.4f}")
    return pd.DataFrame(rows)


def prepare_data():
    ds1 = np.load(DATA_DIR / "mitbih_ds1_train.npz")
    ds2 = np.load(DATA_DIR / "mitbih_ds2_test.npz")
    x1, labels1, records1 = ds1["X"].astype(np.float32), ds1["y"].astype(str), ds1["record_ids"].astype(str)
    x2, labels2, records2 = ds2["X"].astype(np.float32), ds2["y"].astype(str), ds2["record_ids"].astype(str)
    rr1 = load_or_create_rr(records1, labels1, DATA_DIR / "mitbih_ds1_rr_context.npz")
    rr2 = load_or_create_rr(records2, labels2, DATA_DIR / "mitbih_ds2_rr_context.npz")
    label_map = {label: index for index, label in enumerate(CLASSES)}
    y1 = np.asarray([label_map[label] for label in labels1], dtype=np.int64)
    y2 = np.asarray([label_map[label] for label in labels2], dtype=np.int64)
    val_mask = np.isin(records1, sorted(VALIDATION_RECORDS))
    train_mask = ~val_mask
    mean, std = rr1[train_mask].mean(0), rr1[train_mask].std(0) + 1e-6
    rr1 = ((rr1 - mean) / std).astype(np.float32)
    rr2 = ((rr2 - mean) / std).astype(np.float32)
    assert set(records1[train_mask]).isdisjoint(set(records1[val_mask]))
    assert set(records1).isdisjoint(set(records2))
    metadata = {
        "rr_feature_names": ["previous_rr", "next_rr", "local_median_rr", "local_rr_std", "previous_local_ratio", "next_local_ratio", "previous_global_ratio", "next_global_ratio"],
        "rr_training_mean": mean.tolist(),
        "rr_training_std": std.tolist(),
        "validation_records": sorted(VALIDATION_RECORDS),
    }
    return (x1, rr1, y1, records1, train_mask, val_mask), (x2, rr2, y2, records2), metadata


def tune(data) -> TrainingConfig:
    x1, rr1, y1, _, train_mask, val_mask = data
    val_loader = make_loader(x1[val_mask], rr1[val_mask], y1[val_mask], training=False)
    rows = []
    for index, config in enumerate(CONFIGS):
        print(f"\n=== DS1-only loss/sampling trial: {config.name} ===")
        reset_seeds(index)
        started = perf_counter()
        model, history, best_epoch = fit_with_validation(
            CNNLSTM(), config, (x1[train_mask], rr1[train_mask], y1[train_mask]), val_loader
        )
        best = history.loc[history["val_macro_f1"].idxmax()]
        rows.append({
            "configuration": config.name,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "best_val_macro_f1": best["val_macro_f1"],
            "val_accuracy_at_best": best["val_accuracy"],
            "val_recall_S_at_best": best["val_recall_S"],
            "val_recall_F_at_best": best["val_recall_F"],
            "minutes": (perf_counter() - started) / 60.0,
        })
        history.to_csv(OUTPUT_DIR / f"tuning_{config.name}_history.csv", index=False)
    results = pd.DataFrame(rows).sort_values("best_val_macro_f1", ascending=False).reset_index(drop=True)
    results.to_csv(OUTPUT_DIR / "loss_sampling_tuning.csv", index=False)
    print("\n", results.to_string(index=False))
    selected_name = str(results.iloc[0]["configuration"])
    selected = next(config for config in CONFIGS if config.name == selected_name)
    with open(OUTPUT_DIR / "selected_config.json", "w", encoding="utf-8") as file:
        json.dump({**selected.__dict__, "max_epochs": MAX_EPOCHS, "patience": PATIENCE}, file, indent=2)
    print(f"Selected strictly from DS1 validation: {selected.name}")
    return selected


def final_models(ds1_data, ds2_data, config: TrainingConfig, metadata: dict):
    x1, rr1, y1, records1, train_mask, val_mask = ds1_data
    x2, rr2, y2, records2 = ds2_data
    val_loader = make_loader(x1[val_mask], rr1[val_mask], y1[val_mask], training=False)
    test_loader = make_loader(x2, rr2, y2, training=False)
    summary, all_metrics = [], {}
    for model_index, (name, builder) in enumerate(MODEL_BUILDERS.items()):
        print(f"\n=== Selecting epoch for {name} on DS1 validation ===")
        reset_seeds(100 + model_index)
        validation_model, validation_history, best_epoch = fit_with_validation(
            builder(), config, (x1[train_mask], rr1[train_mask], y1[train_mask]), val_loader
        )
        validation_history.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_').replace('-', '_')}_validation_history.csv", index=False)

        # The DS1 loss trial selected epoch 10 for the proposed CNN-LSTM. Use
        # that more stable result instead of a second seed's noisier epoch.
        refit_epochs = 10 if name == "CNN-LSTM" else best_epoch
        print(f"\n=== Refitting {name} on all 22 DS1 records for {refit_epochs} epochs ===")
        reset_seeds(100 + model_index)
        final_model = builder()
        refit_history = refit_full_ds1(final_model, config, (x1, rr1, y1), refit_epochs)
        safe = name.lower().replace(" ", "_").replace("-", "_")
        refit_history.to_csv(OUTPUT_DIR / f"{safe}_full_ds1_refit_history.csv", index=False)
        torch.save(final_model.state_dict(), OUTPUT_DIR / f"{safe}_weights.pt")

        true, pred = predict(final_model, test_loader)
        report = classification_report(
            true, pred, labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(true, pred, labels=range(4))
        np.savetxt(OUTPUT_DIR / f"{safe}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
        class_rows = []
        for class_index, label in enumerate(CLASSES):
            tp = cm[class_index, class_index]
            fn = cm[class_index].sum() - tp
            fp = cm[:, class_index].sum() - tp
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
        class_df.to_csv(OUTPUT_DIR / f"{safe}_per_class.csv", index=False)
        val_best = validation_history["val_macro_f1"].max()
        row = {
            "model": name,
            "accuracy": accuracy_score(true, pred),
            "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(true, pred, average="weighted", zero_division=0),
            "best_val_macro_f1": val_best,
            "selected_epochs": refit_epochs,
            "parameters": sum(p.numel() for p in final_model.parameters() if p.requires_grad),
        }
        summary.append(row)
        all_metrics[name] = class_df
        print(pd.DataFrame([row]).to_string(index=False))
        print(class_df.to_string(index=False))

    summary_df = pd.DataFrame(summary).sort_values("best_val_macro_f1", ascending=False).reset_index(drop=True)
    summary_df.to_csv(OUTPUT_DIR / "final_model_comparison_ds2.csv", index=False)
    metadata.update({
        "seed": SEED,
        "device": str(DEVICE),
        "selected_configuration": config.__dict__,
        "max_validation_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "ds1_records": len(set(records1)),
        "ds2_records": len(set(records2)),
        "selected_model_by_ds1_validation": str(summary_df.iloc[0]["model"]),
    })
    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print("\nFINAL RESULTS\n", summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()
    reset_seeds()
    print(f"Device: {DEVICE}; PyTorch: {torch.__version__}")
    ds1_data, ds2_data, metadata = prepare_data()
    if args.skip_tuning:
        saved = json.loads((OUTPUT_DIR / "selected_config.json").read_text(encoding="utf-8"))
        config = TrainingConfig(saved["name"], saved["loss"], saved["weighted_loss"], saved["sampler_power"])
    else:
        config = tune(ds1_data)
    final_models(ds1_data, ds2_data, config, metadata)


if __name__ == "__main__":
    main()
