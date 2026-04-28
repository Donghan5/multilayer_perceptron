"""J. Evaluation tests (J1, J2, J4 thresholds).

J4 thresholds (medical domain — FN is critical):
  Recall(Malignant)    >= 0.95
  Precision(Malignant) >= 0.90
  F1(Malignant)        >= 0.92
"""
import pytest


# ── J1 / J2: history recorded ────────────────────────────────

def test_history_loss_recorded(trained_model):
    """J1: Train + val loss recorded every epoch."""
    h = trained_model["history"]
    assert len(h["loss"]) > 0
    assert len(h["val_loss"]) > 0
    assert len(h["loss"]) == len(h["val_loss"])


def test_history_accuracy_recorded(trained_model):
    """J2: Train + val accuracy recorded every epoch."""
    h = trained_model["history"]
    assert len(h["accuracy"]) > 0
    assert len(h["val_accuracy"]) > 0
    assert len(h["accuracy"]) == len(h["val_accuracy"])


# ── J4: metric thresholds (class 1 = Malignant) ──────────────

def test_malignant_recall(trained_model, compute_metrics):
    """Recall(M) >= 0.95 — must not miss malignant tumours."""
    _, recall, _ = compute_metrics(
        trained_model["y_true"], trained_model["y_pred"], class_idx=1,
    )
    assert recall >= 0.95, f"Malignant recall {recall:.4f} < 0.95"


def test_malignant_precision(trained_model, compute_metrics):
    """Precision(M) >= 0.90."""
    precision, _, _ = compute_metrics(
        trained_model["y_true"], trained_model["y_pred"], class_idx=1,
    )
    assert precision >= 0.90, f"Malignant precision {precision:.4f} < 0.90"


def test_malignant_f1(trained_model, compute_metrics):
    """F1(M) >= 0.92."""
    _, _, f1 = compute_metrics(
        trained_model["y_true"], trained_model["y_pred"], class_idx=1,
    )
    assert f1 >= 0.92, f"Malignant F1 {f1:.4f} < 0.92"
