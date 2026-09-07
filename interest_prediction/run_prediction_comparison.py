"""Self-contained main and supplementary Fig. 7 comparisons.

GPT 3.5, GPT 4o and GPT 4o-mini all replay their text-file comparisons in
a new shuffled order for each Monte Carlo run; no archived mini result is needed.
NN and decision-tree feature selection is repeated inside every training split.

Run this file directly, or use --help. All project inputs and outputs are local
to this directory; Python, NumPy, PyTorch, sklearn and Matplotlib are required.
"""

# Change this boolean to switch the default normalization method.
# True: fit mean/std on all observations (the original reference convention).
# False: fit mean/std separately on each MC training split.
# Both modes select features using only the current training split.
USE_GLOBAL_NORMALIZATION = False

import argparse
import csv
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import pickle
import platform
import random
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
PIPELINE_VERSION = 3
LOGGER = logging.getLogger("fig7_comparison")
MODELS = {
    "GPT 3.5": ("gpt35", "combined_ELO_results_35.txt", 0),
    "GPT 4o": ("gpt4o", "combined_ELO_results_4o.txt", 100000),
    "Neural Net": ("nn", None, 200000),
}
# Keep the original main-model configuration/signature compatible with saved runs.
MC_MODELS = {**MODELS, "Decision Tree": ("dt", None, 300000),
             "GPT 4o-mini": ("gpt4omini", "combined_ELO_results_4omini.txt", 400000)}
MODEL_TAGS = {name: meta[0] for name, meta in MC_MODELS.items()}
TREE_PARAMETERS = {"max_depth": 5, "min_samples_leaf": 5}
MINI_MATCHES = MC_MODELS["GPT 4o-mini"][1]
COLORS = {"GPT 3.5": "#2878B5", "GPT 4o": "#E58B2A", "Neural Net": "#30945B",
          "GPT 4o-mini": "#C94545", "Decision Tree": "#8864B5"}


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_pickle(path, data):
    atomic_write(path, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))


def save_csv(path, rows):
    if not rows:
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue().encode("utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_logging(output):
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.info("Log file: %s", log_path)
    return log_path


def load_inputs():
    # Never search a parent directory or fall back to an old result file.
    with (ROOT / "all_evaluation_data.pkl").open("rb") as handle:
        data = pickle.load(handle)
    features = np.asarray(data["features"], dtype=np.float64)
    interest = np.asarray(data["interest"], dtype=np.float64)
    if features.ndim != 2 or interest.ndim != 1 or len(features) != len(interest):
        raise ValueError("Expected a rows-by-features matrix and one interest value per row.")
    if features.shape[1] <= 141 or not np.isfinite(features).all() or not np.isfinite(interest).all():
        raise ValueError("Inputs need finite values and impact feature index 141.")
    if len(np.unique(interest > 3)) != 2:
        raise ValueError("Evaluation data must contain both interest classes.")
    elo_results = {}
    for name, (_, filename, _) in MC_MODELS.items():
        if filename is None:
            continue
        matches = np.loadtxt(ROOT / filename, delimiter=",", dtype=np.int64, ndmin=2)
        if matches.shape[1] != 3 or len(matches) == 0:
            raise ValueError(f"{filename}: expected nonempty id1,id2,winner rows.")
        if matches[:, :2].min() < 0 or matches[:, :2].max() >= len(interest):
            raise ValueError(f"{filename}: idea IDs outside the original data row range.")
        if np.any(matches[:, 0] == matches[:, 1]):
            raise ValueError(f"{filename}: a comparison contains the same idea twice.")
        # These input files use positional codes, not winning idea IDs.
        if not np.isin(matches[:, 2], [1, 2]).all():
            raise ValueError(f"{filename}: expected winner 1 (first idea) or 2 (second idea).")
        elo_results[name] = matches
        LOGGER.info("%s: %d comparisons, all used; winner encoding 1/2", name, len(matches))
    LOGGER.info("Data: %d rows, %d features, %d high-interest rows", *features.shape, (interest > 3).sum())
    return features, interest, elo_results


def select_top_features_from_training_data(x_train, y_train, n_features):
    """The same direction-independent, stable AUC ranking as the reference."""
    x_train = np.asarray(x_train)
    labels = np.asarray(y_train) > 3
    if len(np.unique(labels)) != 2:
        raise ValueError("Training split needs both classes for AUC feature selection.")
    if not 0 < n_features <= x_train.shape[1]:
        raise ValueError("Requested feature count is outside the candidate feature range.")
    raw_auc = np.array([roc_auc_score(labels, x_train[:, j]) for j in range(x_train.shape[1])])
    scores = np.maximum(raw_auc, 1.0 - raw_auc)
    selected = np.argsort(-scores, kind="stable")[:n_features]
    return selected, scores, raw_auc


def prepare_nn_split(features, interest, seed, hyper):
    # Preserve reference impact ordering, while saving original row IDs for audit.
    impact_order = np.argsort(-features[:, 141], kind="stable")
    indices = impact_order[np.random.default_rng(seed).permutation(len(interest))]
    train_end = int(len(indices) * hyper["train_ratio"])
    val_end = int(len(indices) * (hyper["train_ratio"] + hyper["val_ratio"]))
    split_ids = dict(zip(("train", "val", "test"), np.split(indices, [train_end, val_end])))
    if any(len(ids) == 0 for ids in split_ids.values()):
        raise ValueError("Each split must contain observations.")
    if len(np.unique(interest[split_ids["test"]] > 3)) != 2:
        raise ValueError(f"Seed {seed}: test split has one class; AUC is undefined. Run was not saved.")
    fit_ids = split_ids["train"] if hyper["normalization"] == "train" else np.arange(len(interest))
    mean = features[fit_ids].mean(axis=0)
    scale = features[fit_ids].std(axis=0)
    scale[scale == 0] = 1.0
    x = {part: ((features[ids] - mean) / scale).astype(np.float32) for part, ids in split_ids.items()}
    y = {part: interest[ids].astype(np.float32) for part, ids in split_ids.items()}
    selected, scores, raw_auc = select_top_features_from_training_data(x["train"], y["train"], hyper["n_features"])
    metadata = {
        **{part + "_indices": ids for part, ids in split_ids.items()},
        "selected_features": selected,
        "feature_auc_scores": scores,
        "feature_raw_auc_scores": raw_auc,
        "normalization_mean": mean,
        "normalization_scale": scale,
        "normalization": hyper["normalization"],
        "feature_selection": "training-only max(AUC, 1-AUC), stable feature-index tie breaking",
    }
    return {part: torch.from_numpy(values[:, selected]) for part, values in x.items()}, {
        part: torch.from_numpy(values) for part, values in y.items()
    }, metadata


class InterestPredictor(nn.Module):
    def __init__(self, input_features, neurons_per_layer, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for width in neurons_per_layer:
            self.layers.extend([nn.Linear(input_features, width), nn.ReLU(), nn.Dropout(dropout_rate)])
            input_features = width
        self.layers.append(nn.Linear(input_features, 1))

    def forward(self, values):
        for layer in self.layers:
            values = layer(values)
        return values


def train_model(model, x, y, hyper):
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper["lr"], weight_decay=hyper["weight_decay"])
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_loss, best_epoch, no_improvement, best_state = float("inf"), 0, 0, None
    for epoch in range(1, hyper["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x["train"]).reshape(-1), y["train"])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x["val"]).reshape(-1), y["val"]).item()
        if not np.isfinite(loss.item()) or not np.isfinite(val_loss):
            raise ValueError(f"Nonfinite loss at epoch {epoch}; run was not saved.")
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch, no_improvement = val_loss, epoch, 0
            # state_dict().copy() would share tensor storage with the live model.
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
        if epoch == 1 or epoch % 500 == 0:
            LOGGER.info("  Epoch %4d/%d | train MSE %.6f | val MSE %.6f | best %.6f at %d",
                        epoch, hyper["epochs"], loss.item(), val_loss, best_loss, best_epoch)
        if no_improvement >= hyper["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {
        "epochs_trained": epoch, "best_epoch": best_epoch, "best_val_loss": best_loss,
        "train_losses": np.asarray(train_losses), "val_losses": np.asarray(val_losses),
        "model_state_dict": {key: value.cpu().numpy().copy() for key, value in best_state.items()},
    }


def ranking_curves(labels):
    labels = np.asarray(labels, dtype=np.int8)
    return np.cumsum(labels) / np.arange(1, len(labels) + 1), np.maximum.accumulate(labels)


def exact_random_success_probability(total_size, n_positive, max_n):
    if not 0 <= n_positive <= total_size or not 0 < max_n <= total_size:
        raise ValueError("Invalid random-baseline population or curve length.")
    negative = total_size - n_positive
    probabilities = np.empty(max_n)
    no_success = 1.0
    for draw in range(max_n):
        no_success *= max(negative - draw, 0) / (total_size - draw)
        probabilities[draw] = 1.0 - no_success
    return probabilities


def run_one_nn_simulation(features, interest, run_seed, hyper):
    started = time.perf_counter()
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    x, y, metadata = prepare_nn_split(features, interest, run_seed, hyper)
    LOGGER.info("  Split train/val/test: %d/%d/%d | selected features: %s",
                len(y["train"]), len(y["val"]), len(y["test"]), metadata["selected_features"].tolist())
    model = InterestPredictor(hyper["n_features"], hyper["neurons_per_layer"], hyper["dropout"])
    training = train_model(model, x, y, hyper)
    with torch.no_grad():
        predictions = model(x["test"]).reshape(-1).cpu().numpy()
    labels = (y["test"].numpy() > 3).astype(np.int8)
    # Regression scores preserve ranking without saturation from an unnecessary sigmoid.
    ranked = np.argsort(-predictions, kind="stable")
    precision, success = ranking_curves(labels[ranked])
    random_order = np.random.default_rng(run_seed + 1000000).permutation(len(labels))
    _, random_success = ranking_curves(labels[random_order])
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    return {
        **metadata, **training, "hyperparameters": dict(hyper), "seed": int(run_seed),
        "auc": float(roc_auc_score(labels, predictions)), "fpr": fpr, "tpr": tpr,
        "roc_thresholds": thresholds, "test_interest": y["test"].numpy(),
        "test_labels": labels, "test_predictions": predictions,
        "ranked_test_indices": metadata["test_indices"][ranked], "ranked_labels": labels[ranked],
        "topNprecision": precision, "successProb": success,
        "random_order": random_order, "randomSuccess": random_success,
        "randomPrecisionScalar": float(labels.mean()),
        "randomSuccessExact": exact_random_success_probability(len(labels), int(labels.sum()), len(labels)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_one_tree_simulation(features, interest, run_seed, split_seed, hyper, tree_parameters):
    """Pair the tree with the NN split; fit and select features on training only.

    Unlike the old SI tree, validation rows are not merged into training. The
    fixed depth/leaf settings need no validation tuning; this matches the NN's
    75% training population and 10% held-out test population.
    """
    started = time.perf_counter()
    x, y, metadata = prepare_nn_split(features, interest, split_seed, hyper)
    model = DecisionTreeClassifier(**tree_parameters, random_state=run_seed)
    model.fit(x["train"].numpy(), (y["train"].numpy() > 3).astype(np.int8))
    predictions = model.predict_proba(x["test"].numpy())[:, np.flatnonzero(model.classes_ == 1)[0]]
    labels = (y["test"].numpy() > 3).astype(np.int8)
    # Stable ties use the already shuffled test order, never test labels.
    ranked = np.argsort(-predictions, kind="stable")
    precision, success = ranking_curves(labels[ranked])
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    random_order = np.random.default_rng(run_seed + 1000000).permutation(len(labels))
    _, random_success = ranking_curves(labels[random_order])
    LOGGER.info("  Tree paired NN split seed=%d | selected features: %s | depth=%d | leaves=%d",
                split_seed, metadata["selected_features"].tolist(), model.get_depth(), model.get_n_leaves())
    return {
        **metadata, "seed": int(run_seed), "split_seed": int(split_seed),
        "hyperparameters": dict(tree_parameters), "training_population": "75% training split only; validation unused",
        "auc": float(roc_auc_score(labels, predictions)), "fpr": fpr, "tpr": tpr,
        "roc_thresholds": thresholds, "test_interest": y["test"].numpy(),
        "test_labels": labels, "test_predictions": predictions,
        "ranked_test_indices": metadata["test_indices"][ranked], "ranked_labels": labels[ranked],
        "topNprecision": precision, "successProb": success, "random_order": random_order,
        "randomSuccess": random_success, "randomPrecisionScalar": float(labels.mean()),
        "randomSuccessExact": exact_random_success_probability(len(labels), int(labels.sum()), len(labels)),
        "model_object": model, "feature_importances": model.feature_importances_,
        "tree_depth": model.get_depth(), "tree_n_leaves": model.get_n_leaves(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def update_elo(scores, id1, id2, winner, k_factor=32.0):
    if winner not in (1, 2):
        raise ValueError("ELO winner must be positional code 1 or 2.")
    expected = 1.0 / (1.0 + 10.0 ** ((scores[id2] - scores[id1]) / 400.0))
    change = k_factor * (float(winner == 1) - expected)
    scores[id1] += change
    scores[id2] -= change


def ordered_subsystem_curves(ranked_labels, subsystem_size, n_subsamples, rng):
    if not 0 < subsystem_size <= len(ranked_labels) or n_subsamples < 1:
        raise ValueError("Invalid ELO subsystem size or subsample count.")
    precision, success = np.zeros(subsystem_size), np.zeros(subsystem_size)
    for _ in range(n_subsamples):
        indices = np.sort(rng.choice(len(ranked_labels), size=subsystem_size, replace=False))
        p, s = ranking_curves(ranked_labels[indices])
        precision += p
        success += s
    return precision / n_subsamples, success / n_subsamples


def run_one_elo_simulation(matches, interest, run_seed, config):
    started = time.perf_counter()
    rng = np.random.default_rng(run_seed)
    labels = (interest > 3).astype(np.int8)
    scores = np.full(len(interest), 1400.0)
    for id1, id2, winner in matches[rng.permutation(len(matches))]:
        update_elo(scores, id1, id2, winner, config["k_factor"])
    ranked = np.argsort(-scores, kind="stable")
    precision, success = ordered_subsystem_curves(labels[ranked], config["subsystem_size"], config["n_subsamples"], rng)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return {
        "seed": int(run_seed), "auc": float(roc_auc_score(labels, scores)),
        "fpr": fpr, "tpr": tpr, "roc_thresholds": thresholds, "elo_scores": scores,
        "ranked_indices": ranked, "ranked_labels": labels[ranked],
        "topNprecision": precision, "successProb": success, "num_matches": len(matches),
        **config, "elapsed_seconds": time.perf_counter() - started,
    }


def distribution(values):
    """Descriptive spread across runs; quantiles are not confidence intervals."""
    values = np.asarray(values, dtype=np.float64)
    sd = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.full(values.shape[1:], np.nan)
    return {"all": values, "mean": np.mean(values, axis=0), "sd": sd,
            "sem": sd / np.sqrt(len(values)), "quantile95": np.percentile(values, [2.5, 97.5], axis=0)}


def aggregate_model_runs(runs, model_kind, roc_points=1001):
    if not runs:
        raise ValueError("No runs to aggregate.")
    grid = np.linspace(0, 1, roc_points)
    interpolated = []
    for run in runs:
        curve = np.interp(grid, run["fpr"], run["tpr"])
        curve[0], curve[-1] = 0.0, 1.0
        interpolated.append(curve)
    out = {"n_runs": len(runs), "fpr": grid, "run_seeds": np.array([r["seed"] for r in runs]),
           "statistics_note": "SD, SEM and empirical 2.5/97.5 percentiles describe Monte Carlo runs on this fixed dataset; not population confidence intervals."}
    metrics = {"auc": [r["auc"] for r in runs], "tpr": interpolated,
               "topNprecision": [r["topNprecision"] for r in runs], "successProb": [r["successProb"] for r in runs]}
    if model_kind in ("nn", "dt"):
        metrics.update({key: [r[key] for r in runs] for key in ("randomSuccess", "randomSuccessExact", "randomPrecisionScalar")})
        selected = np.stack([r["selected_features"] for r in runs])
        n_features = len(runs[0]["feature_auc_scores"])
        counts = np.bincount(selected.ravel(), minlength=n_features)
        out.update(selected_features_per_split=selected,
                   feature_auc_scores_per_split=np.stack([r["feature_auc_scores"] for r in runs]),
                   feature_selection_counts=counts, feature_selection_frequencies=counts / len(runs))
    for metric, values in metrics.items():
        out.update({f"{metric}_{stat}": value for stat, value in distribution(values).items()})
    return out


def save_exports(aggregates, runs_by_model, output):
    run_rows, summary_rows, curve_rows = [], [], []
    for name, agg in aggregates.items():
        tag = MODEL_TAGS[name]
        save_pickle(output / tag / f"full_data_{tag}_avg.pkl", agg)
        for i, run in enumerate(runs_by_model[name]):
            run_rows.append({
                "model": name, "run_index": i, "seed": run["seed"], "auc": run["auc"],
                "precision_at_1": run["topNprecision"][0], "precision_at_5": run["topNprecision"][4],
                "success_at_1": run["successProb"][0], "success_at_5": run["successProb"][4],
                "best_epoch": run.get("best_epoch", ""), "epochs_trained": run.get("epochs_trained", ""),
                "best_val_loss": run.get("best_val_loss", ""),
                "elapsed_seconds": run["elapsed_seconds"],
                "selected_features": " ".join(map(str, run.get("selected_features", []))),
            })
        metrics = {"auc": agg["auc_all"], "precision_at_1": agg["topNprecision_all"][:, 0],
                   "precision_at_5": agg["topNprecision_all"][:, 4],
                   "success_at_1": agg["successProb_all"][:, 0], "success_at_5": agg["successProb_all"][:, 4]}
        for metric, values in metrics.items():
            stats = distribution(values)
            summary_rows.append({"model": name, "metric": metric, "n_runs": agg["n_runs"],
                                 **{key: float(stats[key]) for key in ("mean", "sd", "sem")},
                                 "q025": stats["quantile95"][0], "q975": stats["quantile95"][1]})
        for metric in ("tpr", "topNprecision", "successProb"):
            for j, mean in enumerate(agg[metric + "_mean"]):
                curve_rows.append({"model": name, "metric": metric,
                                   "x": agg["fpr"][j] if metric == "tpr" else j + 1,
                                   "mean": mean, "sd": agg[metric + "_sd"][j],
                                   "sem": agg[metric + "_sem"][j],
                                   "q025": agg[metric + "_quantile95"][0, j],
                                   "q975": agg[metric + "_quantile95"][1, j]})
    save_csv(output / "run_metrics.csv", run_rows)
    save_csv(output / "summary_statistics.csv", summary_rows)
    save_csv(output / "curve_statistics.csv", curve_rows)
    for name in ("Neural Net", "Decision Tree"):
        if name not in aggregates:
            continue
        agg = aggregates[name]
        save_csv(output / MODEL_TAGS[name] / "feature_selection.csv", [
            {"feature_index": j, "selection_count": int(count),
             "selection_frequency": agg["feature_selection_frequencies"][j],
             "mean_training_auc_score": agg["feature_auc_scores_per_split"][:, j].mean()}
            for j, count in enumerate(agg["feature_selection_counts"])
        ])


def metric_label(mean, sem):
    return f"{mean:.4f} ± {sem:.4f}" if np.isfinite(sem) else f"{mean:.4f} (SEM n/a)"


def draw_curve(ax, x, agg, metric, name, label, bands):
    mean = agg[metric + "_mean"][:len(x)]
    ax.plot(x, mean, lw=2.6, color=COLORS[name], label=label)
    if bands == "none" or agg["n_runs"] < 2:
        return
    if bands == "quantile95":
        low, high = agg[metric + "_quantile95"][:, :len(x)]
    else:
        delta = agg[metric + "_" + bands][:len(x)]
        low, high = mean - delta, mean + delta
    ax.fill_between(x, np.clip(low, 0, 1), np.clip(high, 0, 1), color=COLORS[name], alpha=0.14, linewidth=0)


def configure_plot_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 16, "axes.titlesize": 16,
                         "legend.fontsize": 12, "legend.title_fontsize": 12,
                         "xtick.labelsize": 16, "ytick.labelsize": 16,
                         "axes.linewidth": 2.0,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "axes.spines.bottom": True, "axes.spines.left": True,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def plot_comparison(aggregates, interest, output, bands, dpi, supplementary=False):
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    n_top = min(len(a["topNprecision_mean"]) for a in aggregates.values())
    n_success = min(10, n_top)
    for name, agg in aggregates.items():
        draw_curve(axes[0], agg["fpr"], agg, "tpr", name,
                   f"{name}: {metric_label(agg['auc_mean'], agg['auc_sem'])}", bands)
        for ax, metric, length in ((axes[1], "topNprecision", n_top), (axes[2], "successProb", n_success)):
            stats1 = distribution(agg[metric + "_all"][:, 0])
            stats5 = distribution(agg[metric + "_all"][:, 4])
            label = (f"{name}\nTop-1 {metric_label(stats1['mean'], stats1['sem'])}"
                     f"\nTop-5 {metric_label(stats5['mean'], stats5['sem'])}")
            draw_curve(ax, np.arange(1, length + 1), agg, metric, name, label, bands)
    labels = interest > 3
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", lw=3, label="Random: 0.50")
    axes[1].axhline(labels.mean(), ls="--", color="0.5", lw=3, label=f"Random: {labels.mean():.4f}")
    baseline = exact_random_success_probability(len(labels), int(labels.sum()), n_success)
    axes[2].plot(np.arange(1, n_success + 1), baseline, "--", color="0.5", lw=3,
                 label=f"Random\nTop-1 {baseline[0]:.4f}\nTop-5 {baseline[4]:.4f}")
    settings = [
        ("Average ROC curve", "False positive rate", "True positive rate", (0, 1), "lower right"),
        ("Top-N Precision", "Sorted research suggestion", "Precision", (1, n_top), "upper right"),
        ("Top-N Success Probability", "Sorted research suggestion", "Probability", (1, n_success), "lower right"),
    ]
    for i, (ax, (title, xlabel, ylabel, limits, loc)) in enumerate(zip(axes, settings)):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=limits, ylim=(0, 1))
        ax.grid(True, alpha=0.5)
        ax.set_axisbelow(True)
        legend = ax.legend(loc=loc, fontsize=12, framealpha=0.95, edgecolor="0.9",
                           title="Mean AUC ± SEM" if i == 0 else "Mean ± SEM across runs", title_fontsize=12)
        for text in legend.get_texts():
            text.set_multialignment("left")
        ax.text(-0.18, 1.0, f"({chr(ord('a') + i)})", transform=ax.transAxes, fontsize=18, fontweight="bold")
    #n_label = "; ".join(f"{name}: {agg['n_runs']} runs" for name, agg in aggregates.items())
    #note = "" if bands == "none" else f" | Bands: {'empirical 2.5–97.5 percentiles' if bands == 'quantile95' else 'mean ± ' + bands.upper()}"
    #fig.text(0.5, 0.015, n_label + note, ha="center", fontsize=10, color="0.35")
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=2.0)
    base = output / ("SI_predictions_with_DT" if supplementary else "GPT35_vs_GPT4o_vs_NN_avg")
    fig.savefig(base.with_suffix(".png"), dpi=dpi)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def plot_run_statistics(aggregates, output, dpi):
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 1.3]})
    rng = np.random.default_rng(0)
    for position, (name, agg) in enumerate(aggregates.items(), 1):
        values = agg["auc_all"]
        box = axes[0].boxplot(values, positions=[position], widths=0.45, patch_artist=True,
                             showfliers=False, medianprops={"color": "black"})
        box["boxes"][0].set(facecolor=COLORS[name], alpha=0.25)
        axes[0].scatter(position + rng.uniform(-0.12, 0.12, len(values)), values,
                        s=20, color=COLORS[name], alpha=0.7, edgecolors="none")
    axes[0].set_xticks(range(1, len(aggregates) + 1), list(aggregates))
    axes[0].axhline(0.5, ls="--", color="0.6", lw=1.5)
    axes[0].set(ylabel="ROC AUC per run", title="Monte Carlo AUC distributions")
    axes[0].grid(axis="y", alpha=0.2)
    agg = aggregates["Neural Net"]
    frequencies = agg["feature_selection_frequencies"]
    top = np.argsort(-frequencies, kind="stable")[:25][::-1]
    axes[1].barh(np.arange(len(top)), frequencies[top], color=COLORS["Neural Net"], alpha=0.85)
    axes[1].set_yticks(np.arange(len(top)), [str(j) for j in top])
    axes[1].set(xlim=(0, 1.04), xlabel="Fraction of NN training splits selecting feature",
                ylabel="Feature index (zero-based)", title="25 most frequently selected features")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout(w_pad=3)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"run_statistics.{suffix}", dpi=dpi)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--runs", type=int, default=100, help="Total target MC runs for every model")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed; main offsets 0/100000/200000, tree 300000, GPT 4o-mini 400000")
    parser.add_argument("--epochs", type=int, default=8000, help="Maximum NN epochs per MC split")
    parser.add_argument("--patience", type=int, default=500, help="Early stopping patience on validation MSE")
    parser.add_argument("--normalization", choices=("train", "global"), default="global" if USE_GLOBAL_NORMALIZATION else "train", help="Override USE_GLOBAL_NORMALIZATION: fit z-scores on each training split or globally")
    parser.add_argument("--elo-subsamples", type=int, default=1000, help="Ordered subsamples per ELO run")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; one is efficient for this small network")
    parser.add_argument("--plot-every", type=int, default=5, help="Refresh statistics and plots every N complete iterations")
    parser.add_argument("--bands", choices=("none", "sd", "sem", "quantile95"), default="none", help="Optional curve bands; these describe run variability")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution; PDF is also saved")
    parser.add_argument("--output", default=None, help="Output subdirectory inside Fig7_comparison; by default global normalization adds _global_normalization to ELO_compare_GPT35_GPT4o_GPT4omini_withNN")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate statistics/plots from the requested saved runs without training")
    args = parser.parse_args(argv)
    for name in ("runs", "epochs", "patience", "elo_subsamples", "threads", "plot_every", "dpi"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.seed < 2**32 - 1400000 - args.runs:
        parser.error("Seed plus run offsets must fit in the supported seed range")
    if args.output is None:
        args.output = "ELO_compare_GPT35_GPT4o_GPT4omini_withNN" + ("_global_normalization" if args.normalization == "global" else "")
    output = (ROOT / args.output).resolve()
    if output == ROOT or ROOT not in output.parents:
        parser.error("--output must be a subdirectory inside Fig7_comparison")
    args.output_path = output
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output_path
    setup_logging(output)
    started = time.perf_counter()
    try:
        LOGGER.info("Fig7 | mode=%s | requested runs per model=%d", "plot-only" if args.plot_only else "train/resume", args.runs)
        LOGGER.info("Script: %s | Python: %s", Path(__file__).resolve(), sys.executable)
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        features, interest, elo_results = load_inputs()
        hyper = {"n_features": 25, "neurons_per_layer": [50], "lr": 0.003,
                 "train_ratio": 0.75, "val_ratio": 0.15, "test_ratio": 0.1,
                 "dropout": 0.2, "weight_decay": 0.0007, "epochs": args.epochs,
                 "patience": args.patience, "normalization": args.normalization}
        n_test = len(interest) - int(len(interest) * (hyper["train_ratio"] + hyper["val_ratio"]))
        if n_test < 5:
            raise ValueError("At least five test observations are needed for Top-5 statistics.")
        elo_config = {"k_factor": 32.0, "subsystem_size": n_test, "n_subsamples": args.elo_subsamples}
        versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
                    "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
        configuration = {"pipeline_version": PIPELINE_VERSION, "base_seed": args.seed,
                         "nn_hyperparameters": hyper, "elo_configuration": elo_config,
                         "torch_threads": args.threads, "versions": versions,
                         "input_sha256": {name: file_hash(ROOT / name) for name in (
                             "all_evaluation_data.pkl", "combined_ELO_results_35.txt", "combined_ELO_results_4o.txt")}}
        signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["signature"] != signature:
                raise ValueError("Saved experiment uses different data/settings/package versions. Use its original options, or choose a new --output subdirectory; results cannot be mixed.")
        elif args.plot_only:
            raise FileNotFoundError(f"No saved experiment manifest: {manifest_path}")
        else:
            manifest = {"signature": signature, "configuration": configuration,
                        "script_sha256": file_hash(__file__), "created_at": datetime.now().isoformat()}
            atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        supplementary_configuration = {
            "supplementary_version": 2, "main_signature": signature,
            "tree_parameters": TREE_PARAMETERS, "tree_training": "paired NN training split only",
            "tree_seed_offset": 300000, "tree_split_seed_offset": 200000,
            "mini_matches_sha256": file_hash(ROOT / MINI_MATCHES),
            "mini_evaluation": "repeated ELO fits over shuffled match order",
            "mini_seed_offset": 400000,
            "subsystem_size": n_test, "n_subsamples": args.elo_subsamples,
        }
        supplementary_signature = hashlib.sha256(json.dumps(supplementary_configuration, sort_keys=True).encode()).hexdigest()
        supplementary_manifest_path = output / "supplementary_manifest.json"
        if supplementary_manifest_path.exists():
            saved = json.loads(supplementary_manifest_path.read_text(encoding="utf-8"))
            if saved["signature"] != supplementary_signature:
                raise ValueError("Supplementary inputs/settings changed. Restore their original values or use a new --output directory.")
        elif args.plot_only:
            raise FileNotFoundError("Supplementary results are not computed yet. Run once without --plot-only; existing main runs will be reused.")
        else:
            supplementary_manifest = {"signature": supplementary_signature, "configuration": supplementary_configuration,
                                      "script_sha256": file_hash(__file__), "created_at": datetime.now().isoformat()}
            atomic_write(supplementary_manifest_path, (json.dumps(supplementary_manifest, indent=2) + "\n").encode("utf-8"))
        LOGGER.info("NN settings: %s", json.dumps(hyper, sort_keys=True))
        LOGGER.info("ELO settings: %s | original data row order retained for labels", json.dumps(elo_config))
        LOGGER.info("Environment: %s", json.dumps(versions, sort_keys=True))
        LOGGER.info("Experiment signature: %s", signature)
        LOGGER.info("Supplementary signature: %s", supplementary_signature)
        LOGGER.info("Decision Tree settings: %s; paired NN train/test splits, training-only selection; validation rows unused", json.dumps(TREE_PARAMETERS))
        LOGGER.info("GPT 4o-mini: %s; %d shuffled ELO runs, using the same procedure as GPT 3.5 and GPT 4o", MINI_MATCHES, args.runs)
        LOGGER.info("SD/SEM and empirical quantiles describe runs on this fixed dataset; they are not population confidence intervals.")
        if args.normalization == "global":
            LOGGER.warning("Global normalization matches the reference convention and includes held-out feature values in scaling.")
        runs_by_model = {name: [] for name in MC_MODELS}
        # Validate all requested cached runs before doing any expensive new training.
        cached = {}
        for i in range(args.runs):
            for name, (tag, _, offset) in MC_MODELS.items():
                path = output / tag / "runs" / f"run_{i:03d}.pkl"
                if not path.exists():
                    if args.plot_only:
                        raise FileNotFoundError(f"--plot-only needs all requested runs; missing {path}")
                    continue
                with path.open("rb") as handle:
                    run = pickle.load(handle)
                expected_signature = signature if name in MODELS else supplementary_signature
                if (run.get("signature") != expected_signature or run.get("model") != name
                        or run.get("run_index") != i or run.get("seed") != args.seed + offset + i):
                    raise ValueError(f"Incompatible saved run: {path}. Old v2 files cannot be reused.")
                if tag in ("nn", "dt") and "selected_features" not in run:
                    raise ValueError(f"Missing per-split feature selection in {path}")
                cached[name, i] = run
        new_runs = 0
        for i in range(args.runs):
            LOGGER.info("========== Iteration %d/%d ==========", i + 1, args.runs)
            for name, (tag, _, offset) in MC_MODELS.items():
                path = output / tag / "runs" / f"run_{i:03d}.pkl"
                run = cached.get((name, i))
                if run is None:
                    seed = args.seed + offset + i
                    LOGGER.info("%s: starting run %d/%d, seed=%d", name, i + 1, args.runs, seed)
                    if tag == "nn":
                        run = run_one_nn_simulation(features, interest, seed, hyper)
                    elif tag == "dt":
                        run = run_one_tree_simulation(features, interest, seed,
                                                      args.seed + MODELS["Neural Net"][2] + i, hyper, TREE_PARAMETERS)
                    else:
                        run = run_one_elo_simulation(elo_results[name], interest, seed, elo_config)
                    run.update(signature=signature if name in MODELS else supplementary_signature,
                               model=name, run_index=i, created_at=datetime.now().isoformat())
                    save_pickle(path, run)
                    new_runs += 1
                    LOGGER.info("%s: saved %s | AUC %.6f | %.1f seconds", name, path.relative_to(output), run["auc"], run["elapsed_seconds"])
                    if tag == "nn":
                        LOGGER.info("  Restored best epoch %d of %d; validation MSE %.6f", run["best_epoch"], run["epochs_trained"], run["best_val_loss"])
                else:
                    LOGGER.info("%s: reused validated run %d | seed %d | AUC %.6f", name, i + 1, run["seed"], run["auc"])
                runs_by_model[name].append(run)
            refresh = i == args.runs - 1 or (not args.plot_only and (i == 0 or (i + 1) % args.plot_every == 0))
            if refresh:
                aggregates = {name: aggregate_model_runs(runs, MODEL_TAGS[name]) for name, runs in runs_by_model.items()}
                save_exports(aggregates, runs_by_model, output)
                main_aggregates = {name: aggregates[name] for name in MODELS}
                plot_comparison(main_aggregates, interest, output, args.bands, args.dpi)
                supplementary_aggregates = {**main_aggregates, "GPT 4o-mini": aggregates["GPT 4o-mini"],
                                            "Decision Tree": aggregates["Decision Tree"]}
                plot_comparison(supplementary_aggregates, interest, output, args.bands, args.dpi, supplementary=True)
                plot_run_statistics(main_aggregates, output, args.dpi)
                for name in MC_MODELS:
                    agg = aggregates[name]
                    LOGGER.info("%s: %d runs | AUC %s (mean +/- SEM) | SD %.6f",
                                name, agg["n_runs"], metric_label(agg["auc_mean"], agg["auc_sem"]), agg["auc_sd"])
                LOGGER.info("Updated main and supplementary PNG/PDF figures, aggregate pickles and CSV statistics in %s", output)
        LOGGER.info("Finished %d MC runs per model (GPT 3.5, GPT 4o, NN, Decision Tree, GPT 4o-mini); %d newly computed model runs in %.1f seconds.", args.runs, new_runs, time.perf_counter() - started)
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted. Completed run files are saved; rerun the same command to resume.")
        return 130
    except Exception:
        LOGGER.exception("Comparison failed. Completed run files remain available for resume.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
