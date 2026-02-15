#!/usr/bin/env python3
"""
DigiCow Farmer Training Adoption Challenge — Training Script
=============================================================
Standalone script version of improved_models.ipynb.
Runs RF, Baseline NN, and RWN models with 5-fold OOF chaining.

Usage:
    python train.py                          # Run all models
    python train.py --models rf              # Only Random Forest
    python train.py --models nn rwn          # Only NN and RWN
    python train.py --n-splits 3 --seed 42   # Custom folds/seed
    python train.py --output-dir results/    # Save submissions elsewhere

Designed for cloud GPU instances (e.g. Vast.ai RTX 5060 Ti).
"""

import argparse
import ast
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
from category_encoders import TargetEncoder

try:
    from tqdm import tqdm, trange
except ImportError:
    # Fallback if tqdm isn't installed — use simple wrappers
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r  {desc} {i+1}/{total}", end="", flush=True)
            yield item
        print()

    def trange(n, **kwargs):
        return tqdm(range(n), total=n, **kwargs)

warnings.filterwarnings("ignore")


# ============================================================
# CLI Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="DigiCow Training Script — RF + NN + RWN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                        # all models, default settings
  python train.py --models rf            # Random Forest only (~5 min)
  python train.py --models nn rwn        # neural network models only
  python train.py --n-splits 3           # faster with 3 folds
  python train.py --output-dir results/  # save CSVs to results/
        """,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rf", "nn", "rwn"],
        default=["rf", "nn", "rwn"],
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of CV folds (default: 5)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save submission CSVs (default: current dir)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Original Data",
        help='Path to data directory containing Train.csv, Prior.csv, Test.csv (default: "Original Data")',
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the feature ablation study to save time",
    )
    parser.add_argument(
        "--nn-epochs", type=int, default=2000, help="Max epochs for NN/RWN (default: 2000)"
    )
    parser.add_argument(
        "--nn-patience", type=int, default=50, help="Early stopping patience (default: 50)"
    )
    parser.add_argument(
        "--nn-hidden", type=int, default=512, help="Hidden dim for MLP models (default: 512)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for NN training (default: 100)"
    )
    return parser.parse_args()


# ============================================================
# GPU / System Info
# ============================================================
def print_system_info():
    """Print system and GPU information."""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  NumPy:     {np.__version__}")
    print(f"  Pandas:    {pd.__version__}")

    if torch.cuda.is_available():
        print(f"\n  🟢 CUDA Available!")
        print(f"  CUDA version:    {torch.version.cuda}")
        print(f"  GPU count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_mem / (1024**3)
            print(f"  GPU {i}: {props.name}")
            print(f"    VRAM:          {vram_gb:.1f} GB")
            print(f"    Compute:       {props.major}.{props.minor}")
            print(f"    SM count:      {props.multi_processor_count}")
        # Show current VRAM usage
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"\n  VRAM allocated:  {allocated:.2f} GB")
        print(f"  VRAM reserved:   {reserved:.2f} GB")
    else:
        print("\n  🔴 CUDA not available — running on CPU")
        print("     Training will be significantly slower for NN/RWN models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device}")
    print("=" * 70)
    return device


# ============================================================
# Data Loading
# ============================================================
def load_data(data_dir):
    """Load and combine training data."""
    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    train_path = os.path.join(data_dir, "Train.csv")
    prior_path = os.path.join(data_dir, "Prior.csv")
    test_path = os.path.join(data_dir, "Test.csv")

    for p in [train_path, prior_path, test_path]:
        if not os.path.exists(p):
            print(f"  ❌ Missing: {p}")
            sys.exit(1)

    df = pd.read_csv(train_path)
    prior_df = pd.read_csv(prior_path)
    test_df = pd.read_csv(test_path)

    # Parse trainer from list literal
    df["trainer"] = df["trainer"].apply(lambda x: ast.literal_eval(x)[0])

    # Combine datasets
    combined_df = pd.concat([df, prior_df], ignore_index=True)

    # Build target columns from combined data
    TARGETS = [
        "adopted_within_07_days",
        "adopted_within_90_days",
        "adopted_within_120_days",
    ]
    for t in TARGETS:
        df[t] = df[t].astype(int)

    print(f"  Combined dataset: {combined_df.shape}")
    print(f"  Training data:    {df.shape}")
    print(f"  Test data:        {test_df.shape}")
    print(f"\n  Target distribution:")
    for t in TARGETS:
        pos_rate = df[t].mean()
        print(f"    {t}: {pos_rate:.4f} ({df[t].sum()}/{len(df)})")

    return df, combined_df, test_df, TARGETS


# ============================================================
# Preprocessing
# ============================================================
def preprocess(df, combined_df):
    """Shared preprocessing: topic features, TF-IDF, date features."""
    print("\n" + "=" * 70)
    print("PREPROCESSING")
    print("=" * 70)

    # --- Topic parsing and cleaning ---
    def clean_and_flat_topics(raw):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        flat_topics = []
        def flatten(obj):
            if isinstance(obj, list):
                for item in obj:
                    flatten(item)
            elif isinstance(obj, str):
                flat_topics.append(obj)
        flatten(parsed)
        cleaned = sorted(list(set([t.lower().strip() for t in flat_topics if t])))
        return cleaned

    df["clean_topics"] = df["topics_list"].apply(clean_and_flat_topics)

    # MultiLabelBinarizer for topic columns
    mlb = MultiLabelBinarizer()
    topics_encoded = mlb.fit_transform(df["clean_topics"])
    topic_columns = [f"topic_{t}" for t in mlb.classes_]
    topics_df = pd.DataFrame(topics_encoded, columns=topic_columns, index=df.index)
    df = pd.concat([df, topics_df], axis=1)

    # --- Topic text for TF-IDF ---
    def topics_to_text(row):
        active = [col.replace("topic_", "") for col in topic_columns if row[col] == 1]
        return " ".join(active) if active else "no_topics"

    df["topic_text"] = df[topic_columns].apply(topics_to_text, axis=1)

    # --- Date Features ---
    df["training_day"] = pd.to_datetime(df["training_day"], dayfirst=True)
    df["training_year"] = df["training_day"].dt.year
    df["training_month"] = df["training_day"].dt.month
    df["training_day_number"] = df["training_day"].dt.day
    df["training_dayofweek"] = df["training_day"].dt.dayofweek

    # --- Define feature columns ---
    TARGETS = [
        "adopted_within_07_days",
        "adopted_within_90_days",
        "adopted_within_120_days",
    ]
    CAT_COLS = ["county", "sub_county", "ward", "trainer"]
    NUM_COLS = [
        "training_year",
        "training_month",
        "training_day_number",
        "training_dayofweek",
        "number_attendees",
    ]

    # --- TF-IDF ---
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=5)
    tfidf_matrix = tfidf.fit_transform(df["topic_text"])
    tfidf_feature_names = [f"tfidf_{f}" for f in tfidf.get_feature_names_out()]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df.index
    )
    print(f"  TF-IDF features: {len(tfidf_feature_names)}")

    # --- Assemble feature matrices ---
    X_num = df[NUM_COLS].copy()
    X_cat = df[CAT_COLS].copy().astype(str).fillna("NA")
    X_tfidf = tfidf_df.copy()

    total_base = len(NUM_COLS) + len(CAT_COLS) + len(tfidf_feature_names)
    print(f"  Feature breakdown:")
    print(f"    Numeric:     {len(NUM_COLS)}")
    print(f"    Categorical: {len(CAT_COLS)} (target-encoded per fold)")
    print(f"    TF-IDF:      {len(tfidf_feature_names)}")
    print(f"    Total:       {total_base}")
    print(
        f"\n  Note: Binary topic columns ({len(topic_columns)}) and engineered features are NOT included (ablated)."
    )

    return df, X_num, X_cat, X_tfidf, CAT_COLS, NUM_COLS, tfidf, mlb, topic_columns, clean_and_flat_topics, topics_to_text


# ============================================================
# Feature Ablation Study
# ============================================================
def run_ablation(df, X_num, X_cat, X_tfidf, CAT_COLS, topic_columns):
    """Quick ablation to confirm engineered features aren't needed."""
    print("\n" + "=" * 70)
    print("FEATURE ABLATION STUDY")
    print("=" * 70)

    # Build engineered features
    def build_engineered_features(df_in, topic_cols):
        eng = pd.DataFrame(index=df_in.index)
        eng["topic_count"] = df_in[topic_cols].sum(axis=1)

        livestock_kw = [c for c in topic_cols if any(k in c for k in ["cattle", "dairy", "livestock", "poultry", "goat", "sheep", "pig"])]
        crop_kw = [c for c in topic_cols if any(k in c for k in ["crop", "maize", "wheat", "fertilizer", "seed", "irrigation"])]
        business_kw = [c for c in topic_cols if any(k in c for k in ["business", "market", "record", "finance", "insurance"])]
        health_kw = [c for c in topic_cols if any(k in c for k in ["disease", "health", "biosecurity", "vaccination", "parasite"])]
        feed_kw = [c for c in topic_cols if any(k in c for k in ["feed", "nutrition", "fodder", "silage", "hay"])]

        def count_domain(row, kw_cols):
            return sum(row[c] for c in kw_cols if c in row.index)

        eng["livestock_count"] = df_in.apply(lambda r: count_domain(r, livestock_kw), axis=1)
        eng["crop_count"] = df_in.apply(lambda r: count_domain(r, crop_kw), axis=1)
        eng["business_count"] = df_in.apply(lambda r: count_domain(r, business_kw), axis=1)
        eng["health_count"] = df_in.apply(lambda r: count_domain(r, health_kw), axis=1)
        eng["feed_count"] = df_in.apply(lambda r: count_domain(r, feed_kw), axis=1)

        for domain in ["livestock", "crop", "business", "health", "feed"]:
            eng[f"{domain}_pct"] = eng[f"{domain}_count"] / eng["topic_count"].clip(lower=1)

        for domain in ["livestock", "crop", "business", "health", "feed"]:
            eng[f"has_{domain}"] = (eng[f"{domain}_count"] > 0).astype(int)

        domain_cols = [f"{d}_count" for d in ["livestock", "crop", "business", "health", "feed"]]
        eng["num_domains_covered"] = (eng[domain_cols] > 0).sum(axis=1)
        eng["max_domain_count"] = eng[domain_cols].max(axis=1)
        eng["avg_topics_per_domain"] = eng["topic_count"] / eng["num_domains_covered"].clip(lower=1)
        return eng

    print("  Building engineered features for ablation test...")
    eng_features_df = build_engineered_features(df, topic_columns)
    print(f"  Engineered features: {len(eng_features_df.columns)}")

    target = "adopted_within_07_days"
    y_abl = df[target].values
    skf_abl = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for ablation_name, extra_df in [("Base + TF-IDF ONLY", None), ("Base + TF-IDF + Engineered", eng_features_df)]:
        oof_probs = np.zeros(len(df))
        for fold, (train_idx, val_idx) in enumerate(skf_abl.split(np.zeros(len(df)), y_abl)):
            X_train_num = X_num.iloc[train_idx].copy()
            X_val_num = X_num.iloc[val_idx].copy()
            te = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
            X_train_cat_enc = te.fit_transform(X_cat.iloc[train_idx], y_abl[train_idx])
            X_val_cat_enc = te.transform(X_cat.iloc[val_idx])
            X_train_tfidf = X_tfidf.iloc[train_idx].values
            X_val_tfidf = X_tfidf.iloc[val_idx].values
            X_tr = np.hstack([X_train_num.values, X_train_cat_enc.values, X_train_tfidf])
            X_va = np.hstack([X_val_num.values, X_val_cat_enc.values, X_val_tfidf])
            if extra_df is not None:
                X_tr = np.hstack([X_tr, extra_df.iloc[train_idx].values])
                X_va = np.hstack([X_va, extra_df.iloc[val_idx].values])
            rf = RandomForestClassifier(
                n_estimators=300, max_features="sqrt", min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            )
            rf.fit(X_tr, y_abl[train_idx])
            oof_probs[val_idx] = rf.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_abl, oof_probs)
        ll = log_loss(y_abl, oof_probs)
        print(f"  {ablation_name}: AUC={auc:.4f}, LogLoss={ll:.4f}")

    print("\n  → If 'Base + TF-IDF ONLY' is similar or better, engineered features are noise.")
    print("    We proceed WITHOUT engineered features for cleaner models.")


# ============================================================
# Random Forest Model
# ============================================================
def train_rf(df, X_num, X_cat, X_tfidf, CAT_COLS, target_mapping, n_splits, seed):
    """Train Random Forest with 5-fold OOF + SMOTE + Calibration."""
    print("\n" + "=" * 70)
    print("RANDOM FOREST MODEL")
    print("=" * 70)
    t0 = time.time()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rf_results = {}
    rf_models = {}
    rf_target_encoders = {}
    rf_oof_predictions = {}
    rf_chained_probs = pd.DataFrame(index=df.index)

    for period, target in target_mapping.items():
        print(f"\n{'='*30} {period} {'='*30}")
        y = df[target].values
        chain_cols = [c for c in rf_chained_probs.columns]

        oof_probs = np.zeros(len(df))
        fold_models_list = []
        fold_te_list = []

        for fold, (train_idx, val_idx) in enumerate(
            tqdm(skf.split(np.zeros(len(df)), y), total=n_splits, desc=f"  RF {period}")
        ):
            # Target encode categoricals
            te = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
            X_train_cat_enc = te.fit_transform(X_cat.iloc[train_idx], y[train_idx])
            X_val_cat_enc = te.transform(X_cat.iloc[val_idx])

            # Assemble features
            parts_train = [X_num.iloc[train_idx].values, X_train_cat_enc.values, X_tfidf.iloc[train_idx].values]
            parts_val = [X_num.iloc[val_idx].values, X_val_cat_enc.values, X_tfidf.iloc[val_idx].values]
            if len(chain_cols) > 0:
                parts_train.append(rf_chained_probs[chain_cols].iloc[train_idx].values)
                parts_val.append(rf_chained_probs[chain_cols].iloc[val_idx].values)

            X_train = np.hstack(parts_train)
            X_val = np.hstack(parts_val)

            # SMOTE
            smote = SMOTE(random_state=seed, sampling_strategy=0.3)
            try:
                X_train_sm, y_train_sm = smote.fit_resample(X_train, y[train_idx])
            except ValueError:
                X_train_sm, y_train_sm = X_train, y[train_idx]

            # Train RF + Calibration
            rf = RandomForestClassifier(
                n_estimators=800, max_features="sqrt", min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
            )
            calibrated_rf = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)
            calibrated_rf.fit(X_train_sm, y_train_sm)

            # OOF predictions
            fold_probs = calibrated_rf.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = fold_probs
            fold_models_list.append(calibrated_rf)
            fold_te_list.append(te)

            fold_auc = roc_auc_score(y[val_idx], fold_probs)
            print(f"    Fold {fold+1}: AUC={fold_auc:.4f}")

        # Overall OOF metrics
        oof_auc = roc_auc_score(y, oof_probs)
        oof_ll = log_loss(y, oof_probs)
        print(f"  ✓ OOF AUC={oof_auc:.4f}, LogLoss={oof_ll:.4f}")

        rf_chained_probs[target] = oof_probs
        rf_results[period] = {"auc": oof_auc, "logloss": oof_ll}
        rf_models[period] = fold_models_list
        rf_target_encoders[period] = fold_te_list
        rf_oof_predictions[period] = oof_probs

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("RF SUMMARY:")
    for period, res in rf_results.items():
        print(f"  {period}: AUC={res['auc']:.4f}, LogLoss={res['logloss']:.4f}")
    print(f"  ⏱ Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    return rf_results, rf_models, rf_target_encoders, rf_oof_predictions


# ============================================================
# Neural Network Model
# ============================================================
class DigiCowMLP(nn.Module):
    """MLP with architecture h -> h/2 -> h/4 -> h/2 -> 1."""

    def __init__(self, input_dim, h=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.BatchNorm1d(h),
            nn.Dropout(dropout),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.BatchNorm1d(h // 2),
            nn.Dropout(dropout),
            nn.Linear(h // 2, h // 4),
            nn.ReLU(),
            nn.BatchNorm1d(h // 4),
            nn.Dropout(dropout),
            nn.Linear(h // 4, h // 2),
            nn.ReLU(),
            nn.BatchNorm1d(h // 2),
            nn.Dropout(dropout),
            nn.Linear(h // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_nn_model_inner(
    X_train, y_train, X_val, y_val, input_dim, device,
    h=512, lr=1e-3, batch_size=100, max_epochs=2000, patience=50, dropout=0.3,
):
    """Train the baseline MLP with early stopping."""
    model = DigiCowMLP(input_dim, h=h, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    y_va = torch.FloatTensor(y_val).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze()
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va).squeeze()
            val_loss = criterion(val_logits, y_va).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss


def train_nn(df, X_num, X_cat, X_tfidf, CAT_COLS, target_mapping, device, args):
    """Train baseline neural network with 5-fold OOF + SMOTE."""
    print("\n" + "=" * 70)
    print("BASELINE NEURAL NETWORK")
    print("=" * 70)
    t0 = time.time()

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    nn_results = {}
    nn_models = {}
    nn_target_encoders = {}
    nn_oof_predictions = {}
    nn_chained_probs = pd.DataFrame(index=df.index)

    for period, target in target_mapping.items():
        print(f"\n{'='*30} {period} {'='*30}")
        y = df[target].values
        chain_cols = [c for c in nn_chained_probs.columns]

        oof_probs = np.zeros(len(df))
        fold_models_list = []
        fold_te_list = []

        for fold, (train_idx, val_idx) in enumerate(
            tqdm(skf.split(np.zeros(len(df)), y), total=args.n_splits, desc=f"  NN {period}")
        ):
            te = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
            X_train_cat_enc = te.fit_transform(X_cat.iloc[train_idx], y[train_idx])
            X_val_cat_enc = te.transform(X_cat.iloc[val_idx])

            parts_train = [X_num.iloc[train_idx].values, X_train_cat_enc.values, X_tfidf.iloc[train_idx].values]
            parts_val = [X_num.iloc[val_idx].values, X_val_cat_enc.values, X_tfidf.iloc[val_idx].values]
            if len(chain_cols) > 0:
                parts_train.append(nn_chained_probs[chain_cols].iloc[train_idx].values)
                parts_val.append(nn_chained_probs[chain_cols].iloc[val_idx].values)

            X_train = np.hstack(parts_train)
            X_val = np.hstack(parts_val)

            smote = SMOTE(random_state=args.seed, sampling_strategy=0.3)
            try:
                X_train_sm, y_train_sm = smote.fit_resample(X_train, y[train_idx])
            except ValueError:
                X_train_sm, y_train_sm = X_train, y[train_idx]

            input_dim = X_train_sm.shape[1]
            model, val_loss = train_nn_model_inner(
                X_train_sm, y_train_sm, X_val, y[val_idx],
                input_dim=input_dim, device=device,
                h=args.nn_hidden, lr=1e-3, batch_size=args.batch_size,
                max_epochs=args.nn_epochs, patience=args.nn_patience, dropout=0.3,
            )

            with torch.no_grad():
                X_va_t = torch.FloatTensor(X_val).to(device)
                logits = model(X_va_t).squeeze()
                fold_probs = torch.sigmoid(logits).cpu().numpy()

            oof_probs[val_idx] = fold_probs
            fold_models_list.append(model)
            fold_te_list.append(te)

            fold_auc = roc_auc_score(y[val_idx], fold_probs)
            print(f"    Fold {fold+1}: AUC={fold_auc:.4f}, val_loss={val_loss:.4f}")

        oof_auc = roc_auc_score(y, oof_probs)
        oof_ll = log_loss(y, oof_probs)
        print(f"  ✓ OOF AUC={oof_auc:.4f}, LogLoss={oof_ll:.4f}")

        nn_chained_probs[target] = oof_probs
        nn_results[period] = {"auc": oof_auc, "logloss": oof_ll}
        nn_models[period] = fold_models_list
        nn_target_encoders[period] = fold_te_list
        nn_oof_predictions[period] = oof_probs

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("NEURAL NETWORK SUMMARY:")
    for period, res in nn_results.items():
        print(f"  {period}: AUC={res['auc']:.4f}, LogLoss={res['logloss']:.4f}")
    print(f"  ⏱ Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    return nn_results, nn_models, nn_target_encoders, nn_oof_predictions


# ============================================================
# RWN — Random Forest Weighted Neural Network
# ============================================================
def compute_rf_kernel_weights(rf_model, X):
    """Compute RF kernel weights using leaf co-occurrence."""
    leaf_indices = rf_model.apply(X)  # shape: (n_samples, n_trees)
    n_samples, n_trees = leaf_indices.shape

    pair_counts = defaultdict(int)
    for t in tqdm(range(n_trees), desc="    RF kernel trees"):
        leaf_groups = defaultdict(list)
        for i in range(n_samples):
            leaf_groups[leaf_indices[i, t]].append(i)
        for leaf, members in leaf_groups.items():
            if len(members) > 1:
                for ii in range(len(members)):
                    for jj in range(ii + 1, len(members)):
                        key = (members[ii], members[jj])
                        pair_counts[key] += 1

    weights = {}
    for (i, j), count in pair_counts.items():
        w = count / n_trees
        if w > 0.01:
            weights[(i, j)] = w
            weights[(j, i)] = w

    print(f"    RF kernel: {len(weights)//2} unique pairs with w > 0.01")
    return weights


def train_rwn_model_inner(
    X_train, y_train, X_val, y_val, input_dim, rf_weights, train_indices, device,
    tau=None, h=512, lr=1e-3, batch_size=100, max_epochs=2000, patience=50, dropout=0.3,
):
    """Train RWN with combined global + local RF-weighted loss."""
    n = len(X_train)
    if tau is None:
        tau = 1.0 / n

    model = DigiCowMLP(input_dim, h=h, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    y_va = torch.FloatTensor(y_val).to(device)

    # Map global indices to local (training set) indices
    global_to_local = {g: l for l, g in enumerate(train_indices)}

    local_pairs_i = []
    local_pairs_j = []
    local_weights = []
    for (gi, gj), w in rf_weights.items():
        if gi in global_to_local and gj in global_to_local:
            li, lj = global_to_local[gi], global_to_local[gj]
            local_pairs_i.append(li)
            local_pairs_j.append(lj)
            local_weights.append(w)

    if len(local_weights) == 0:
        print("    Warning: No RF weight pairs found, falling back to standard NN")
        tau = 1.0

    pairs_i_t = torch.LongTensor(local_pairs_i).to(device)
    pairs_j_t = torch.LongTensor(local_pairs_j).to(device)
    weights_t = torch.FloatTensor(local_weights).to(device)

    bce_element = nn.BCEWithLogitsLoss(reduction="mean")
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()

        perm = torch.randperm(n)[:batch_size]
        xb = X_tr[perm]
        yb = y_tr[perm]

        optimizer.zero_grad()

        # Global loss: standard BCE
        logits_global = model(xb).squeeze()
        loss_global = bce_element(logits_global, yb)

        # Local RF-weighted loss
        if tau < 1.0 and len(local_weights) > 0:
            n_pairs = min(len(local_weights), batch_size * 10)
            pair_perm = torch.randperm(len(local_weights))[:n_pairs]

            pi = pairs_i_t[pair_perm]
            pj = pairs_j_t[pair_perm]
            pw = weights_t[pair_perm]

            logits_i = model(X_tr[pi]).squeeze()
            targets_j = y_tr[pj]

            per_pair_loss = nn.functional.binary_cross_entropy_with_logits(
                logits_i, targets_j, reduction="none"
            )
            loss_local = (per_pair_loss * pw).mean()
            loss = tau * loss_global + (1 - tau) * loss_local
        else:
            loss = loss_global

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va).squeeze()
            val_loss = bce_element(val_logits, y_va).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss


def train_rwn(df, X_num, X_cat, X_tfidf, CAT_COLS, target_mapping, device, args):
    """Train RWN — Random Forest Weighted Neural Network."""
    print("\n" + "=" * 70)
    print("RWN — RANDOM FOREST WEIGHTED NEURAL NETWORK")
    print("=" * 70)
    t0 = time.time()

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # --- τ Selection via CV on first target ---
    print("\n  Selecting τ via 3-fold CV on 7-day target...")
    first_target = "adopted_within_07_days"
    y_tau = df[first_target].values
    skf_tau = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)

    print("  Training RF for kernel weights...")
    te_full = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
    X_cat_enc_full = te_full.fit_transform(X_cat, y_tau)
    X_full = np.hstack([X_num.values, X_cat_enc_full.values, X_tfidf.values])

    rf_for_kernel = RandomForestClassifier(
        n_estimators=100, max_features="sqrt", min_samples_split=5,
        class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1,
    )
    rf_for_kernel.fit(X_full, y_tau)
    print("  Computing RF kernel weights...")
    rf_weights_full = compute_rf_kernel_weights(rf_for_kernel, X_full)

    n_samples = len(df)
    tau_candidates = [
        1 / (4 * n_samples), 1 / (2 * n_samples), 1 / n_samples,
        2 / n_samples, 4 / n_samples,
    ]

    best_tau = tau_candidates[2]
    best_tau_score = float("inf")

    for tau_cand in tqdm(tau_candidates, desc="  τ search"):
        tau_scores = []
        for fold, (tr_idx, va_idx) in enumerate(skf_tau.split(np.zeros(len(df)), y_tau)):
            te_tmp = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
            X_tr_cat = te_tmp.fit_transform(X_cat.iloc[tr_idx], y_tau[tr_idx])
            X_va_cat = te_tmp.transform(X_cat.iloc[va_idx])

            X_tr = np.hstack([X_num.iloc[tr_idx].values, X_tr_cat.values, X_tfidf.iloc[tr_idx].values])
            X_va = np.hstack([X_num.iloc[va_idx].values, X_va_cat.values, X_tfidf.iloc[va_idx].values])

            smote_tmp = SMOTE(random_state=args.seed, sampling_strategy=0.3)
            try:
                X_tr_sm, y_tr_sm = smote_tmp.fit_resample(X_tr, y_tau[tr_idx])
            except ValueError:
                X_tr_sm, y_tr_sm = X_tr, y_tau[tr_idx]

            model_tmp, _ = train_rwn_model_inner(
                X_tr_sm, y_tr_sm, X_va, y_tau[va_idx],
                input_dim=X_tr_sm.shape[1], rf_weights=rf_weights_full,
                train_indices=tr_idx, device=device, tau=tau_cand, h=256,
                max_epochs=500, patience=20, dropout=0.3,
            )
            with torch.no_grad():
                logits = model_tmp(torch.FloatTensor(X_va).to(device)).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
            tau_scores.append(log_loss(y_tau[va_idx], probs))

        mean_ll = np.mean(tau_scores)
        print(f"    τ={tau_cand:.6f}: LogLoss={mean_ll:.4f}")
        if mean_ll < best_tau_score:
            best_tau_score = mean_ll
            best_tau = tau_cand

    print(f"  → Best τ = {best_tau:.6f} (LogLoss={best_tau_score:.4f})")

    # --- Full OOF Training with best τ ---
    print(f"\n  Training RWN with τ={best_tau:.6f}...")

    rwn_results = {}
    rwn_models = {}
    rwn_target_encoders = {}
    rwn_oof_predictions = {}
    rwn_chained_probs = pd.DataFrame(index=df.index)

    for period, target in target_mapping.items():
        print(f"\n{'='*30} {period} {'='*30}")
        y = df[target].values
        chain_cols = [c for c in rwn_chained_probs.columns]

        # Train RF for this target's kernel weights
        te_rf = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
        X_cat_enc_rf = te_rf.fit_transform(X_cat, y)
        X_rf = np.hstack([X_num.values, X_cat_enc_rf.values, X_tfidf.values])

        rf_kernel = RandomForestClassifier(
            n_estimators=100, max_features="sqrt", min_samples_split=5,
            class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1,
        )
        rf_kernel.fit(X_rf, y)
        rf_weights = compute_rf_kernel_weights(rf_kernel, X_rf)

        oof_probs = np.zeros(len(df))
        fold_models_list = []
        fold_te_list = []

        for fold, (train_idx, val_idx) in enumerate(
            tqdm(skf.split(np.zeros(len(df)), y), total=args.n_splits, desc=f"  RWN {period}")
        ):
            te = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
            X_train_cat_enc = te.fit_transform(X_cat.iloc[train_idx], y[train_idx])
            X_val_cat_enc = te.transform(X_cat.iloc[val_idx])

            parts_train = [X_num.iloc[train_idx].values, X_train_cat_enc.values, X_tfidf.iloc[train_idx].values]
            parts_val = [X_num.iloc[val_idx].values, X_val_cat_enc.values, X_tfidf.iloc[val_idx].values]
            if len(chain_cols) > 0:
                parts_train.append(rwn_chained_probs[chain_cols].iloc[train_idx].values)
                parts_val.append(rwn_chained_probs[chain_cols].iloc[val_idx].values)

            X_train = np.hstack(parts_train)
            X_val = np.hstack(parts_val)

            smote = SMOTE(random_state=args.seed, sampling_strategy=0.3)
            try:
                X_train_sm, y_train_sm = smote.fit_resample(X_train, y[train_idx])
            except ValueError:
                X_train_sm, y_train_sm = X_train, y[train_idx]

            input_dim = X_train_sm.shape[1]
            model, val_loss = train_rwn_model_inner(
                X_train_sm, y_train_sm, X_val, y[val_idx],
                input_dim=input_dim, rf_weights=rf_weights,
                train_indices=train_idx, device=device, tau=best_tau, h=args.nn_hidden,
                max_epochs=args.nn_epochs, patience=args.nn_patience, dropout=0.3,
            )

            with torch.no_grad():
                X_va_t = torch.FloatTensor(X_val).to(device)
                logits = model(X_va_t).squeeze()
                fold_probs = torch.sigmoid(logits).cpu().numpy()

            oof_probs[val_idx] = fold_probs
            fold_models_list.append(model)
            fold_te_list.append(te)

            fold_auc = roc_auc_score(y[val_idx], fold_probs)
            print(f"    Fold {fold+1}: AUC={fold_auc:.4f}, val_loss={val_loss:.4f}")

        oof_auc = roc_auc_score(y, oof_probs)
        oof_ll = log_loss(y, oof_probs)
        print(f"  ✓ OOF AUC={oof_auc:.4f}, LogLoss={oof_ll:.4f}")

        rwn_chained_probs[target] = oof_probs
        rwn_results[period] = {"auc": oof_auc, "logloss": oof_ll}
        rwn_models[period] = fold_models_list
        rwn_target_encoders[period] = fold_te_list
        rwn_oof_predictions[period] = oof_probs

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("RWN SUMMARY:")
    for period, res in rwn_results.items():
        print(f"  {period}: AUC={res['auc']:.4f}, LogLoss={res['logloss']:.4f}")
    print(f"  ⏱ Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    return rwn_results, rwn_models, rwn_target_encoders, rwn_oof_predictions


# ============================================================
# Model Comparison
# ============================================================
def print_comparison(all_results):
    """Print side-by-side model comparison."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<8} {'Period':<10} {'AUC':>8} {'LogLoss':>10}")
    print("-" * 38)
    for period in ["7 Days", "90 Days", "120 Days"]:
        for name, results in all_results.items():
            if results is not None and period in results:
                r = results[period]
                print(f"{name:<8} {period:<10} {r['auc']:>8.4f} {r['logloss']:>10.4f}")
        print()

    print("Average LogLoss:")
    for name, results in all_results.items():
        if results is not None:
            avg_ll = np.mean([r["logloss"] for r in results.values()])
            avg_auc = np.mean([r["auc"] for r in results.values()])
            print(f"  {name}: AUC={avg_auc:.4f}, LogLoss={avg_ll:.4f}")


# ============================================================
# Test Inference & Submission
# ============================================================
def generate_submissions(
    test_df, X_num_cols, CAT_COLS, tfidf, mlb, topic_columns,
    clean_and_flat_topics_fn, topics_to_text_fn,
    all_models, all_te, output_dir, device,
):
    """Generate all submission CSVs."""
    print("\n" + "=" * 70)
    print("TEST INFERENCE & SUBMISSION")
    print("=" * 70)

    # --- Preprocess test data ---
    test_df["clean_topics"] = test_df["topics_list"].apply(clean_and_flat_topics_fn)
    topics_encoded_test = mlb.transform(test_df["clean_topics"])
    topic_columns_list = [f"topic_{t}" for t in mlb.classes_]
    topics_df_test = pd.DataFrame(topics_encoded_test, columns=topic_columns_list, index=test_df.index)
    test_df = pd.concat([test_df, topics_df_test], axis=1)

    test_df["topic_text"] = test_df[topic_columns].apply(topics_to_text_fn, axis=1)

    test_df["training_day"] = pd.to_datetime(test_df["training_day"], dayfirst=True)
    test_df["training_year"] = test_df["training_day"].dt.year
    test_df["training_month"] = test_df["training_day"].dt.month
    test_df["training_day_number"] = test_df["training_day"].dt.day
    test_df["training_dayofweek"] = test_df["training_day"].dt.dayofweek

    NUM_COLS = [
        "training_year", "training_month", "training_day_number",
        "training_dayofweek", "number_attendees",
    ]
    X_test_num = test_df[NUM_COLS].copy()
    X_test_cat = test_df[CAT_COLS].copy().astype(str).fillna("NA")
    X_test_tfidf = tfidf.transform(test_df["topic_text"]).toarray()

    submission_mapping = {
        "7 Days": ["Target_07_AUC", "Target_07_LogLoss"],
        "90 Days": ["Target_90_AUC", "Target_90_LogLoss"],
        "120 Days": ["Target_120_AUC", "Target_120_LogLoss"],
    }
    target_order = [
        ("7 Days", "adopted_within_07_days"),
        ("90 Days", "adopted_within_90_days"),
        ("120 Days", "adopted_within_120_days"),
    ]

    os.makedirs(output_dir, exist_ok=True)

    def generate_submission_sklearn(model_name, models_dict, te_dict):
        submission = pd.DataFrame()
        submission["ID"] = test_df["ID"]
        chained_test_probs = pd.DataFrame(index=test_df.index)

        for period, target in target_order:
            fold_models = models_dict[period]
            fold_tes = te_dict[period]
            chain_cols = [c for c in chained_test_probs.columns]
            fold_predictions = []

            for fold_model, fold_te in zip(fold_models, fold_tes):
                X_test_cat_enc = fold_te.transform(X_test_cat)
                parts = [X_test_num.values, X_test_cat_enc.values, X_test_tfidf]
                if len(chain_cols) > 0:
                    parts.append(chained_test_probs[chain_cols].values)
                X_test_final = np.hstack(parts)
                probs = fold_model.predict_proba(X_test_final)[:, 1]
                fold_predictions.append(probs)

            avg_probs = np.mean(fold_predictions, axis=0)
            for col in submission_mapping[period]:
                submission[col] = avg_probs
            chained_test_probs[target] = avg_probs
            print(f"  {model_name} {period}: min={avg_probs.min():.4f}, max={avg_probs.max():.4f}, mean={avg_probs.mean():.4f}")
        return submission

    def generate_submission_torch(model_name, models_dict, te_dict):
        submission = pd.DataFrame()
        submission["ID"] = test_df["ID"]
        chained_test_probs = pd.DataFrame(index=test_df.index)

        for period, target in target_order:
            fold_models = models_dict[period]
            fold_tes = te_dict[period]
            chain_cols = [c for c in chained_test_probs.columns]
            fold_predictions = []

            for fold_model, fold_te in zip(fold_models, fold_tes):
                X_test_cat_enc = fold_te.transform(X_test_cat)
                parts = [X_test_num.values, X_test_cat_enc.values, X_test_tfidf]
                if len(chain_cols) > 0:
                    parts.append(chained_test_probs[chain_cols].values)
                X_test_final = np.hstack(parts)

                fold_model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_test_final).to(device)
                    logits = fold_model(X_t).squeeze()
                    probs = torch.sigmoid(logits).cpu().numpy()
                fold_predictions.append(probs)

            avg_probs = np.mean(fold_predictions, axis=0)
            for col in submission_mapping[period]:
                submission[col] = avg_probs
            chained_test_probs[target] = avg_probs
            print(f"  {model_name} {period}: min={avg_probs.min():.4f}, max={avg_probs.max():.4f}, mean={avg_probs.mean():.4f}")
        return submission

    # --- Generate submissions for each trained model ---
    for model_name in all_models:
        models_dict = all_models[model_name]
        te_dict = all_te[model_name]

        if model_name == "RF":
            print(f"\nGenerating {model_name} submission...")
            sub = generate_submission_sklearn(model_name, models_dict, te_dict)
        else:
            print(f"\nGenerating {model_name} submission...")
            sub = generate_submission_torch(model_name, models_dict, te_dict)

        out_path = os.path.join(output_dir, f"submission_{model_name.lower()}.csv")
        sub.to_csv(out_path, index=False)
        print(f"  ✓ Saved: {out_path}")

    print(f"\n{'='*70}")
    print("All submissions generated!")
    print("=" * 70)


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    total_t0 = time.time()

    # --- NLTK data ---
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    # --- System info ---
    device = print_system_info()

    # --- Load data ---
    df, combined_df, test_df, TARGETS = load_data(args.data_dir)

    # --- Preprocess ---
    (
        df, X_num, X_cat, X_tfidf, CAT_COLS, NUM_COLS,
        tfidf, mlb, topic_columns, clean_and_flat_topics, topics_to_text,
    ) = preprocess(df, combined_df)

    target_mapping = {
        "7 Days": "adopted_within_07_days",
        "90 Days": "adopted_within_90_days",
        "120 Days": "adopted_within_120_days",
    }

    # --- Ablation ---
    if not args.skip_ablation:
        run_ablation(df, X_num, X_cat, X_tfidf, CAT_COLS, topic_columns)

    # --- Train models ---
    all_results = {}
    all_models = {}
    all_te = {}

    if "rf" in args.models:
        rf_results, rf_models, rf_te, rf_oof = train_rf(
            df, X_num, X_cat, X_tfidf, CAT_COLS, target_mapping, args.n_splits, args.seed,
        )
        all_results["RF"] = rf_results
        all_models["RF"] = rf_models
        all_te["RF"] = rf_te
    else:
        all_results["RF"] = None

    if "nn" in args.models:
        nn_results, nn_models, nn_te, nn_oof = train_nn(
            df, X_num, X_cat, X_tfidf, CAT_COLS, target_mapping, device, args,
        )
        all_results["NN"] = nn_results
        all_models["NN"] = nn_models
        all_te["NN"] = nn_te
    else:
        all_results["NN"] = None

    if "rwn" in args.models:
        rwn_results, rwn_models, rwn_te, rwn_oof = train_rwn(
            df, X_num, X_cat, X_tfidf, CAT_COLS, target_mapping, device, args,
        )
        all_results["RWN"] = rwn_results
        all_models["RWN"] = rwn_models
        all_te["RWN"] = rwn_te
    else:
        all_results["RWN"] = None

    # --- Comparison ---
    print_comparison(all_results)

    # --- Inference & submission ---
    if all_models:
        generate_submissions(
            test_df, NUM_COLS, CAT_COLS, tfidf, mlb, topic_columns,
            clean_and_flat_topics, topics_to_text,
            all_models, all_te, args.output_dir, device,
        )

    # --- Done ---
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"🏁 TOTAL TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
