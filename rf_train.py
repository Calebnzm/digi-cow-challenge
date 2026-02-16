#!/usr/bin/env python3
"""
DigiCow Farmer Training Adoption Challenge — RF Training Script
================================================================
Random Forest with Word2Vec cluster-distance features.
Script version of improved_models.ipynb for easy terminal execution.

Usage:
    python rf_train.py                          # Run with defaults
    python rf_train.py --n-splits 3 --seed 42   # Custom folds/seed
    python rf_train.py --force-k 8              # Force k=8 clusters
    python rf_train.py --output-dir results/    # Save submissions elsewhere
    python rf_train.py --skip-plots             # Skip cluster plots (cloud)

Designed for cloud GPU instances (e.g. Vast.ai RTX 5060 Ti).
"""

import argparse
import ast
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import cosine
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
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
        description="DigiCow Training — RF with Word2Vec cluster features"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Original Data",
        help="Path to data directory containing Train.csv, Prior.csv, Test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save submission CSVs and plots",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=800, help="RF n_estimators (default: 800)"
    )
    parser.add_argument(
        "--w2v-dim", type=int, default=50, help="Word2Vec embedding dimension (default: 50)"
    )
    parser.add_argument(
        "--w2v-epochs", type=int, default=100, help="Word2Vec training epochs (default: 100)"
    )
    parser.add_argument(
        "--max-k", type=int, default=20, help="Max k to search for clustering (default: 20)"
    )
    parser.add_argument(
        "--force-k", type=int, default=None, help="Force a specific k (skip silhouette search)"
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip generating cluster selection plots"
    )
    return parser.parse_args()


# ============================================================
# System Info
# ============================================================
def print_system_info():
    """Print system information."""
    print("=" * 70)
    print("SYSTEM INFO")
    print("=" * 70)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy:  {np.__version__}")
    print(f"  Pandas: {pd.__version__}")

    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU:    {gpu} ({vram:.1f} GB VRAM)")
        else:
            print("  GPU:    None (CPU only)")
    except ImportError:
        print("  GPU:    torch not installed")

    print(f"  CPU:    {os.cpu_count()} cores")
    return


# ============================================================
# Data Loading
# ============================================================
def load_data(data_dir):
    """Load and combine training data."""
    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    df = pd.read_csv(os.path.join(data_dir, "Train.csv"))
    prior_df = pd.read_csv(os.path.join(data_dir, "Prior.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "Test.csv"))

    df["trainer"] = df["trainer"].apply(lambda x: ast.literal_eval(x)[0])
    combined_df = pd.concat([df, prior_df], ignore_index=True)

    TARGETS = [
        "adopted_within_07_days",
        "adopted_within_90_days",
        "adopted_within_120_days",
    ]
    for t in TARGETS:
        df[t] = df[t].astype(int)

    print(f"  Training:  {df.shape}")
    print(f"  Combined:  {combined_df.shape}")
    print(f"  Test:      {test_df.shape}")
    for t in TARGETS:
        print(f"    {t}: {df[t].mean():.4f} ({df[t].sum()}/{len(df)})")

    return df, combined_df, test_df, TARGETS


# ============================================================
# Topic Parsing
# ============================================================
def clean_and_flat_topics(raw):
    """Parse nested topic lists into flat, cleaned list of topic strings."""
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


# ============================================================
# Word2Vec Training
# ============================================================
def train_word2vec(combined_df, unique_topics, args):
    """Train Word2Vec on topic corpus."""
    print("\n" + "=" * 70)
    print("WORD2VEC TRAINING")
    print("=" * 70)
    t0 = time.time()

    # Build corpus
    topic_sentences = [t.split() for t in unique_topics]
    farmer_sentences = []
    for topics in combined_df["clean_topics"]:
        if len(topics) > 0:
            words = []
            for t in topics:
                words.extend(t.split())
            farmer_sentences.append(words)

    all_sentences = topic_sentences + farmer_sentences
    print(f"  Corpus: {len(all_sentences)} sentences")
    print(f"    Topic names: {len(topic_sentences)}")
    print(f"    Farmer contexts: {len(farmer_sentences)}")

    # Train
    w2v = Word2Vec(
        sentences=all_sentences,
        vector_size=args.w2v_dim,
        window=5,
        min_count=1,
        workers=os.cpu_count() or 4,
        epochs=args.w2v_epochs,
        seed=args.seed,
        sg=1,  # Skip-gram
    )

    elapsed = time.time() - t0
    print(f"  Vocabulary: {len(w2v.wv)} words")
    print(f"  Dimensions: {args.w2v_dim}")
    print(f"  ⏱ {elapsed:.1f}s")

    return w2v


# ============================================================
# Topic Embeddings
# ============================================================
def build_topic_embeddings(unique_topics, w2v_model, w2v_dim):
    """Average word vectors per topic to get topic embeddings."""
    topic_embeddings = {}
    for topic in unique_topics:
        words = topic.split()
        vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
        if len(vectors) == 0:
            topic_embeddings[topic] = np.zeros(w2v_dim)
        else:
            topic_embeddings[topic] = np.mean(vectors, axis=0)

    topic_names = list(topic_embeddings.keys())
    topic_vectors = np.array([topic_embeddings[t] for t in topic_names])
    print(f"\n  Topic embeddings: {topic_vectors.shape}")

    return topic_embeddings, topic_names, topic_vectors


# ============================================================
# Optimal k Selection
# ============================================================
def find_optimal_k(topic_vectors, args):
    """Find optimal k using silhouette scores + elbow method."""
    print("\n" + "=" * 70)
    print("OPTIMAL k SELECTION")
    print("=" * 70)

    if args.force_k:
        print(f"  Using forced k={args.force_k}")
        return args.force_k, [], []

    k_range = range(2, args.max_k + 1)
    silhouette_scores = []
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(topic_vectors)
        sil = silhouette_score(topic_vectors, labels)
        silhouette_scores.append(sil)
        inertias.append(kmeans.inertia_)
        print(f"  k={k:2d}: silhouette={sil:.4f}, inertia={kmeans.inertia_:.1f}")

    best_k = list(k_range)[np.argmax(silhouette_scores)]
    best_sil = max(silhouette_scores)
    print(f"\n  → Best k = {best_k} (silhouette = {best_sil:.4f})")

    # Plot
    if not args.skip_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(k_range, silhouette_scores, "b-o", linewidth=2, markersize=6)
            ax1.axvline(x=best_k, color="r", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
            ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
            ax1.set_ylabel("Silhouette Score", fontsize=12)
            ax1.set_title("Silhouette Score vs k", fontsize=14)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            ax2.plot(k_range, inertias, "g-o", linewidth=2, markersize=6)
            ax2.axvline(x=best_k, color="r", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
            ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
            ax2.set_ylabel("Inertia (SSE)", fontsize=12)
            ax2.set_title("Elbow Method", fontsize=14)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

            plt.suptitle(
                f"Optimal Cluster Selection — Best k = {best_k}",
                fontsize=16, fontweight="bold",
            )
            plt.tight_layout()
            plot_path = os.path.join(args.output_dir, "cluster_selection.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Plot saved: {plot_path}")
        except ImportError:
            print("  (matplotlib not available, skipping plot)")

    return best_k, silhouette_scores, inertias


# ============================================================
# Clustering & Distance Features
# ============================================================
def build_cluster_features(topic_vectors, topic_names, topic_embeddings, best_k, seed):
    """Fit K-Means and build cluster info."""
    print(f"\n  Fitting K-Means with k={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=seed, n_init=10, max_iter=300)
    topic_labels = kmeans.fit_predict(topic_vectors)
    centroids = kmeans.cluster_centers_

    print(f"  Cluster contents:")
    for c in range(best_k):
        members = [topic_names[i] for i in range(len(topic_names)) if topic_labels[i] == c]
        preview = members[:5]
        suffix = f"...+{len(members)-5}" if len(members) > 5 else ""
        print(f"    Cluster {c} ({len(members)}): {preview}{suffix}")

    return kmeans, centroids


def compute_farmer_cluster_distances(topics_list, topic_embeddings, centroids, w2v_dim):
    """Compute cosine distance from each farmer's avg embedding to each centroid."""
    k = len(centroids)
    n = len(topics_list)
    distances = np.zeros((n, k))

    for i, topics in enumerate(topics_list):
        topic_vecs = [topic_embeddings.get(t, np.zeros(w2v_dim)) for t in topics]
        valid_vecs = [v for v in topic_vecs if np.any(v != 0)]

        if len(valid_vecs) == 0:
            distances[i, :] = 1.0
        else:
            farmer_embedding = np.mean(valid_vecs, axis=0)
            for c in range(k):
                distances[i, c] = cosine(farmer_embedding, centroids[c])

    return distances


# ============================================================
# Preprocessing Pipeline
# ============================================================
def preprocess(df, combined_df, args):
    """Full preprocessing: topic parsing, Word2Vec, clustering, features."""
    print("\n" + "=" * 70)
    print("PREPROCESSING")
    print("=" * 70)

    # Topic parsing
    df["clean_topics"] = df["topics_list"].apply(clean_and_flat_topics)
    combined_df["clean_topics"] = combined_df["topics_list"].apply(clean_and_flat_topics)

    all_topics_flat = []
    for topics in combined_df["clean_topics"]:
        all_topics_flat.extend(topics)
    unique_topics = sorted(set(all_topics_flat))
    print(f"  Unique topics: {len(unique_topics)}")

    # Word2Vec
    w2v_model = train_word2vec(combined_df, unique_topics, args)

    # Topic embeddings
    topic_embeddings, topic_names, topic_vectors = build_topic_embeddings(
        unique_topics, w2v_model, args.w2v_dim
    )

    # Optimal k
    best_k, sil_scores, inertias = find_optimal_k(topic_vectors, args)

    # K-Means
    kmeans, centroids = build_cluster_features(
        topic_vectors, topic_names, topic_embeddings, best_k, args.seed
    )

    # Cluster distance features for training data
    cluster_distances = compute_farmer_cluster_distances(
        df["clean_topics"].values, topic_embeddings, centroids, args.w2v_dim
    )
    cluster_feature_names = [f"cluster_dist_{c}" for c in range(best_k)]
    cluster_df = pd.DataFrame(cluster_distances, columns=cluster_feature_names, index=df.index)

    # Date features
    df["training_day"] = pd.to_datetime(df["training_day"], dayfirst=True)
    df["training_year"] = df["training_day"].dt.year
    df["training_month"] = df["training_day"].dt.month
    df["training_day_number"] = df["training_day"].dt.day
    df["training_dayofweek"] = df["training_day"].dt.dayofweek

    # Feature columns
    CAT_COLS = [
        "gender", "registration", "age", "group_name",
        "county", "subcounty", "ward", "trainer",
    ]
    NUM_COLS = [
        "belong_to_cooperative", "has_topic_trained_on",
        "training_year", "training_month", "training_day_number", "training_dayofweek",
    ]

    X_num = df[NUM_COLS].copy()
    X_cat = df[CAT_COLS].copy().astype(str).fillna("NA")
    X_cluster = cluster_df.copy()

    print(f"\n  Features: {len(NUM_COLS)} num + {len(CAT_COLS)} cat + {len(cluster_feature_names)} cluster")

    return (
        df, X_num, X_cat, X_cluster, CAT_COLS, NUM_COLS,
        cluster_feature_names, topic_embeddings, centroids, args.w2v_dim,
    )


# ============================================================
# Random Forest Model
# ============================================================
def train_rf(df, X_num, X_cat, X_cluster, CAT_COLS, target_mapping, n_splits, seed, n_estimators=800):
    """Train Random Forest with 5-fold OOF + SMOTE + Calibration."""
    print("\n" + "=" * 70)
    print("RANDOM FOREST")
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
            te = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
            X_train_cat_enc = te.fit_transform(X_cat.iloc[train_idx], y[train_idx])
            X_val_cat_enc = te.transform(X_cat.iloc[val_idx])

            parts_train = [
                X_num.iloc[train_idx].values,
                X_train_cat_enc.values,
                X_cluster.iloc[train_idx].values,
            ]
            parts_val = [
                X_num.iloc[val_idx].values,
                X_val_cat_enc.values,
                X_cluster.iloc[val_idx].values,
            ]
            if len(chain_cols) > 0:
                parts_train.append(rf_chained_probs[chain_cols].iloc[train_idx].values)
                parts_val.append(rf_chained_probs[chain_cols].iloc[val_idx].values)

            X_train = np.hstack(parts_train)
            X_val = np.hstack(parts_val)

            smote = SMOTE(random_state=seed, sampling_strategy=0.3)
            try:
                X_train_sm, y_train_sm = smote.fit_resample(X_train, y[train_idx])
            except ValueError:
                X_train_sm, y_train_sm = X_train, y[train_idx]

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="sqrt",
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
            calibrated_rf = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)
            calibrated_rf.fit(X_train_sm, y_train_sm)

            fold_probs = calibrated_rf.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = fold_probs
            fold_models_list.append(calibrated_rf)
            fold_te_list.append(te)

            fold_auc = roc_auc_score(y[val_idx], fold_probs)
            print(f"    Fold {fold+1}: AUC={fold_auc:.4f}")

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
# Test Inference & Submission
# ============================================================
def generate_submissions(
    test_df, NUM_COLS, CAT_COLS, cluster_feature_names,
    topic_embeddings, centroids, w2v_dim,
    rf_models, rf_te, output_dir,
):
    """Generate submission CSV."""
    print("\n" + "=" * 70)
    print("TEST INFERENCE & SUBMISSION")
    print("=" * 70)

    # --- Preprocess test data ---
    test_df["clean_topics"] = test_df["topics_list"].apply(clean_and_flat_topics)

    # Cluster distance features
    test_cluster_distances = compute_farmer_cluster_distances(
        test_df["clean_topics"].values, topic_embeddings, centroids, w2v_dim
    )
    test_cluster_df = pd.DataFrame(
        test_cluster_distances, columns=cluster_feature_names, index=test_df.index
    )

    # Date features
    test_df["training_day"] = pd.to_datetime(test_df["training_day"], dayfirst=True)
    test_df["training_year"] = test_df["training_day"].dt.year
    test_df["training_month"] = test_df["training_day"].dt.month
    test_df["training_day_number"] = test_df["training_day"].dt.day
    test_df["training_dayofweek"] = test_df["training_day"].dt.dayofweek

    X_test_num = test_df[NUM_COLS].copy()
    X_test_cat = test_df[CAT_COLS].copy().astype(str).fillna("NA")
    X_test_cluster = test_cluster_df.copy()

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

    # --- Generate RF submission ---
    print("\nGenerating RF submission...")
    submission = pd.DataFrame()
    submission["ID"] = test_df["ID"]
    chained_test_probs = pd.DataFrame(index=test_df.index)

    for period, target in target_order:
        fold_models = rf_models[period]
        fold_tes = rf_te[period]
        chain_cols = [c for c in chained_test_probs.columns]
        fold_predictions = []

        for fold_model, fold_te in zip(fold_models, fold_tes):
            X_test_cat_enc = fold_te.transform(X_test_cat)
            parts = [X_test_num.values, X_test_cat_enc.values, X_test_cluster.values]
            if len(chain_cols) > 0:
                parts.append(chained_test_probs[chain_cols].values)
            X_test_final = np.hstack(parts)
            probs = fold_model.predict_proba(X_test_final)[:, 1]
            fold_predictions.append(probs)

        avg_probs = np.mean(fold_predictions, axis=0)
        for col in submission_mapping[period]:
            submission[col] = avg_probs
        chained_test_probs[target] = avg_probs
        print(f"  {period}: min={avg_probs.min():.4f}, max={avg_probs.max():.4f}, mean={avg_probs.mean():.4f}")

    out_path = os.path.join(output_dir, "submission_rf_w2v.csv")
    submission.to_csv(out_path, index=False)
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

    # --- System info ---
    print_system_info()

    # --- Load data ---
    df, combined_df, test_df, TARGETS = load_data(args.data_dir)

    # --- Preprocess (Word2Vec → clustering → features) ---
    (
        df, X_num, X_cat, X_cluster, CAT_COLS, NUM_COLS,
        cluster_feature_names, topic_embeddings, centroids, w2v_dim,
    ) = preprocess(df, combined_df, args)

    target_mapping = {
        "7 Days": "adopted_within_07_days",
        "90 Days": "adopted_within_90_days",
        "120 Days": "adopted_within_120_days",
    }

    # --- Train RF ---
    rf_results, rf_models, rf_te, rf_oof = train_rf(
        df, X_num, X_cat, X_cluster, CAT_COLS, target_mapping,
        args.n_splits, args.seed, args.n_estimators,
    )

    # --- Inference & submission ---
    generate_submissions(
        test_df, NUM_COLS, CAT_COLS, cluster_feature_names,
        topic_embeddings, centroids, w2v_dim,
        rf_models, rf_te, args.output_dir,
    )

    # --- Done ---
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"🏁 TOTAL TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
