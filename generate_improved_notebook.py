"""
Generate improved_models.ipynb with RF model using Word2Vec cluster-distance features.
Replaces TF-IDF with: Word2Vec → K-Means clustering → cosine distance features.
"""

import json


def make_code_cell(source_str, cell_id=None):
    """Create a notebook code cell from a source string."""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_str.split("\n"),
    }
    # Fix: add newlines to all lines except the last
    cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
    if cell_id:
        cell["id"] = cell_id
    return cell


def make_markdown_cell(source_str, cell_id=None):
    """Create a notebook markdown cell from a source string."""
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_str.split("\n"),
    }
    cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
    if cell_id:
        cell["id"] = cell_id
    return cell


# ============================================================
# CELL SOURCES
# ============================================================

cell_imports = """\
import pandas as pd
import numpy as np
import ast
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

print("All imports successful ✓")"""

cell_load_data = """\
# ============================================================
# Load and combine training data
# ============================================================
df = pd.read_csv("Original Data/Train.csv")
prior_df = pd.read_csv("Original Data/Prior.csv")
test_df = pd.read_csv("Original Data/Test.csv")

# Parse trainer from list literal
df["trainer"] = df["trainer"].apply(lambda x: ast.literal_eval(x)[0])

# Combine datasets for Word2Vec training (use all available topic text)
combined_df = pd.concat([df, prior_df], ignore_index=True)

# Build target columns
TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
for t in TARGETS:
    df[t] = df[t].astype(int)

print(f"Training data: {df.shape}")
print(f"Combined data: {combined_df.shape}")
print(f"Test data:     {test_df.shape}")

for t in TARGETS:
    pos_rate = df[t].mean()
    print(f"  {t}: {pos_rate:.4f} ({df[t].sum()}/{len(df)})")"""

cell_topic_parsing = """\
# ============================================================
# Topic Parsing & Cleaning
# ============================================================
def clean_and_flat_topics(raw):
    \"\"\"Parse nested topic lists into flat, cleaned list of topic strings.\"\"\"
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

# Apply to training and combined data
df['clean_topics'] = df['topics_list'].apply(clean_and_flat_topics)
combined_df['clean_topics'] = combined_df['topics_list'].apply(clean_and_flat_topics)

# Collect all unique topics
all_topics_flat = []
for topics in combined_df['clean_topics']:
    all_topics_flat.extend(topics)
unique_topics = sorted(set(all_topics_flat))

print(f"Unique topics: {len(unique_topics)}")
print(f"Total topic mentions: {len(all_topics_flat)}")
print(f"Avg topics per farmer: {df['clean_topics'].apply(len).mean():.1f}")
print(f"\\nSample topics:")
for t in unique_topics[:10]:
    print(f"  - {t}")"""

cell_word2vec_training = """\
# ============================================================
# Word2Vec Training
# ============================================================
# Build corpus: each topic name is a "sentence" (list of words)
# Also include farmer topic sets as sentences for co-occurrence context

# 1. Individual topic names as sentences
topic_sentences = [t.split() for t in unique_topics]

# 2. Farmer topic sets as sentences (captures which topics co-occur)
farmer_sentences = []
for topics in combined_df['clean_topics']:
    if len(topics) > 0:
        # Each farmer's topic set as a bag of words
        words = []
        for t in topics:
            words.extend(t.split())
        farmer_sentences.append(words)

all_sentences = topic_sentences + farmer_sentences
print(f"Word2Vec corpus: {len(all_sentences)} sentences")
print(f"  Topic name sentences: {len(topic_sentences)}")
print(f"  Farmer context sentences: {len(farmer_sentences)}")

# Train Word2Vec
W2V_DIM = 50  # Embedding dimension (small corpus needs small dim)
w2v_model = Word2Vec(
    sentences=all_sentences,
    vector_size=W2V_DIM,
    window=5,
    min_count=1,  # Keep all words (small domain vocabulary)
    workers=4,
    epochs=100,   # More epochs for small corpus
    seed=42,
    sg=1,         # Skip-gram (better for small datasets)
)

print(f"\\nWord2Vec trained:")
print(f"  Vocabulary size: {len(w2v_model.wv)}")
print(f"  Embedding dim: {W2V_DIM}")
print(f"  Epochs: 100")"""

cell_topic_embeddings = """\
# ============================================================
# Topic Embeddings
# ============================================================
# For each unique topic, average the word vectors to get a topic embedding

def get_topic_embedding(topic_name, w2v):
    \"\"\"Average word vectors for a topic name.\"\"\"
    words = topic_name.split()
    vectors = [w2v.wv[w] for w in words if w in w2v.wv]
    if len(vectors) == 0:
        return np.zeros(w2v.wv.vector_size)
    return np.mean(vectors, axis=0)

# Build topic embedding matrix
topic_embeddings = {}
for topic in unique_topics:
    topic_embeddings[topic] = get_topic_embedding(topic, w2v_model)

topic_names_list = list(topic_embeddings.keys())
topic_vectors = np.array([topic_embeddings[t] for t in topic_names_list])

print(f"Topic embedding matrix: {topic_vectors.shape}")
print(f"  {len(topic_names_list)} topics × {W2V_DIM} dimensions")

# Quick sanity check: find most similar topics
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(topic_vectors)
np.fill_diagonal(sim_matrix, 0)

print(f"\\nTop 5 most similar topic pairs:")
for _ in range(5):
    i, j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    print(f"  {sim_matrix[i,j]:.3f}: '{topic_names_list[i]}' ↔ '{topic_names_list[j]}'")
    sim_matrix[i, j] = 0
    sim_matrix[j, i] = 0"""

cell_optimal_k = """\
# ============================================================
# Optimal k Selection (Silhouette + Elbow)
# ============================================================
k_range = range(2, 21)
silhouette_scores = []
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(topic_vectors)
    sil = silhouette_score(topic_vectors, labels)
    silhouette_scores.append(sil)
    inertias.append(kmeans.inertia_)
    print(f"  k={k:2d}: silhouette={sil:.4f}, inertia={kmeans.inertia_:.1f}")

best_k = list(k_range)[np.argmax(silhouette_scores)]
best_sil = max(silhouette_scores)
print(f"\\n→ Best k = {best_k} (silhouette = {best_sil:.4f})")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Silhouette scores
ax1.plot(k_range, silhouette_scores, 'b-o', linewidth=2, markersize=6)
ax1.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Silhouette Score', fontsize=12)
ax1.set_title('Silhouette Score vs k', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Elbow curve
ax2.plot(k_range, inertias, 'g-o', linewidth=2, markersize=6)
ax2.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
ax2.set_ylabel('Inertia (SSE)', fontsize=12)
ax2.set_title('Elbow Method', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Optimal Cluster Selection — Best k = {best_k}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_selection.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\\nPlot saved: cluster_selection.png")"""

cell_clustering_and_features = """\
# ============================================================
# K-Means Clustering & Farmer Distance Features
# ============================================================
BEST_K = best_k
print(f"Clustering with k={BEST_K}")

# Fit final K-Means
kmeans_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=10, max_iter=300)
topic_cluster_labels = kmeans_final.fit_predict(topic_vectors)
centroids = kmeans_final.cluster_centers_

# Show cluster contents
print(f"\\nCluster contents:")
for c in range(BEST_K):
    members = [topic_names_list[i] for i in range(len(topic_names_list)) if topic_cluster_labels[i] == c]
    print(f"  Cluster {c} ({len(members)} topics): {members[:5]}{'...' if len(members) > 5 else ''}")

# --- Build farmer distance features ---
def compute_farmer_cluster_distances(topics_list, topic_emb_dict, centroids, w2v_dim):
    \"\"\"For each farmer, average their topic embeddings and compute
    cosine distance to each cluster centroid.
    Returns k distance features.
    \"\"\"
    k = len(centroids)
    n = len(topics_list)
    distances = np.zeros((n, k))

    for i, topics in enumerate(topics_list):
        # Get embeddings for this farmer's topics
        topic_vecs = [topic_emb_dict.get(t, np.zeros(w2v_dim)) for t in topics]
        valid_vecs = [v for v in topic_vecs if np.any(v != 0)]

        if len(valid_vecs) == 0:
            # No valid embeddings — use maximum distance
            distances[i, :] = 1.0
        else:
            farmer_embedding = np.mean(valid_vecs, axis=0)
            for c in range(k):
                distances[i, c] = cosine(farmer_embedding, centroids[c])

    return distances

# Compute for training data
cluster_distances = compute_farmer_cluster_distances(
    df['clean_topics'].values, topic_embeddings, centroids, W2V_DIM
)
cluster_feature_names = [f'cluster_dist_{c}' for c in range(BEST_K)]
cluster_df = pd.DataFrame(cluster_distances, columns=cluster_feature_names, index=df.index)

print(f"\\nCluster distance features: {cluster_df.shape}")
print(f"  Feature names: {cluster_feature_names}")
print(f"\\n  Stats:")
print(cluster_df.describe().round(4))"""

cell_feature_assembly = """\
# ============================================================
# Feature Matrix Assembly
# ============================================================
# Date features
df['training_day'] = pd.to_datetime(df['training_day'], dayfirst=True)
df['training_year'] = df['training_day'].dt.year
df['training_month'] = df['training_day'].dt.month
df['training_day_number'] = df['training_day'].dt.day
df['training_dayofweek'] = df['training_day'].dt.dayofweek

# Define feature columns
CAT_COLS = ['gender', 'registration', 'age', 'group_name', 'county', 'subcounty', 'ward', 'trainer']
NUM_COLS = ['belong_to_cooperative', 'has_topic_trained_on',
            'training_year', 'training_month', 'training_day_number', 'training_dayofweek']

# Assemble
X_num = df[NUM_COLS].copy()
X_cat = df[CAT_COLS].copy().astype(str).fillna("NA")
X_cluster = cluster_df.copy()

ALL_FEATURES = NUM_COLS + CAT_COLS + cluster_feature_names
print(f"Feature breakdown:")
print(f"  Numeric:          {len(NUM_COLS)}")
print(f"  Categorical:      {len(CAT_COLS)} (target-encoded per fold)")
print(f"  Cluster distances: {len(cluster_feature_names)}")
print(f"  Total:            {len(ALL_FEATURES)}")"""

cell_rf_training = """\
# ============================================================
# Random Forest with 5-Fold OOF Chaining
# ============================================================
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

N_SPLITS = 5
SEED = 42

target_mapping = {
    '7 Days': 'adopted_within_07_days',
    '90 Days': 'adopted_within_90_days',
    '120 Days': 'adopted_within_120_days',
}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

rf_results = {}
rf_models = {}
rf_target_encoders = {}
rf_oof_predictions = {}
rf_chained_probs = pd.DataFrame(index=df.index)

for period, target in target_mapping.items():
    print(f"\\n{'='*30} {period} {'='*30}")
    y = df[target].values
    chain_cols = [c for c in rf_chained_probs.columns]

    oof_probs = np.zeros(len(df))
    fold_models_list = []
    fold_te_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), y)):
        # Target encode categoricals
        te = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
        X_train_cat_enc = te.fit_transform(X_cat.iloc[train_idx], y[train_idx])
        X_val_cat_enc = te.transform(X_cat.iloc[val_idx])

        # Assemble features: numeric + encoded cats + cluster distances + chain
        parts_train = [X_num.iloc[train_idx].values, X_train_cat_enc.values, X_cluster.iloc[train_idx].values]
        parts_val = [X_num.iloc[val_idx].values, X_val_cat_enc.values, X_cluster.iloc[val_idx].values]
        if len(chain_cols) > 0:
            parts_train.append(rf_chained_probs[chain_cols].iloc[train_idx].values)
            parts_val.append(rf_chained_probs[chain_cols].iloc[val_idx].values)

        X_train = np.hstack(parts_train)
        X_val = np.hstack(parts_val)

        # SMOTE
        smote = SMOTE(random_state=SEED, sampling_strategy=0.3)
        try:
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y[train_idx])
        except ValueError:
            X_train_sm, y_train_sm = X_train, y[train_idx]

        # Train RF + Calibration
        rf = RandomForestClassifier(
            n_estimators=800, max_features='sqrt', min_samples_leaf=3,
            class_weight='balanced_subsample', random_state=SEED, n_jobs=-1,
        )
        calibrated_rf = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv=3)
        calibrated_rf.fit(X_train_sm, y_train_sm)

        # OOF predictions
        fold_probs = calibrated_rf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = fold_probs
        fold_models_list.append(calibrated_rf)
        fold_te_list.append(te)

        fold_auc = roc_auc_score(y[val_idx], fold_probs)
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f}")

    # Overall OOF metrics
    oof_auc = roc_auc_score(y, oof_probs)
    oof_ll = log_loss(y, oof_probs)
    print(f"  ✓ OOF AUC={oof_auc:.4f}, LogLoss={oof_ll:.4f}")

    rf_chained_probs[target] = oof_probs
    rf_results[period] = {'auc': oof_auc, 'logloss': oof_ll}
    rf_models[period] = fold_models_list
    rf_target_encoders[period] = fold_te_list
    rf_oof_predictions[period] = oof_probs

# --- Summary ---
print(f"\\n{'='*50}")
print("RF RESULTS SUMMARY")
print(f"{'='*50}")
for period, res in rf_results.items():
    print(f"  {period}: AUC={res['auc']:.4f}, LogLoss={res['logloss']:.4f}")"""

cell_inference = """\
# ============================================================
# Test Inference & Submission
# ============================================================
# Preprocess test data with same pipeline

# Topic parsing
test_df['clean_topics'] = test_df['topics_list'].apply(clean_and_flat_topics)

# Compute cluster distance features for test data
test_cluster_distances = compute_farmer_cluster_distances(
    test_df['clean_topics'].values, topic_embeddings, centroids, W2V_DIM
)
test_cluster_df = pd.DataFrame(test_cluster_distances, columns=cluster_feature_names, index=test_df.index)

# Date features
test_df['training_day'] = pd.to_datetime(test_df['training_day'], dayfirst=True)
test_df['training_year'] = test_df['training_day'].dt.year
test_df['training_month'] = test_df['training_day'].dt.month
test_df['training_day_number'] = test_df['training_day'].dt.day
test_df['training_dayofweek'] = test_df['training_day'].dt.dayofweek

X_test_num = test_df[NUM_COLS].copy()
X_test_cat = test_df[CAT_COLS].copy().astype(str).fillna("NA")
X_test_cluster = test_cluster_df.copy()

# Generate submission
submission = pd.DataFrame()
submission['ID'] = test_df['ID']

submission_mapping = {
    '7 Days': ['Target_07_AUC', 'Target_07_LogLoss'],
    '90 Days': ['Target_90_AUC', 'Target_90_LogLoss'],
    '120 Days': ['Target_120_AUC', 'Target_120_LogLoss'],
}

target_order = [
    ('7 Days', 'adopted_within_07_days'),
    ('90 Days', 'adopted_within_90_days'),
    ('120 Days', 'adopted_within_120_days'),
]

chained_test_probs = pd.DataFrame(index=test_df.index)

for period, target in target_order:
    fold_models = rf_models[period]
    fold_tes = rf_target_encoders[period]
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

submission.to_csv('submission_rf_w2v.csv', index=False)
print(f"\\n✓ Saved: submission_rf_w2v.csv")
print(f"  Shape: {submission.shape}")
print(f"\\nSubmission preview:")
submission.head()"""


# ============================================================
# Build Notebook
# ============================================================
cells = [
    make_markdown_cell("# DigiCow Farmer Training Adoption Challenge\n## Random Forest with Word2Vec Cluster-Distance Features\n\n**Pipeline:** Topic names → Word2Vec → K-Means clustering → Cosine distance features → RF"),
    make_code_cell(cell_imports, "imports"),
    make_markdown_cell("## 1. Data Loading"),
    make_code_cell(cell_load_data, "load_data"),
    make_markdown_cell("## 2. Topic Parsing"),
    make_code_cell(cell_topic_parsing, "topic_parsing"),
    make_markdown_cell("## 3. Word2Vec Training\nTrain on topic names + farmer topic sets to capture co-occurrence patterns."),
    make_code_cell(cell_word2vec_training, "word2vec"),
    make_markdown_cell("## 4. Topic Embeddings\nAverage word vectors per topic name to get 149 topic embedding vectors."),
    make_code_cell(cell_topic_embeddings, "topic_embeddings"),
    make_markdown_cell("## 5. Optimal k Selection\nFind the best number of clusters using **silhouette scores** (higher = better) and **elbow method** (look for the bend)."),
    make_code_cell(cell_optimal_k, "optimal_k"),
    make_markdown_cell("## 6. Clustering & Distance Features\nCluster topics, then compute cosine distance from each farmer's average embedding to centroids."),
    make_code_cell(cell_clustering_and_features, "clustering"),
    make_markdown_cell("## 7. Feature Assembly"),
    make_code_cell(cell_feature_assembly, "features"),
    make_markdown_cell("## 8. Random Forest Training\n5-fold OOF chaining with SMOTE + calibration."),
    make_code_cell(cell_rf_training, "rf_training"),
    make_markdown_cell("## 9. Test Inference & Submission"),
    make_code_cell(cell_inference, "inference"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("improved_models.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("✓ Generated improved_models.ipynb")
print("  Cells:", len(cells))
print("  Pipeline: Word2Vec → K-Means → Cosine Distance → RF")
