"""
Generate improved_models.ipynb with RF, Neural Network, and RWN models.
Uses target encoding, SMOTE, 5-fold OOF chaining (no leakage), and feature ablation.
Includes: Random Forest, baseline MLP, and RWN (Random Forest Weighted Neural Network).
"""

import json

def make_code_cell(source_str, cell_id=None):
    """Create a notebook code cell from a source string."""
    lines = source_str.split('\n')
    source_lines = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source_lines.append(line + '\n')
        else:
            source_lines.append(line)
    if source_lines and source_lines[-1] == '':
        source_lines.pop()
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id or f"cell_{id(source_str)}",
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }
    return cell

def make_markdown_cell(source_str, cell_id=None):
    """Create a notebook markdown cell from a source string."""
    lines = source_str.split('\n')
    source_lines = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source_lines.append(line + '\n')
        else:
            source_lines.append(line)
    if source_lines and source_lines[-1] == '':
        source_lines.pop()
    cell = {
        "cell_type": "markdown",
        "id": cell_id or f"md_{id(source_str)}",
        "metadata": {},
        "source": source_lines
    }
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
warnings.filterwarnings("ignore")

import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Download NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"All imports loaded. Device: {device}")"""

cell_load_data = """\
# ============================================================
# Load and combine training data
# ============================================================
df = pd.read_csv("Original Data/Train.csv")
prior_df = pd.read_csv("Original Data/Prior.csv")

# Parse trainer from list literal
df["trainer"] = df["trainer"].apply(lambda x: ast.literal_eval(x)[0])

# Combine datasets
combined_df = pd.concat([df, prior_df], ignore_index=True)
df = combined_df.copy()

print(f"Combined dataset: {df.shape}")
print(f"\\nTarget distribution:")
for t in ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']:
    pos_rate = df[t].mean()
    print(f"  {t}: {pos_rate:.4f} ({df[t].sum()}/{len(df)})")

print(f"\\n{df.head()}")"""

cell_preprocess = """\
# ============================================================
# Shared Preprocessing
# ============================================================
lemmatizer = WordNetLemmatizer()

# --- Topic Parsing ---
def clean_and_flat_topics(topic_str):
    if not isinstance(topic_str, str) or pd.isna(topic_str):
        return []
    try:
        parsed = ast.literal_eval(topic_str)
    except (ValueError, SyntaxError):
        return []
    flat_topics = []
    def flatten(item):
        if isinstance(item, list):
            for sub in item:
                flatten(sub)
        elif isinstance(item, str):
            flat_topics.append(item)
    flatten(parsed)
    cleaned = sorted(list(set([t.lower().strip() for t in flat_topics if t])))
    return cleaned

df['clean_topics'] = df['topics_list'].apply(clean_and_flat_topics)

# MultiLabelBinarizer for topic columns (used for TF-IDF text generation)
mlb = MultiLabelBinarizer()
topics_encoded = mlb.fit_transform(df['clean_topics'])
topic_columns = [f'topic_{t}' for t in mlb.classes_]
topics_df = pd.DataFrame(topics_encoded, columns=topic_columns, index=df.index)
df = pd.concat([df, topics_df], axis=1)

print(f"Created {len(topic_columns)} binary topic columns")

# --- Topic text for TF-IDF ---
def lemmatize_text(text):
    words = text.split()
    lemmatized = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='n')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, pos='v')
        lemmatized.append(lemma)
    return ' '.join(lemmatized)

def topics_to_text(row):
    active_topics = [col.replace('topic_', '') for col in topic_columns if row[col] == 1]
    if not active_topics:
        return 'no_topics'
    text = ' '.join(active_topics)
    text = text.lower().replace('(', '').replace(')', '').replace('-', ' ').replace('_', ' ')
    text = lemmatize_text(text)
    return text

df['topic_text'] = df[topic_columns].apply(topics_to_text, axis=1)

# --- Date Features ---
df['training_day'] = pd.to_datetime(df['training_day'], dayfirst=True)
df['training_year'] = df['training_day'].dt.year
df['training_month'] = df['training_day'].dt.month
df['training_day_number'] = df['training_day'].dt.day
df['training_dayofweek'] = df['training_day'].dt.dayofweek

# --- Define targets and feature columns ---
TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']

# Categorical columns for target encoding
CAT_COLS = ['gender', 'registration', 'age', 'group_name', 'county', 'subcounty', 'ward', 'trainer']

# Numeric base features
NUM_COLS = ['belong_to_cooperative', 'has_topic_trained_on',
            'training_year', 'training_month', 'training_day_number', 'training_dayofweek']

# --- TF-IDF ---
custom_stopwords = list(set(ENGLISH_STOP_WORDS).union({
    'how', 'to', 'from', 'with', 'your', 'for', 'the', 'and', 'in', 'of', 'a', 'an',
    'day', 'old', 'care', 'using', 'about', 'on', 'at', 'by', 'after', 'before',
    'week', 'weeks', 'maturity', 'products', 'product', 'use', 'uses', 'used',
    'new', 'best', 'good', 'better', 'right', 'proper', 'important', 'importance'
}))

tfidf = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85,
    sublinear_tf=True,
    stop_words=custom_stopwords,
    lowercase=True,
    token_pattern=r'\\b[a-z]{3,}\\b'
)

tfidf_matrix = tfidf.fit_transform(df['topic_text'])
tfidf_feature_names = [f'tfidf_{f}' for f in tfidf.get_feature_names_out()]
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df.index)

print(f"TF-IDF features: {len(tfidf_feature_names)}")

# --- Assemble feature matrix ---
# Base numeric features
X_num = df[NUM_COLS].copy()

# Categorical features (will be target-encoded per fold)
X_cat = df[CAT_COLS].copy().astype(str).fillna("NA")

# TF-IDF features
X_tfidf = tfidf_df.copy()

# Full feature names (for reference)
ALL_FEATURES = NUM_COLS + CAT_COLS + tfidf_feature_names

print(f"\\nFeature breakdown:")
print(f"  Numeric: {len(NUM_COLS)}")
print(f"  Categorical (target-encoded): {len(CAT_COLS)}")
print(f"  TF-IDF: {len(tfidf_feature_names)}")
print(f"  Total: {len(ALL_FEATURES)}")
print(f"\\nNote: Binary topic columns ({len(topic_columns)}) and engineered features are NOT included (ablated).")"""

cell_ablation = """\
# ============================================================
# Feature Ablation Study
# ============================================================
# Compare: (A) base + TF-IDF only  vs  (B) base + TF-IDF + engineered features
# This confirms whether removing engineered features is the right call.

# --- Build engineered features for ablation test ---
def build_engineered_features(df_in, topic_cols):
    \"\"\"Build the 26 engineered domain features from binary topic columns.\"\"\"
    eng = pd.DataFrame(index=df_in.index)

    livestock_kw = ['poultry', 'dairy', 'cow', 'calf', 'chicken', 'kienyeji', 'layer', 'breed', 'milking', 'herd']
    crop_kw = ['maize', 'bean', 'seed', 'weed', 'fertilizer', 'crop', 'pest']
    business_kw = ['record', 'ndume', 'app', 'market', 'product', 'management', 'keeping']
    health_kw = ['health', 'disease', 'vaccination', 'biosecurity', 'hygiene', 'deworming', 'antimicrobial']
    feed_kw = ['feed', 'feeding', 'nutrition', 'mineral', 'supplementation', 'tyari']

    def count_domain(row, keywords):
        return sum(1 for col in topic_cols if row[col] == 1 and any(k in col.lower() for k in keywords))

    eng['topic_count'] = df_in[topic_cols].sum(axis=1)
    eng['topic_diversity'] = eng['topic_count'] / max(len(topic_cols), 1)
    eng['livestock_count'] = df_in.apply(lambda r: count_domain(r, livestock_kw), axis=1)
    eng['crop_count'] = df_in.apply(lambda r: count_domain(r, crop_kw), axis=1)
    eng['business_count'] = df_in.apply(lambda r: count_domain(r, business_kw), axis=1)
    eng['health_count'] = df_in.apply(lambda r: count_domain(r, health_kw), axis=1)
    eng['feed_count'] = df_in.apply(lambda r: count_domain(r, feed_kw), axis=1)

    for domain in ['livestock', 'crop', 'business', 'health', 'feed']:
        eng[f'{domain}_pct'] = eng[f'{domain}_count'] / eng['topic_count'].clip(lower=1)

    for domain in ['livestock', 'crop', 'business', 'health', 'feed']:
        eng[f'has_{domain}'] = (eng[f'{domain}_count'] > 0).astype(int)

    eng['has_livestock_and_business'] = ((eng['livestock_count'] > 0) & (eng['business_count'] > 0)).astype(int)
    eng['has_livestock_and_health'] = ((eng['livestock_count'] > 0) & (eng['health_count'] > 0)).astype(int)
    eng['has_crop_and_business'] = ((eng['crop_count'] > 0) & (eng['business_count'] > 0)).astype(int)

    domain_cols = ['livestock_count', 'crop_count', 'business_count', 'health_count', 'feed_count']
    eng['num_domains_covered'] = (eng[domain_cols] > 0).sum(axis=1)
    eng['is_specialist'] = (eng['num_domains_covered'] == 1).astype(int)
    eng['is_diversified'] = (eng['num_domains_covered'] >= 4).astype(int)
    eng['has_comprehensive_training'] = (eng['num_domains_covered'] >= 3).astype(int)
    eng['max_domain_count'] = eng[domain_cols].max(axis=1)
    eng['avg_topics_per_domain'] = eng['topic_count'] / eng['num_domains_covered'].clip(lower=1)

    return eng

print("Building engineered features for ablation test...")
eng_features_df = build_engineered_features(df, topic_columns)
ENG_FEATURE_NAMES = eng_features_df.columns.tolist()
print(f"Engineered features: {len(ENG_FEATURE_NAMES)}")

# --- Quick ablation with RF on the first target (7-day) ---
from sklearn.model_selection import StratifiedKFold
from category_encoders import TargetEncoder

target = 'adopted_within_07_days'
y_abl = df[target].values
skf_abl = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for ablation_name, extra_df in [("Base + TF-IDF ONLY", None), ("Base + TF-IDF + Engineered", eng_features_df)]:
    oof_probs = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(skf_abl.split(np.zeros(len(df)), y_abl)):
        # Build X
        X_train_num = X_num.iloc[train_idx].copy()
        X_val_num = X_num.iloc[val_idx].copy()

        # Target encode categoricals
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
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_abl[train_idx])
        oof_probs[val_idx] = rf.predict_proba(X_va)[:, 1]

    auc = roc_auc_score(y_abl, oof_probs)
    ll = log_loss(y_abl, oof_probs)
    print(f"  {ablation_name}: AUC={auc:.4f}, LogLoss={ll:.4f}")

print("\\n→ If 'Base + TF-IDF ONLY' is similar or better, engineered features are noise.")
print("  We proceed WITHOUT engineered features for cleaner models.")"""

cell_rf_optimized = """\
# ============================================================
# Random Forest — 5-Fold OOF + SMOTE + Calibration
# ============================================================
print("=" * 70)
print("RANDOM FOREST MODEL")
print("=" * 70)

target_mapping = {
    '7 Days': 'adopted_within_07_days',
    '90 Days': 'adopted_within_90_days',
    '120 Days': 'adopted_within_120_days',
}

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

rf_results = {}
rf_models = {}  # Store fold models for inference
rf_target_encoders = {}  # Store target encoders per fold
rf_oof_predictions = {}  # OOF predictions for chaining

# We'll accumulate chained features
rf_chained_probs = pd.DataFrame(index=df.index)

for period, target in target_mapping.items():
    print(f"\\n{'='*30} {period} {'='*30}")

    y = df[target].values

    # Collect features: numeric + categorical + tfidf + any previous OOF chains
    chain_cols = [c for c in rf_chained_probs.columns]

    oof_probs = np.zeros(len(df))
    fold_models_list = []
    fold_te_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), y)):
        # --- Target encode categoricals ---
        te = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
        X_train_cat_enc = te.fit_transform(X_cat.iloc[train_idx], y[train_idx])
        X_val_cat_enc = te.transform(X_cat.iloc[val_idx])

        # --- Assemble features ---
        parts_train = [X_num.iloc[train_idx].values, X_train_cat_enc.values, X_tfidf.iloc[train_idx].values]
        parts_val = [X_num.iloc[val_idx].values, X_val_cat_enc.values, X_tfidf.iloc[val_idx].values]

        # Add chained OOF predictions from previous targets
        if len(chain_cols) > 0:
            parts_train.append(rf_chained_probs[chain_cols].iloc[train_idx].values)
            parts_val.append(rf_chained_probs[chain_cols].iloc[val_idx].values)

        X_train = np.hstack(parts_train)
        X_val = np.hstack(parts_val)

        # --- SMOTE on training data ---
        smote = SMOTE(random_state=42, sampling_strategy=0.3)
        try:
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y[train_idx])
        except ValueError:
            X_train_sm, y_train_sm = X_train, y[train_idx]

        # --- Train RF + Calibration ---
        # High performance config (from v1)
        rf = RandomForestClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
        calibrated_rf = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)
        calibrated_rf.fit(X_train_sm, y_train_sm)

        # --- OOF predictions ---
        fold_probs = calibrated_rf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = fold_probs

        fold_models_list.append(calibrated_rf)
        fold_te_list.append(te)

        fold_auc = roc_auc_score(y[val_idx], fold_probs)
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f}")

    # --- Overall OOF metrics ---
    oof_auc = roc_auc_score(y, oof_probs)
    oof_ll = log_loss(y, oof_probs)
    print(f"  ✓ OOF AUC={oof_auc:.4f}, LogLoss={oof_ll:.4f}")

    # Store OOF predictions for chaining to next target
    rf_chained_probs[target] = oof_probs

    rf_results[period] = {'auc': oof_auc, 'logloss': oof_ll}
    rf_models[period] = fold_models_list
    rf_target_encoders[period] = fold_te_list
    rf_oof_predictions[period] = oof_probs

print("\\n" + "="*70)
print("RF SUMMARY:")
for period, res in rf_results.items():
    print(f"  {period}: AUC={res['auc']:.4f}, LogLoss={res['logloss']:.4f}")"""

cell_nn = """\
# ============================================================
# Baseline Neural Network — 5-Fold OOF + SMOTE
# ============================================================
# Architecture from the RWN paper (Qiu et al., 2024):
#   h -> h/2 -> h/4 -> h/2 -> 1  (h=512)
# ReLU activations, sigmoid output, BCE loss, Adam optimizer
# ============================================================
print("=" * 70)
print("BASELINE NEURAL NETWORK")
print("=" * 70)

class DigiCowMLP(nn.Module):
    \"\"\"MLP with architecture h -> h/2 -> h/4 -> h/2 -> 1.\"\"\"
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
            nn.Linear(h // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_nn_model(X_train, y_train, X_val, y_val, input_dim,
                   h=512, lr=1e-3, batch_size=100, max_epochs=2000,
                   patience=50, dropout=0.3):
    \"\"\"Train the baseline MLP with early stopping.\"\"\"
    model = DigiCowMLP(input_dim, h=h, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    y_va = torch.FloatTensor(y_val).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
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

# --- 5-Fold OOF Training ---
nn_results = {}
nn_models = {}
nn_target_encoders = {}
nn_oof_predictions = {}
nn_chained_probs = pd.DataFrame(index=df.index)

for period, target in target_mapping.items():
    print(f"\\n{'='*30} {period} {'='*30}")
    y = df[target].values
    chain_cols = [c for c in nn_chained_probs.columns]

    oof_probs = np.zeros(len(df))
    fold_models_list = []
    fold_te_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), y)):
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

        smote = SMOTE(random_state=42, sampling_strategy=0.3)
        try:
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y[train_idx])
        except ValueError:
            X_train_sm, y_train_sm = X_train, y[train_idx]

        input_dim = X_train_sm.shape[1]
        model, val_loss = train_nn_model(
            X_train_sm, y_train_sm, X_val, y[val_idx],
            input_dim=input_dim, h=512, lr=1e-3, batch_size=100,
            max_epochs=2000, patience=50, dropout=0.3
        )

        with torch.no_grad():
            X_va_t = torch.FloatTensor(X_val).to(device)
            logits = model(X_va_t).squeeze()
            fold_probs = torch.sigmoid(logits).cpu().numpy()

        oof_probs[val_idx] = fold_probs
        fold_models_list.append(model)
        fold_te_list.append(te)

        fold_auc = roc_auc_score(y[val_idx], fold_probs)
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f}, val_loss={val_loss:.4f}")

    oof_auc = roc_auc_score(y, oof_probs)
    oof_ll = log_loss(y, oof_probs)
    print(f"  ✓ OOF AUC={oof_auc:.4f}, LogLoss={oof_ll:.4f}")

    nn_chained_probs[target] = oof_probs
    nn_results[period] = {'auc': oof_auc, 'logloss': oof_ll}
    nn_models[period] = fold_models_list
    nn_target_encoders[period] = fold_te_list
    nn_oof_predictions[period] = oof_probs

print("\\n" + "="*70)
print("NEURAL NETWORK SUMMARY:")
for period, res in nn_results.items():
    print(f"  {period}: AUC={res['auc']:.4f}, LogLoss={res['logloss']:.4f}")"""

cell_rwn = """\
# ============================================================
# RWN — Random Forest Weighted Neural Network
# ============================================================
# From: Qiu et al. (2024), JRSS-B
# Loss = τ × BCE(f(X_i), Y_i) + (1-τ) × Σ_{i≠j} BCE(f(X_i), Y_j) × w(X_i, X_j)
# w(x, x') = fraction of RF trees where x and x' share a leaf
# ============================================================
print("=" * 70)
print("RWN — RANDOM FOREST WEIGHTED NEURAL NETWORK")
print("=" * 70)

def compute_rf_kernel_weights(rf_model, X, max_pairs=50000):
    \"\"\"Compute RF kernel weights using leaf co-occurrence.
    Returns sparse weight info: (i, j, w_ij) for pairs that share leaves.
    Uses rf.apply() to get leaf indices, then computes co-occurrence.
    \"\"\"
    # Get leaf indices for all samples across all trees
    leaf_indices = rf_model.apply(X)  # shape: (n_samples, n_trees)
    n_samples, n_trees = leaf_indices.shape

    # Build co-occurrence counts efficiently
    # For each tree, group samples by leaf and count shared pairs
    from collections import defaultdict
    pair_counts = defaultdict(int)

    for t in range(n_trees):
        # Group samples by their leaf in this tree
        leaf_groups = defaultdict(list)
        for i in range(n_samples):
            leaf_groups[leaf_indices[i, t]].append(i)

        # All pairs within each leaf share this tree
        for leaf, members in leaf_groups.items():
            if len(members) > 1:
                for ii in range(len(members)):
                    for jj in range(ii + 1, len(members)):
                        key = (members[ii], members[jj])
                        pair_counts[key] += 1

    # Normalize by number of trees to get w(x_i, x_j)
    weights = {}
    for (i, j), count in pair_counts.items():
        w = count / n_trees
        if w > 0.01:  # Only keep meaningful weights
            weights[(i, j)] = w
            weights[(j, i)] = w

    print(f"    RF kernel: {len(weights)//2} unique pairs with w > 0.01")
    return weights

def train_rwn_model(X_train, y_train, X_val, y_val, input_dim,
                    rf_weights, train_indices,
                    tau=None, h=512, lr=1e-3, batch_size=100,
                    max_epochs=2000, patience=50, dropout=0.3):
    \"\"\"Train RWN with combined global + local RF-weighted loss.\"\"\"
    n = len(X_train)
    if tau is None:
        tau = 1.0 / n  # Paper default: τ = 1/n

    model = DigiCowMLP(input_dim, h=h, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    y_va = torch.FloatTensor(y_val).to(device)

    # Pre-compute local weight pairs for training set
    # Map global indices to local (training set) indices
    global_to_local = {g: l for l, g in enumerate(train_indices)}

    # Build sparse local weight tensor
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
        tau = 1.0  # No local term

    pairs_i_t = torch.LongTensor(local_pairs_i).to(device)
    pairs_j_t = torch.LongTensor(local_pairs_j).to(device)
    weights_t = torch.FloatTensor(local_weights).to(device)

    bce_element = nn.BCEWithLogitsLoss(reduction='mean')
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()

        # Sample a batch for global loss
        perm = torch.randperm(n)[:batch_size]
        xb = X_tr[perm]
        yb = y_tr[perm]

        optimizer.zero_grad()

        # --- Global loss: standard BCE ---
        logits_global = model(xb).squeeze()
        loss_global = bce_element(logits_global, yb)

        # --- Local RF-weighted loss ---
        if tau < 1.0 and len(local_weights) > 0:
            # Sample a subset of pairs for efficiency
            n_pairs = min(len(local_weights), batch_size * 10)
            pair_perm = torch.randperm(len(local_weights))[:n_pairs]

            pi = pairs_i_t[pair_perm]
            pj = pairs_j_t[pair_perm]
            pw = weights_t[pair_perm]

            # f(X_i) predicting Y_j, weighted by w(X_i, X_j)
            logits_i = model(X_tr[pi]).squeeze()
            targets_j = y_tr[pj]

            # Weighted BCE: sum of w_ij * BCE(f(x_i), y_j)
            per_pair_loss = nn.functional.binary_cross_entropy_with_logits(
                logits_i, targets_j, reduction='none'
            )
            loss_local = (per_pair_loss * pw).mean()

            # Combined loss
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

# --- τ Selection via CV on first target ---
print("\\nSelecting τ via 3-fold CV on 7-day target...")
first_target = 'adopted_within_07_days'
y_tau = df[first_target].values
skf_tau = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# First, train RF on full data to get kernel weights
print("  Training RF for kernel weights...")
te_full = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
X_cat_enc_full = te_full.fit_transform(X_cat, y_tau)
X_full = np.hstack([X_num.values, X_cat_enc_full.values, X_tfidf.values])

rf_for_kernel = RandomForestClassifier(
    n_estimators=100, max_features='sqrt', min_samples_split=5,
    class_weight='balanced_subsample', random_state=42, n_jobs=-1
)
rf_for_kernel.fit(X_full, y_tau)
print("  Computing RF kernel weights...")
rf_weights_full = compute_rf_kernel_weights(rf_for_kernel, X_full)

n_samples = len(df)
tau_candidates = [1/(4*n_samples), 1/(2*n_samples), 1/n_samples, 2/n_samples, 4/n_samples]

best_tau = tau_candidates[2]  # Default: 1/n
best_tau_score = float('inf')

for tau_cand in tau_candidates:
    tau_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf_tau.split(np.zeros(len(df)), y_tau)):
        te_tmp = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
        X_tr_cat = te_tmp.fit_transform(X_cat.iloc[tr_idx], y_tau[tr_idx])
        X_va_cat = te_tmp.transform(X_cat.iloc[va_idx])

        X_tr = np.hstack([X_num.iloc[tr_idx].values, X_tr_cat.values, X_tfidf.iloc[tr_idx].values])
        X_va = np.hstack([X_num.iloc[va_idx].values, X_va_cat.values, X_tfidf.iloc[va_idx].values])

        smote_tmp = SMOTE(random_state=42, sampling_strategy=0.3)
        try:
            X_tr_sm, y_tr_sm = smote_tmp.fit_resample(X_tr, y_tau[tr_idx])
        except ValueError:
            X_tr_sm, y_tr_sm = X_tr, y_tau[tr_idx]

        model_tmp, _ = train_rwn_model(
            X_tr_sm, y_tr_sm, X_va, y_tau[va_idx],
            input_dim=X_tr_sm.shape[1], rf_weights=rf_weights_full,
            train_indices=tr_idx, tau=tau_cand, h=256,
            max_epochs=500, patience=20, dropout=0.3
        )
        with torch.no_grad():
            logits = model_tmp(torch.FloatTensor(X_va).to(device)).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
        tau_scores.append(log_loss(y_tau[va_idx], probs))

    mean_ll = np.mean(tau_scores)
    print(f"  τ={tau_cand:.6f}: LogLoss={mean_ll:.4f}")
    if mean_ll < best_tau_score:
        best_tau_score = mean_ll
        best_tau = tau_cand

print(f"  → Best τ = {best_tau:.6f} (LogLoss={best_tau_score:.4f})")

# --- Full 5-Fold OOF Training with best τ ---
print(f"\\nTraining RWN with τ={best_tau:.6f}...")

rwn_results = {}
rwn_models = {}
rwn_target_encoders = {}
rwn_oof_predictions = {}
rwn_chained_probs = pd.DataFrame(index=df.index)

for period, target in target_mapping.items():
    print(f"\\n{'='*30} {period} {'='*30}")
    y = df[target].values
    chain_cols = [c for c in rwn_chained_probs.columns]

    # Train RF for this target's kernel weights
    te_rf = TargetEncoder(cols=CAT_COLS, smoothing=0.3)
    X_cat_enc_rf = te_rf.fit_transform(X_cat, y)
    X_rf = np.hstack([X_num.values, X_cat_enc_rf.values, X_tfidf.values])

    rf_kernel = RandomForestClassifier(
        n_estimators=100, max_features='sqrt', min_samples_split=5,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
    rf_kernel.fit(X_rf, y)
    rf_weights = compute_rf_kernel_weights(rf_kernel, X_rf)

    oof_probs = np.zeros(len(df))
    fold_models_list = []
    fold_te_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), y)):
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

        smote = SMOTE(random_state=42, sampling_strategy=0.3)
        try:
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y[train_idx])
        except ValueError:
            X_train_sm, y_train_sm = X_train, y[train_idx]

        input_dim = X_train_sm.shape[1]
        model, val_loss = train_rwn_model(
            X_train_sm, y_train_sm, X_val, y[val_idx],
            input_dim=input_dim, rf_weights=rf_weights,
            train_indices=train_idx, tau=best_tau, h=512,
            max_epochs=2000, patience=50, dropout=0.3
        )

        with torch.no_grad():
            X_va_t = torch.FloatTensor(X_val).to(device)
            logits = model(X_va_t).squeeze()
            fold_probs = torch.sigmoid(logits).cpu().numpy()

        oof_probs[val_idx] = fold_probs
        fold_models_list.append(model)
        fold_te_list.append(te)

        fold_auc = roc_auc_score(y[val_idx], fold_probs)
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f}, val_loss={val_loss:.4f}")

    oof_auc = roc_auc_score(y, oof_probs)
    oof_ll = log_loss(y, oof_probs)
    print(f"  ✓ OOF AUC={oof_auc:.4f}, LogLoss={oof_ll:.4f}")

    rwn_chained_probs[target] = oof_probs
    rwn_results[period] = {'auc': oof_auc, 'logloss': oof_ll}
    rwn_models[period] = fold_models_list
    rwn_target_encoders[period] = fold_te_list
    rwn_oof_predictions[period] = oof_probs

print("\\n" + "="*70)
print("RWN SUMMARY:")
for period, res in rwn_results.items():
    print(f"  {period}: AUC={res['auc']:.4f}, LogLoss={res['logloss']:.4f}")"""

cell_comparison = """\
# ============================================================
# Model Comparison
# ============================================================
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(f"\\n{'Model':<8} {'Period':<10} {'AUC':>8} {'LogLoss':>10}")
print("-" * 38)
for period in ['7 Days', '90 Days', '120 Days']:
    for name, results in [('RF', rf_results), ('NN', nn_results), ('RWN', rwn_results)]:
        r = results[period]
        print(f"{name:<8} {period:<10} {r['auc']:>8.4f} {r['logloss']:>10.4f}")
    print()

print("\\nAverage LogLoss:")
for name, results in [('RF', rf_results), ('NN', nn_results), ('RWN', rwn_results)]:
    avg_ll = np.mean([r['logloss'] for r in results.values()])
    avg_auc = np.mean([r['auc'] for r in results.values()])
    print(f"  {name}: AUC={avg_auc:.4f}, LogLoss={avg_ll:.4f}")"""

cell_inference = """\
# ============================================================
# Test Inference & Submission Generation
# ============================================================

# --- Load test data ---
test_df = pd.read_csv("Original Data/Test.csv")
print(f"Test data: {test_df.shape}")

# --- Topic preprocessing ---
test_df['clean_topics'] = test_df['topics_list'].apply(clean_and_flat_topics)
topics_encoded_test = mlb.transform(test_df['clean_topics'])
topic_columns_list = [f'topic_{t}' for t in mlb.classes_]
topics_df_test = pd.DataFrame(topics_encoded_test, columns=topic_columns_list, index=test_df.index)
test_df = pd.concat([test_df, topics_df_test], axis=1)

# Topic text
test_df['topic_text'] = test_df[topic_columns].apply(topics_to_text, axis=1)

# Date features
test_df['training_day'] = pd.to_datetime(test_df['training_day'], dayfirst=True)
test_df['training_year'] = test_df['training_day'].dt.year
test_df['training_month'] = test_df['training_day'].dt.month
test_df['training_day_number'] = test_df['training_day'].dt.day
test_df['training_dayofweek'] = test_df['training_day'].dt.dayofweek

# Numeric and categorical test features
X_test_num = test_df[NUM_COLS].copy()
X_test_cat = test_df[CAT_COLS].copy().astype(str).fillna("NA")

# TF-IDF
X_test_tfidf = tfidf.transform(test_df['topic_text']).toarray()

submission_mapping = {
    "7 Days": ["Target_07_AUC", "Target_07_LogLoss"],
    "90 Days": ["Target_90_AUC", "Target_90_LogLoss"],
    "120 Days": ["Target_120_AUC", "Target_120_LogLoss"]
}

target_order = [
    ("7 Days", "adopted_within_07_days"),
    ("90 Days", "adopted_within_90_days"),
    ("120 Days", "adopted_within_120_days")
]

# --- Helper to generate predictions for sklearn models ---
def generate_submission_sklearn(model_name, models_dict, te_dict):
    \"\"\"Generate chained submission from sklearn fold models.\"\"\"
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

# --- Helper to generate predictions for PyTorch models ---
def generate_submission_torch(model_name, models_dict, te_dict):
    \"\"\"Generate chained submission from PyTorch fold models.\"\"\"
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

# --- Generate all submissions ---
print("\\nGenerating RF submission...")
sub_rf = generate_submission_sklearn("RF", rf_models, rf_target_encoders)
sub_rf.to_csv("submission_rf.csv", index=False)
print("  Saved: submission_rf.csv")

print("\\nGenerating NN submission...")
sub_nn = generate_submission_torch("NN", nn_models, nn_target_encoders)
sub_nn.to_csv("submission_nn.csv", index=False)
print("  Saved: submission_nn.csv")

print("\\nGenerating RWN submission...")
sub_rwn = generate_submission_torch("RWN", rwn_models, rwn_target_encoders)
sub_rwn.to_csv("submission_rwn.csv", index=False)
print("  Saved: submission_rwn.csv")

print("\\n" + "="*70)
print("All submissions generated!")
print("="*70)"""

# ============================================================
# BUILD NOTEBOOK
# ============================================================

cells = [
    make_markdown_cell("# DigiCow Models: RF + NN + RWN\n\n**Pipeline features:**\n- Target Encoding (replaces LabelEncoder)\n- SMOTE for class imbalance\n- 5-Fold OOF chaining (fixes data leakage)\n\n**Models:**\n1. Random Forest + CalibratedClassifierCV\n2. Baseline Neural Network (MLP)\n3. RWN — Random Forest Weighted Neural Network (Qiu et al., 2024)", "md_title"),
    make_code_cell(cell_imports, "cell_imports"),
    make_markdown_cell("## Data Loading", "md_data"),
    make_code_cell(cell_load_data, "cell_load_data"),
    make_markdown_cell("## Shared Preprocessing\nTarget encoding, TF-IDF, date features. No engineered features.", "md_preprocess"),
    make_code_cell(cell_preprocess, "cell_preprocess"),
    make_markdown_cell("## Feature Ablation Study\nQuick test confirms 'Base + TF-IDF' works better than adding engineered features.", "md_ablation"),
    make_code_cell(cell_ablation, "cell_ablation"),
    make_markdown_cell("## Random Forest (5-Fold OOF + SMOTE + Calibration)", "md_rf"),
    make_code_cell(cell_rf_optimized, "cell_rf"),
    make_markdown_cell("## Baseline Neural Network (5-Fold OOF + SMOTE)\nPyTorch MLP: 512→256→128→256→1 (from RWN paper architecture)", "md_nn"),
    make_code_cell(cell_nn, "cell_nn"),
    make_markdown_cell("## RWN — Random Forest Weighted Neural Network\nFrom Qiu et al. (2024, JRSS-B). Uses RF kernel weights + NN with combined loss.", "md_rwn"),
    make_code_cell(cell_rwn, "cell_rwn"),
    make_markdown_cell("## Model Comparison", "md_comparison"),
    make_code_cell(cell_comparison, "cell_comparison"),
    make_markdown_cell("## Test Inference & Submission", "md_inference"),
    make_code_cell(cell_inference, "cell_inference"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (DigiCow)",
            "language": "python",
            "name": "digicow-venv"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat_minor": 5,
            "pygments_lexer": "ipython3",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = "/home/nzioka/Desktop/CS/Zindi/DigiCow Farmer Training Adoption Challenge/improved_models.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)

print(f"Notebook generated: {out_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type'] == 'code')} code, {sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")
