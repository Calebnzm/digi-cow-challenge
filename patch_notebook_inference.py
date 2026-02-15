
import json
import os

nb_path = "/home/nzioka/Desktop/CS/Zindi/DigiCow Farmer Training Adoption Challenge/random_forest.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new source code for the cell containing preprocess_test_data
# We need to include 'mlb=None' in the signature and add the topic logic.

new_source = [
    "def preprocess_test_data(df_test, FEATURES, label_encoders, mlb=None):\n",
    "    df_test = df_test.copy()\n",
    "    \n",
    "    # --- Date features ---\n",
    "    if 'training_day' in df_test.columns:\n",
    "        df_test['training_day'] = pd.to_datetime(df_test['training_day'], errors='coerce')\n",
    "        df_test['training_year'] = df_test['training_day'].dt.year.fillna(0).astype(int)\n",
    "        df_test['training_month'] = df_test['training_day'].dt.month.fillna(0).astype(int)\n",
    "        df_test['training_day_number'] = df_test['training_day'].dt.day.fillna(0).astype(int)\n",
    "        df_test['training_dayofweek'] = df_test['training_day'].dt.dayofweek.fillna(0).astype(int)\n",
    "    \n",
    "    # --- Topic Features ---\n",
    "    # We need to handle this BEFORE encoding/alignment to ensure new columns are generated\n",
    "    if 'topics_list' in df_test.columns and mlb is not None:\n",
    "        # Use the global clean_and_flat_topics function we defined earlier\n",
    "        df_test['clean_topics'] = df_test['topics_list'].apply(clean_and_flat_topics)\n",
    "        \n",
    "        # Transform using the fitted mlb (from training)\n",
    "        topics_encoded = mlb.transform(df_test['clean_topics'])\n",
    "        topic_columns = [f'topic_{t}' for t in mlb.classes_]\n",
    "        topics_df = pd.DataFrame(topics_encoded, columns=topic_columns, index=df_test.index)\n",
    "        \n",
    "        df_test = pd.concat([df_test, topics_df], axis=1)\n",
    "        print(f\"Created {len(topic_columns)} topic features for test set.\")\n",
    "    \n",
    "    # --- Encode categorical columns safely ---\n",
    "    for col in FEATURES:\n",
    "        if col in df_test.columns:\n",
    "            # print(col) # Reduce verbosity\n",
    "            if col in label_encoders:\n",
    "                le = label_encoders[col]\n",
    "                df_test[col] = df_test[col].astype(str).fillna(\"NA\")\n",
    "                df_test[col] = df_test[col].map(lambda s: s if s in le.classes_ else le.classes_[0])\n",
    "                df_test[col] = le.transform(df_test[col])\n",
    "            else:\n",
    "                # If the feature is still object/string and no encoder exists, fill with 0\n",
    "                if df_test[col].dtype == \"object\":\n",
    "                    df_test[col] = 0\n",
    "    \n",
    "    # --- Ensure all FEATURES exist ---\n",
    "    for feat in FEATURES:\n",
    "        if feat not in df_test.columns:\n",
    "            df_test[feat] = 0\n",
    "\n",
    "    # --- Reorder columns to match training ---\n",
    "    X_test = df_test[FEATURES].copy()\n",
    "    \n",
    "    return X_test\n",
    "\n",
    "# --- Load and preprocess test data ---\n",
    "test_df = pd.read_csv(\"Original Data/Test.csv\")\n",
    "# Update call to pass mlb\n",
    "X_test_aug = preprocess_test_data(test_df, FEATURES, label_encoders, mlb)\n",
    "\n",
    "\n",
    "# --- Prepare submission dataframe ---\n",
    "submission = pd.DataFrame()\n",
    "if \"ID\" in test_df.columns:\n",
    "    submission[\"ID\"] = test_df[\"ID\"]\n",
    "else:\n",
    "    submission[\"ID\"] = range(1, len(test_df) + 1)  # fallback if no ID column\n",
    "\n",
    "# Submission columns mapping\n",
    "submission_mapping = {\n",
    "    \"7 Days\": [\"Target_07_AUC\", \"Target_07_LogLoss\"],\n",
    "    \"90 Days\": [\"Target_90_AUC\", \"Target_90_LogLoss\"],\n",
    "    \"120 Days\": [\"Target_120_AUC\", \"Target_120_LogLoss\"]\n",
    "}\n",
    "\n",
    "# --- Generate chained predictions ---\n",
    "print(\"Generating chained predictions...\")\n",
    "\n",
    "prev_predictions = []\n",
    "\n",
    "for period, target in [\n",
    "    (\"7 Days\", \"adopted_within_07_days\"),\n",
    "    (\"90 Days\", \"adopted_within_90_days\"),\n",
    "    (\"120 Days\", \"adopted_within_120_days\")\n",
    "]:\n",
    "    model = models[period]\n",
    "\n",
    "    # Predict probabilities for current horizon\n",
    "    probs = model.predict_proba(X_test_aug)[:, 1]\n",
    "\n",
    "    # Fill the submission columns\n",
    "    for col in submission_mapping[period]:\n",
    "        submission[col] = probs\n",
    "\n",
    "    # Append prediction as feature for the next horizon\n",
    "    X_test_aug[target] = probs\n",
    "    prev_predictions.append(target)\n",
    "\n",
    "# --- Save submission ---\n",
    "submission.to_csv(\"submission.csv\", index=False)\n",
    "print(\"Done!\")\n",
    "print(submission.head())\n"
]

# Find and replace the cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def preprocess_test_data" in source:
            print("Found preprocess_test_data cell.")
            cell['source'] = new_source
            break

# Save back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
    
print("Notebook updated successfully with inference logic.")
