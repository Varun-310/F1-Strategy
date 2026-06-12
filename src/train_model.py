import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# -- Feature columns used by both models ------------------------------------
FEATURES = [
    'lap_time_delta',
    'tyre_age_normalised',
    'relative_position',
    'laps_remaining',
    'compound_encoded',
    'tyre_life',
    'is_pit_lap',
    'lap_number',
]

# Compound encoding (must match feature_engineering.py)
COMPOUND_MAP = {
    'SOFT': 0,
    'MEDIUM': 1,
    'HARD': 2,
    'INTERMEDIATE': 3,
    'WET': 4,
}
COMPOUND_LABELS = {v: k for k, v in COMPOUND_MAP.items()}


def derive_next_compound(df):
    """
    For each driver in each race, find the compound used in the next stint.
    The next stint starts on the lap after a pit stop (is_pit_lap == 1).
    Returns a Series aligned with df's index, with NaN for laps that don't
    have a known next compound (last stint, or no pit stop ahead).
    """
    df = df.sort_values(['year', 'circuit', 'driver', 'lap_number']).copy()
    df['next_compound'] = np.nan

    for (year, circuit, driver), grp in df.groupby(['year', 'circuit', 'driver']):
        idx = grp.index
        compounds = grp['compound'].values
        pit_flags = grp['is_pit_lap'].values
        laps = grp['lap_number'].values

        # Find each stint boundary: when compound changes
        # For each lap, the "next compound" is the compound of the next stint
        next_comp = [np.nan] * len(compounds)

        # Walk through and find each pit stop, then assign next stint compound
        for i in range(len(compounds)):
            if pit_flags[i] == 1:
                # The next lap (i+1) has the new compound
                if i + 1 < len(compounds):
                    new_compound = compounds[i + 1]
                    # Assign this next compound to ALL laps in the current stint
                    # Walk backwards from i to find the start of this stint
                    j = i
                    while j >= 0:
                        next_comp[j] = new_compound
                        # Stop if we hit the start of this stint
                        # (previous lap was a pit lap = start of current stint)
                        if j > 0 and pit_flags[j - 1] == 1:
                            break
                        if j == 0:
                            break
                        j -= 1

        df.loc[idx, 'next_compound'] = next_comp

    return df


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def train_pit_model(train_df, test_df):
    """Train and evaluate the pit stop prediction model."""
    print_separator("PIT STOP PREDICTION MODEL (pit_in_next_3_laps)")

    X_train = train_df[FEATURES].copy()
    y_train = train_df['pit_in_next_3_laps'].copy()
    X_test = test_df[FEATURES].copy()
    y_test = test_df['pit_in_next_3_laps'].copy()

    # Handle any remaining NaNs in features
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Calculate class imbalance ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"\nTrain set: {len(X_train)} rows")
    print(f"  Positive (pit): {pos_count} ({100*pos_count/len(y_train):.1f}%)")
    print(f"  Negative (no pit): {neg_count} ({100*neg_count/len(y_train):.1f}%)")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
    )

    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest set: {len(X_test)} rows")
    print(f"  Positive (pit): {(y_test==1).sum()}")
    print(f"  Negative (no pit): {(y_test==0).sum()}")

    print(f"\n-- Evaluation Metrics --")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\n-- Confusion Matrix --")
    print(f"  (rows=actual, cols=predicted)")
    print(f"              Pred:No  Pred:Yes")
    print(f"  Actual:No   {cm[0][0]:>7}   {cm[0][1]:>7}")
    print(f"  Actual:Yes  {cm[1][0]:>7}   {cm[1][1]:>7}")

    print(f"\n-- Classification Report --")
    print(classification_report(y_test, y_pred, target_names=['No Pit', 'Pit']))

    # Feature importance
    print(f"-- Feature Importance --")
    importances = model.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat:<25s} {imp:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/pit_model.json')
    print(f"\nModel saved to models/pit_model.json")

    return model


def train_compound_model(train_df, test_df):
    """Train and evaluate the next compound prediction model."""
    print_separator("NEXT COMPOUND PREDICTION MODEL (next_compound)")

    # Derive next_compound for both sets
    print("Deriving next_compound labels...")
    train_df = derive_next_compound(train_df)
    test_df = derive_next_compound(test_df)

    # Encode next_compound
    train_df['next_compound_encoded'] = train_df['next_compound'].map(COMPOUND_MAP)
    test_df['next_compound_encoded'] = test_df['next_compound'].map(COMPOUND_MAP)

    # Drop rows without a known next compound
    train_clean = train_df.dropna(subset=['next_compound_encoded']).copy()
    test_clean = test_df.dropna(subset=['next_compound_encoded']).copy()

    print(f"\nTrain set (with next_compound): {len(train_clean)} rows")
    print(f"  Distribution:")
    for val, name in COMPOUND_LABELS.items():
        count = (train_clean['next_compound_encoded'] == val).sum()
        if count > 0:
            print(f"    {name}: {count}")

    print(f"\nTest set (with next_compound): {len(test_clean)} rows")
    print(f"  Distribution:")
    for val, name in COMPOUND_LABELS.items():
        count = (test_clean['next_compound_encoded'] == val).sum()
        if count > 0:
            print(f"    {name}: {count}")

    X_train = train_clean[FEATURES].fillna(0)
    y_train = train_clean['next_compound_encoded'].astype(int)
    X_test = test_clean[FEATURES].fillna(0)
    y_test = test_clean['next_compound_encoded'].astype(int)

    # Determine number of classes present
    num_classes = len(set(y_train.unique()) | set(y_test.unique()))

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=max(num_classes, 5),  # at least 5 for SOFT/MED/HARD/INTER/WET
        eval_metric='mlogloss',
        random_state=42,
    )

    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Build label names for the classes actually present
    all_classes = sorted(set(y_test.unique()) | set(y_pred))
    class_names = [COMPOUND_LABELS.get(c, f'CLASS_{c}') for c in all_classes]

    print(f"\n-- Evaluation Metrics (weighted avg) --")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\n-- Confusion Matrix --")
    print(f"  (rows=actual, cols=predicted)")
    header = "".join(f"{n:>10}" for n in class_names)
    print(f"{'':>15}{header}")
    for i, row_name in enumerate(class_names):
        row_vals = "".join(f"{cm[i][j]:>10}" for j in range(len(class_names)))
        print(f"  {row_name:>13}{row_vals}")

    print(f"\n-- Classification Report --")
    print(classification_report(y_test, y_pred, labels=all_classes,
                                target_names=class_names))

    # Feature importance
    print(f"-- Feature Importance --")
    importances = model.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat:<25s} {imp:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/compound_model.json')
    print(f"\nModel saved to models/compound_model.json")

    return model


def main():
    print("Loading data/features.csv...")
    df = pd.read_csv('data/features.csv')
    print(f"Total rows: {len(df)}")

    # ── Time-based split: 2019-2023 train, 2024 test ─────────────────────
    train_df = df[df['year'] <= 2023].copy()
    test_df = df[df['year'] == 2024].copy()

    print(f"\nTrain (2019-2023): {len(train_df)} rows")
    print(f"Test  (2024):      {len(test_df)} rows")

    # ── Model 1: Pit Stop Prediction ─────────────────────────────────────
    pit_model = train_pit_model(train_df, test_df)

    # ── Model 2: Next Compound Prediction ────────────────────────────────
    compound_model = train_compound_model(train_df, test_df)

    print_separator("TRAINING COMPLETE")
    print("  models/pit_model.json      - pit stop classifier")
    print("  models/compound_model.json - next compound classifier")


if __name__ == '__main__':
    main()
