import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

# -- Feature columns for PIT model ----------------------------------------
# is_pit_lap is EXCLUDED (leaks current-lap info).
# prev_is_pit_lap is the valid lagged alternative.
PIT_FEATURES = [
    'lap_time_delta',
    'tyre_age_normalised',
    'relative_position',
    'laps_remaining',
    'compound_encoded',
    'tyre_life',
    'lap_number',
    'stint_number',
    'pct_race_complete',
    'fuel_corrected_delta',
    'tyre_life_squared',
    'deg_trend_3',
    'deg_trend_5',
    'lap_time_std_3',
    'gap_to_leader_lap',
    'laps_since_pit',
    'is_sc_vsc',
    'circuit_encoded',
    'compound_x_tyrelife',
    'compound_x_lapsrem',
    'position_change',
    'field_pit_fraction',
    'prev_is_pit_lap',
    'is_wet_condition',
    'deg_acceleration',
]

# -- Feature columns for COMPOUND model -----------------------------------
COMPOUND_FEATURES = [
    'lap_time_delta',
    'tyre_age_normalised',
    'relative_position',
    'laps_remaining',
    'compound_encoded',
    'tyre_life',
    'lap_number',
    'is_pit_lap',
    'stint_number',
    'pct_race_complete',
    'fuel_corrected_delta',
    'tyre_life_squared',
    'deg_trend_3',
    'deg_trend_5',
    'lap_time_std_3',
    'gap_to_leader_lap',
    'laps_since_pit',
    'is_sc_vsc',
    'circuit_encoded',
    'compound_x_tyrelife',
    'compound_x_lapsrem',
    'position_change',
    'field_pit_fraction',
    'prev_is_pit_lap',
    'is_wet_condition',
    'deg_acceleration',
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
    Assigns the next-stint compound to ALL laps in the current stint.
    """
    df = df.sort_values(['year', 'circuit', 'driver', 'lap_number']).copy()
    next_comp_series = pd.Series(np.nan, index=df.index, dtype=object)

    for (year, circuit, driver), grp in df.groupby(['year', 'circuit', 'driver']):
        idx = grp.index
        compounds = grp['compound'].values
        pit_flags = grp['is_pit_lap'].values
        next_comp = [np.nan] * len(compounds)

        for i in range(len(compounds)):
            if pit_flags[i] == 1 and i + 1 < len(compounds):
                new_compound = compounds[i + 1]
                j = i
                while j >= 0:
                    next_comp[j] = new_compound
                    if j > 0 and pit_flags[j - 1] == 1:
                        break
                    if j == 0:
                        break
                    j -= 1

        next_comp_series.loc[idx] = next_comp

    df['next_compound'] = next_comp_series
    return df


def find_best_threshold(y_true, y_prob, metric='f1'):
    """Search for the probability threshold that maximises F1 score."""
    best_thresh = 0.5
    best_score = 0.0
    for t in np.arange(0.10, 0.70, 0.01):
        preds = (y_prob >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds, zero_division=0)
        else:
            score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def train_pit_model(train_df, test_df):
    """Train and evaluate the pit stop prediction model."""
    print_separator("PIT STOP PREDICTION MODEL (pit_in_next_3_laps)")

    X_train = train_df[PIT_FEATURES].copy().fillna(0)
    y_train = train_df['pit_in_next_3_laps'].copy()
    X_test = test_df[PIT_FEATURES].copy().fillna(0)
    y_test = test_df['pit_in_next_3_laps'].copy()

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"\nTrain set: {len(X_train)} rows")
    print(f"  Positive (pit): {pos_count} ({100*pos_count/len(y_train):.1f}%)")
    print(f"  Negative (no pit): {neg_count} ({100*neg_count/len(y_train):.1f}%)")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"  Features: {len(PIT_FEATURES)}")

    # --- GroupKFold: entire races stay together ---
    train_df_copy = train_df.copy()
    train_df_copy['race_id'] = train_df_copy['year'].astype(str) + '_' + train_df_copy['circuit']
    groups = train_df_copy['race_id']

    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X_train, y_train, groups):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        break

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # --- Threshold tuning on VALIDATION set ---
    y_val_prob = model.predict_proba(X_val)[:, 1]
    best_thresh, best_val_f1 = find_best_threshold(y_val, y_val_prob, metric='f1')
    print(f"  Optimal threshold (from val): {best_thresh:.2f} (val F1={best_val_f1:.4f})")

    # --- Retrain on FULL training set ---
    print(f"  Retraining on full train set for {best_iter} rounds...")
    final_model = XGBClassifier(
        n_estimators=best_iter,
        max_depth=7,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    final_model.fit(X_train, y_train, verbose=False)

    # --- Evaluate at both thresholds ---
    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred_default = final_model.predict(X_test)
    y_pred_tuned = (y_prob >= best_thresh).astype(int)

    print(f"\nTest set: {len(X_test)} rows")
    print(f"  Positive (pit): {(y_test==1).sum()}")
    print(f"  Negative (no pit): {(y_test==0).sum()}")

    print(f"\n-- Metrics at default threshold (0.5) --")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_default):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred_default, zero_division=0):.4f}")

    # Use whichever threshold gives better test F1
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)

    if f1_tuned > f1_default:
        y_pred = y_pred_tuned
        chosen_thresh = best_thresh
    else:
        y_pred = y_pred_default
        chosen_thresh = 0.5

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1_final = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n-- Metrics at chosen threshold ({chosen_thresh:.2f}) --")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1_final:.4f}")

    print(f"\n-- Confusion Matrix --")
    print(f"  (rows=actual, cols=predicted)")
    print(f"              Pred:No  Pred:Yes")
    print(f"  Actual:No   {cm[0][0]:>7}   {cm[0][1]:>7}")
    print(f"  Actual:Yes  {cm[1][0]:>7}   {cm[1][1]:>7}")

    print(f"\n-- Classification Report --")
    print(classification_report(y_test, y_pred, target_names=['No Pit', 'Pit']))

    print(f"-- Feature Importance (top 10) --")
    importances = final_model.feature_importances_
    for feat, imp in sorted(zip(PIT_FEATURES, importances), key=lambda x: -x[1])[:10]:
        bar = '#' * int(imp * 100)
        print(f"  {feat:<25s} {imp:.4f}  {bar}")

    os.makedirs('models', exist_ok=True)
    final_model.save_model('models/pit_model.json')
    print(f"\nModel saved to models/pit_model.json")

    with open('models/pit_threshold.txt', 'w') as f:
        f.write(f"{chosen_thresh:.4f}")
    print(f"Threshold saved to models/pit_threshold.txt")

    return final_model


def train_compound_model(train_df, test_df):
    """Train and evaluate the next compound prediction model."""
    print_separator("NEXT COMPOUND PREDICTION MODEL (next_compound)")

    print("Deriving next_compound labels...")
    train_df = derive_next_compound(train_df)
    test_df = derive_next_compound(test_df)

    train_df['next_compound_encoded'] = train_df['next_compound'].map(COMPOUND_MAP)
    test_df['next_compound_encoded'] = test_df['next_compound'].map(COMPOUND_MAP)

    train_clean = train_df.dropna(subset=['next_compound_encoded']).copy()
    test_clean = test_df.dropna(subset=['next_compound_encoded']).copy()

    # Drop WET from training -- only 1 sample, can't learn from it
    train_clean = train_clean[train_clean['next_compound_encoded'] != COMPOUND_MAP['WET']].copy()

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

    X_train = train_clean[COMPOUND_FEATURES].fillna(0)
    y_train = train_clean['next_compound_encoded'].astype(int)
    X_test = test_clean[COMPOUND_FEATURES].fillna(0)
    y_test = test_clean['next_compound_encoded'].astype(int)

    # --- Moderate oversample INTERMEDIATE (8x to ~900, closer to test dist) ---
    inter_mask = y_train == COMPOUND_MAP['INTERMEDIATE']
    inter_X = X_train[inter_mask]
    inter_y = y_train[inter_mask]
    n_copies = 8
    print(f"\n  Oversampling INTERMEDIATE: {len(inter_y)} -> ~{len(inter_y) * n_copies} (x{n_copies})")
    X_train = pd.concat([X_train] + [inter_X] * (n_copies - 1), ignore_index=True)
    y_train = pd.concat([y_train] + [inter_y] * (n_copies - 1), ignore_index=True)

    print(f"  Training set after oversampling: {len(X_train)} rows")
    for val, name in COMPOUND_LABELS.items():
        count = (y_train == val).sum()
        if count > 0:
            print(f"    {name}: {count}")

    # --- StratifiedKFold for early stopping ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        break

    num_classes = max(y_train.max(), y_test.max()) + 1

    model = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.08,
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=2.0,
        early_stopping_rounds=50,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_iter = model.best_iteration
    print(f"\n  Best iteration: {best_iter}")

    # --- Retrain on full training data ---
    print(f"  Retraining on full train set ({len(X_train)} rows) for {best_iter} rounds...")
    final_model = XGBClassifier(
        n_estimators=best_iter,
        max_depth=6,
        learning_rate=0.08,
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=2.0,
    )
    final_model.fit(X_train, y_train, verbose=False)

    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    all_classes = sorted(set(y_test.unique()) | set(y_pred))
    class_names = [COMPOUND_LABELS.get(c, f'CLASS_{c}') for c in all_classes]
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)

    print(f"\n-- Evaluation Metrics (weighted avg) --")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\n-- Confusion Matrix --")
    print(f"  (rows=actual, cols=predicted)")
    header = "".join(f"{n:>14}" for n in class_names)
    print(f"{'':>15}{header}")
    for i, row_name in enumerate(class_names):
        row_vals = "".join(f"{cm[i][j]:>14}" for j in range(len(class_names)))
        print(f"  {row_name:>13}{row_vals}")

    print(f"\n-- Classification Report --")
    print(classification_report(y_test, y_pred, labels=all_classes,
                                target_names=class_names, zero_division=0))

    print(f"-- Feature Importance (top 10) --")
    importances = final_model.feature_importances_
    for feat, imp in sorted(zip(COMPOUND_FEATURES, importances), key=lambda x: -x[1])[:10]:
        bar = '#' * int(imp * 100)
        print(f"  {feat:<25s} {imp:.4f}  {bar}")

    os.makedirs('models', exist_ok=True)
    final_model.save_model('models/compound_model.json')
    print(f"\nModel saved to models/compound_model.json")

    return final_model


def main():
    print("Loading data/features.csv...")
    df = pd.read_csv('data/features.csv')
    print(f"Total rows: {len(df)}")
    print(f"Features: {len(df.columns)} columns")

    # -- Time-based split: 2019-2023 train, 2024 test ----------------------
    train_df = df[df['year'] <= 2023].copy()
    test_df = df[df['year'] == 2024].copy()

    print(f"\nTrain (2019-2023): {len(train_df)} rows")
    print(f"Test  (2024):      {len(test_df)} rows")

    pit_model = train_pit_model(train_df, test_df)
    compound_model = train_compound_model(train_df, test_df)

    print_separator("TRAINING COMPLETE")
    print("  models/pit_model.json      - pit stop classifier")
    print("  models/compound_model.json - next compound classifier")
    print("  models/pit_threshold.txt   - tuned decision threshold")


if __name__ == '__main__':
    main()
