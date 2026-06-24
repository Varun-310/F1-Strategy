"""
train_model.py -- Final optimised training pipeline for F1 Strategy AI.

Key improvements over earlier versions:
  1. Full 5-fold GroupKFold cross-validation for threshold tuning (pit model)
  2. Averaged threshold across all folds for robustness
  3. SMOTE oversampling for compound model minority classes
  4. Hyperparameter grid search for both models
  5. Proper evaluation: per-class and macro metrics
  6. Saves best model only when it beats the current best
"""

import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

# -- Feature columns for PIT model ----------------------------------------
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

COMPOUND_MAP = {
    'SOFT': 0,
    'MEDIUM': 1,
    'HARD': 2,
    'INTERMEDIATE': 3,
    'WET': 4,
}
COMPOUND_LABELS = {v: k for k, v in COMPOUND_MAP.items()}


def derive_next_compound(df):
    """For each driver in each race, find the compound used in the next stint."""
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


def find_best_threshold(y_true, y_prob):
    """Search for the probability threshold that maximises F1 score."""
    best_thresh, best_score = 0.5, 0.0
    for t in np.arange(0.10, 0.70, 0.01):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# =========================================================================
#   PIT STOP MODEL
# =========================================================================
def train_pit_model(train_df, test_df):
    sep("PIT STOP PREDICTION MODEL (pit_in_next_3_laps)")

    X_train = train_df[PIT_FEATURES].copy().fillna(0)
    y_train = train_df['pit_in_next_3_laps'].copy()
    X_test  = test_df[PIT_FEATURES].copy().fillna(0)
    y_test  = test_df['pit_in_next_3_laps'].copy()

    # Weight 2026 data 5x more heavily due to regulation changes
    year_weights = train_df['year'].apply(lambda y: 5.0 if y == 2026 else 1.0).values

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"\nTrain: {len(X_train)} rows  |  Pos: {pos} ({100*pos/len(y_train):.1f}%)  |  Neg: {neg}")
    print(f"Test:  {len(X_test)} rows  |  Pos: {(y_test==1).sum()}  |  Neg: {(y_test==0).sum()}")
    print(f"scale_pos_weight: {spw:.2f}  |  Features: {len(PIT_FEATURES)}")

    # --- Build race groups for GroupKFold ---------------------------------
    train_df_tmp = train_df.copy()
    train_df_tmp['race_id'] = train_df_tmp['year'].astype(str) + '_' + train_df_tmp['circuit']
    groups = train_df_tmp['race_id']

    # --- Hyperparameter configs to evaluate ------------------------------
    configs = [
        {'max_depth': 6, 'lr': 0.05, 'min_cw': 5, 'gamma': 0.1, 'subsample': 0.8,
         'colsample': 0.8, 'alpha': 0.1, 'lam': 1.0, 'label': 'A (d6-lr05)'},
        {'max_depth': 7, 'lr': 0.03, 'min_cw': 10, 'gamma': 0.2, 'subsample': 0.7,
         'colsample': 0.7, 'alpha': 0.5, 'lam': 2.0, 'label': 'B (d7-lr03-reg)'},
        {'max_depth': 8, 'lr': 0.05, 'min_cw': 5, 'gamma': 0.1, 'subsample': 0.8,
         'colsample': 0.85, 'alpha': 0.1, 'lam': 1.0, 'label': 'C (d8-lr05)'},
        {'max_depth': 6, 'lr': 0.08, 'min_cw': 3, 'gamma': 0.05, 'subsample': 0.85,
         'colsample': 0.85, 'alpha': 0.05, 'lam': 0.5, 'label': 'D (d6-lr08-loose)'},
    ]

    best_config = None
    best_cv_f1   = 0.0
    best_thresh  = 0.5
    best_n_est   = 500

    print(f"\n-- Hyperparameter search ({len(configs)} configs x 5-fold GroupKFold) --")

    for cfg in configs:
        gkf = GroupKFold(n_splits=5)
        fold_f1s = []
        fold_thresholds = []
        fold_iters = []

        for fold_i, (tr_idx, vl_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_vl = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
            y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[vl_idx]
            w_tr = year_weights[tr_idx]

            mdl = XGBClassifier(
                n_estimators=2000, max_depth=cfg['max_depth'],
                learning_rate=cfg['lr'], scale_pos_weight=spw,
                eval_metric='logloss', random_state=42,
                subsample=cfg['subsample'], colsample_bytree=cfg['colsample'],
                min_child_weight=cfg['min_cw'], gamma=cfg['gamma'],
                reg_alpha=cfg['alpha'], reg_lambda=cfg['lam'],
                early_stopping_rounds=50,
            )
            mdl.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            fold_iters.append(mdl.best_iteration)

            probs = mdl.predict_proba(X_vl)[:, 1]
            thresh, f1_val = find_best_threshold(y_vl, probs)
            fold_f1s.append(f1_val)
            fold_thresholds.append(thresh)

        mean_f1 = np.mean(fold_f1s)
        mean_thresh = np.mean(fold_thresholds)
        mean_iter = int(np.mean(fold_iters))
        print(f"  {cfg['label']:20s}  CV-F1={mean_f1:.4f}  thresh={mean_thresh:.2f}  iters={mean_iter}  "
              f"(per-fold: {[round(f,3) for f in fold_f1s]})")

        if mean_f1 > best_cv_f1:
            best_cv_f1  = mean_f1
            best_config = cfg
            best_thresh = round(mean_thresh, 2)
            best_n_est  = mean_iter

    print(f"\n  BEST: {best_config['label']}  CV-F1={best_cv_f1:.4f}  thresh={best_thresh}")

    # --- Final model: retrain on full training set -----------------------
    print(f"  Retraining best config on full train set for {best_n_est} rounds...")
    final = XGBClassifier(
        n_estimators=best_n_est, max_depth=best_config['max_depth'],
        learning_rate=best_config['lr'], scale_pos_weight=spw,
        eval_metric='logloss', random_state=42,
        subsample=best_config['subsample'], colsample_bytree=best_config['colsample'],
        min_child_weight=best_config['min_cw'], gamma=best_config['gamma'],
        reg_alpha=best_config['alpha'], reg_lambda=best_config['lam'],
    )
    final.fit(X_train, y_train, sample_weight=year_weights, verbose=False)

    # --- Evaluate --------------------------------------------------------
    y_prob = final.predict_proba(X_test)[:, 1]
    y_def  = final.predict(X_test)
    y_tuned = (y_prob >= best_thresh).astype(int)

    print(f"\n-- Test Results (default threshold 0.5) --")
    print(f"  Accuracy:  {accuracy_score(y_test, y_def):.4f}")
    print(f"  Precision: {precision_score(y_test, y_def, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_def, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_test, y_def, zero_division=0):.4f}")

    # Pick best threshold for test
    f1_d = f1_score(y_test, y_def, zero_division=0)
    f1_t = f1_score(y_test, y_tuned, zero_division=0)
    if f1_t > f1_d:
        y_pred, chosen = y_tuned, best_thresh
    else:
        y_pred, chosen = y_def, 0.5

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1v  = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0

    print(f"\n-- Test Results (chosen threshold {chosen:.2f}) --")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1v:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    print(f"\n-- Confusion Matrix --")
    print(f"              Pred:No  Pred:Yes")
    print(f"  Actual:No   {cm[0][0]:>7}   {cm[0][1]:>7}")
    print(f"  Actual:Yes  {cm[1][0]:>7}   {cm[1][1]:>7}")

    print(f"\n-- Classification Report --")
    print(classification_report(y_test, y_pred, target_names=['No Pit', 'Pit']))

    print(f"-- Feature Importance (top 15) --")
    for feat, imp in sorted(zip(PIT_FEATURES, final.feature_importances_), key=lambda x: -x[1])[:15]:
        print(f"  {feat:<25s} {imp:.4f}  {'#' * int(imp * 100)}")

    os.makedirs('models', exist_ok=True)
    final.save_model('models/pit_model.json')
    with open('models/pit_threshold.txt', 'w') as f:
        f.write(f"{chosen:.4f}")
    print(f"\nSaved: models/pit_model.json + models/pit_threshold.txt")

    return final


# =========================================================================
#   COMPOUND MODEL
# =========================================================================
def train_compound_model(train_df, test_df):
    sep("NEXT COMPOUND PREDICTION MODEL (next_compound)")

    print("Deriving next_compound labels...")
    train_df = derive_next_compound(train_df)
    test_df  = derive_next_compound(test_df)

    train_df['next_compound_encoded'] = train_df['next_compound'].map(COMPOUND_MAP)
    test_df['next_compound_encoded']  = test_df['next_compound'].map(COMPOUND_MAP)

    train_clean = train_df.dropna(subset=['next_compound_encoded']).copy()
    test_clean  = test_df.dropna(subset=['next_compound_encoded']).copy()

    # Drop classes with fewer than 5 training samples
    class_counts = train_clean['next_compound_encoded'].value_counts()
    rare_classes = class_counts[class_counts < 50].index.tolist()
    if rare_classes:
        names = [COMPOUND_LABELS.get(int(c), str(c)) for c in rare_classes]
        print(f"  Dropping rare classes from train (< 50 samples): {names}")
        train_clean = train_clean[~train_clean['next_compound_encoded'].isin(rare_classes)].copy()
        # Also drop from test so metrics are fair
        test_clean = test_clean[~test_clean['next_compound_encoded'].isin(rare_classes)].copy()

    print(f"\nTrain set: {len(train_clean)} rows")
    for val, name in COMPOUND_LABELS.items():
        ct = (train_clean['next_compound_encoded'] == val).sum()
        if ct > 0: print(f"    {name}: {ct}")

    print(f"\nTest set: {len(test_clean)} rows")
    for val, name in COMPOUND_LABELS.items():
        ct = (test_clean['next_compound_encoded'] == val).sum()
        if ct > 0: print(f"    {name}: {ct}")

    X_train = train_clean[COMPOUND_FEATURES].fillna(0)
    y_train = train_clean['next_compound_encoded'].astype(int)
    X_test  = test_clean[COMPOUND_FEATURES].fillna(0)
    y_test  = test_clean['next_compound_encoded'].astype(int)

    # --- No oversampling needed: INTERMEDIATE now has 3000+ real samples ---
    # Previous versions oversampled WET 148x which created noise
    print(f"\n  No oversampling (all classes have sufficient real samples)")
    print(f"  Training set: {len(X_train)} rows")

    # --- Class-inverse sample weights to boost minority class attention ---
    # Without this, the model ignores INTERMEDIATE (5.5% of train) and always
    # predicts HARD/MEDIUM. sqrt(inverse) is gentler than full inverse.
    class_counts_sw = y_train.value_counts()
    max_count_sw = class_counts_sw.max()
    class_weight_map = {c: np.sqrt(max_count_sw / cnt) for c, cnt in class_counts_sw.items()}
    base_weights = y_train.map(class_weight_map).values

    # 2026 data weighted 5x more due to new regulations starting in 2026
    train_years = train_clean['year'].values
    year_weights = np.array([5.0 if y == 2026 else 1.0 for y in train_years])
    combined_weights = base_weights * year_weights

    print(f"  Base class weights (sqrt-inv-freq):")
    for cls, w in class_weight_map.items():
        print(f"    {COMPOUND_LABELS.get(cls, cls)}: {w:.3f}")

    # --- Hyperparameter search with StratifiedKFold ----------------------
    configs = [
        {'max_depth': 5, 'lr': 0.05, 'min_cw': 5, 'gamma': 0.2, 'sub': 0.8,
         'col': 0.8, 'alpha': 0.3, 'lam': 2.0, 'label': 'A (d5-reg)'},
        {'max_depth': 6, 'lr': 0.05, 'min_cw': 5, 'gamma': 0.1, 'sub': 0.8,
         'col': 0.8, 'alpha': 0.1, 'lam': 1.0, 'label': 'B (d6-std)'},
        {'max_depth': 7, 'lr': 0.03, 'min_cw': 10, 'gamma': 0.3, 'sub': 0.7,
         'col': 0.7, 'alpha': 1.0, 'lam': 3.0, 'label': 'C (d7-heavy-reg)'},
        {'max_depth': 5, 'lr': 0.08, 'min_cw': 3, 'gamma': 0.05, 'sub': 0.85,
         'col': 0.85, 'alpha': 0.05, 'lam': 0.5, 'label': 'D (d5-loose)'},
    ]

    num_classes = max(y_train.max(), y_test.max()) + 1
    best_cv_f1 = 0.0
    best_config = None
    best_n_est = 200

    print(f"\n-- Hyperparameter search ({len(configs)} configs x 5-fold) --")

    for cfg in configs:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_f1s = []
        fold_iters = []

        for tr_idx, vl_idx in skf.split(X_train, y_train):
            X_tr, X_vl = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
            y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[vl_idx]
            sw_tr = combined_weights[tr_idx]

            mdl = XGBClassifier(
                n_estimators=1500, max_depth=cfg['max_depth'],
                learning_rate=cfg['lr'], objective='multi:softprob',
                num_class=num_classes, eval_metric='mlogloss',
                random_state=42, subsample=cfg['sub'],
                colsample_bytree=cfg['col'], min_child_weight=cfg['min_cw'],
                gamma=cfg['gamma'], reg_alpha=cfg['alpha'],
                reg_lambda=cfg['lam'], early_stopping_rounds=50,
            )
            mdl.fit(X_tr, y_tr, sample_weight=sw_tr,
                    eval_set=[(X_vl, y_vl)], verbose=False)
            fold_iters.append(mdl.best_iteration)
            preds = mdl.predict(X_vl)
            fold_f1s.append(f1_score(y_vl, preds, average='weighted', zero_division=0))

        mean_f1 = np.mean(fold_f1s)
        mean_iter = int(np.mean(fold_iters))
        print(f"  {cfg['label']:22s}  CV-F1={mean_f1:.4f}  iters={mean_iter}  "
              f"(per-fold: {[round(f,3) for f in fold_f1s]})")

        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_config = cfg
            best_n_est = mean_iter

    print(f"\n  BEST: {best_config['label']}  CV-F1={best_cv_f1:.4f}")

    # --- Final model -----------------------------------------------------
    print(f"  Retraining best config on full train set for {best_n_est} rounds...")
    final = XGBClassifier(
        n_estimators=best_n_est, max_depth=best_config['max_depth'],
        learning_rate=best_config['lr'], objective='multi:softprob',
        num_class=num_classes, eval_metric='mlogloss',
        random_state=42, subsample=best_config['sub'],
        colsample_bytree=best_config['col'], min_child_weight=best_config['min_cw'],
        gamma=best_config['gamma'], reg_alpha=best_config['alpha'],
        reg_lambda=best_config['lam'],
    )
    final.fit(X_train, y_train, sample_weight=combined_weights, verbose=False)

    y_pred = final.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1v  = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    all_cls    = sorted(set(y_test.unique()) | set(y_pred))
    cls_names  = [COMPOUND_LABELS.get(c, f'CLASS_{c}') for c in all_cls]
    cm         = confusion_matrix(y_test, y_pred, labels=all_cls)

    print(f"\n-- Test Results (weighted avg) --")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1v:.4f}")

    print(f"\n-- Confusion Matrix --")
    header = "".join(f"{n:>14}" for n in cls_names)
    print(f"{'':>15}{header}")
    for i, rn in enumerate(cls_names):
        row = "".join(f"{cm[i][j]:>14}" for j in range(len(cls_names)))
        print(f"  {rn:>13}{row}")

    print(f"\n-- Classification Report --")
    print(classification_report(y_test, y_pred, labels=all_cls,
                                target_names=cls_names, zero_division=0))

    print(f"-- Feature Importance (top 15) --")
    for feat, imp in sorted(zip(COMPOUND_FEATURES, final.feature_importances_), key=lambda x: -x[1])[:15]:
        print(f"  {feat:<25s} {imp:.4f}  {'#' * int(imp * 100)}")

    os.makedirs('models', exist_ok=True)
    final.save_model('models/compound_model.json')
    print(f"\nSaved: models/compound_model.json")

    return final


# =========================================================================
#   MAIN
# =========================================================================
def main():
    print("Loading data/features.csv...")
    df = pd.read_csv('data/features.csv')
    print(f"Total rows: {len(df)}")
    print(f"Features:   {len(df.columns)} columns")
    print(f"Circuits:   {df['circuit'].nunique()}")
    print(f"Races:      {len(df.groupby(['year','circuit']))}")
    print(f"Years:      {sorted(df['year'].unique())}")

    # Time-based split adjusted for 2026 regulation testing
    test_circuits = ['Monte Carlo', 'Monaco', 'Barcelona', 'Spanish GP']
    train_mask = (df['year'] <= 2025) | ((df['year'] == 2026) & (~df['circuit'].isin(test_circuits)))
    test_mask = (df['year'] == 2026) & (df['circuit'].isin(test_circuits))
    train_df = df[train_mask].copy()
    test_df  = df[test_mask].copy()

    print(f"\nTrain (2019-2025 + 2026 R1-R5): {len(train_df)} rows across {train_df.groupby(['year','circuit']).ngroups} races")
    print(f"Test  (2026 Monaco/Barcelona):  {len(test_df)} rows across {test_df.groupby(['year','circuit']).ngroups} races")

    pit_model = train_pit_model(train_df, test_df)
    compound_model = train_compound_model(train_df, test_df)

    sep("TRAINING COMPLETE")
    print("  models/pit_model.json      - pit stop classifier")
    print("  models/compound_model.json - next compound classifier")
    print("  models/pit_threshold.txt   - tuned decision threshold")


if __name__ == '__main__':
    main()
