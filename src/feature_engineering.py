import pandas as pd
import numpy as np

# Average stint lengths per compound (approximate from F1 data)
AVG_STINT_LENGTH = {
    'SOFT': 18,
    'MEDIUM': 28,
    'HARD': 38,
    'INTERMEDIATE': 20,
    'INTER': 20,
    'WET': 15,
}
TYRE_AGE_CAP = 1.5

# Compound encoding
COMPOUND_MAP = {
    'SOFT': 0,
    'MEDIUM': 1,
    'HARD': 2,
    'INTERMEDIATE': 3,
    'WET': 4,
}

# Approximate fuel-effect: each lap the car gets ~0.06s faster from fuel burn
FUEL_EFFECT_PER_LAP = 0.06


def engineer_features():
    """Load historical laps, compute derived features and the pit label, save to CSV."""
    df = pd.read_csv('data/historical_laps.csv')
    print(f"Loaded {len(df)} rows from data/historical_laps.csv")

    # -- Remove placeholder rows from cancelled races ----------------------
    df = df[df['driver'] != 'NONE'].copy()

    # -- Drop rows with null lap times ------------------------------------
    df = df.dropna(subset=['lap_time_seconds']).copy()
    print(f"After dropping null lap times: {len(df)} rows")

    # -- Basic info --------------------------------------------------------
    print(f"Unique circuits: {df['circuit'].nunique()}")
    print(f"Unique races: {len(df.groupby(['year', 'circuit']))}")
    print(f"Years: {sorted(df['year'].unique())}")

    # -- Ensure correct types ----------------------------------------------
    df['lap_number'] = df['lap_number'].astype(int)
    df['is_pit_lap'] = df['is_pit_lap'].astype(int)

    # -- Sort consistently for rolling / shift operations ------------------
    df = df.sort_values(['year', 'circuit', 'driver', 'lap_number']).reset_index(drop=True)

    # Define group key for per-driver-per-race operations
    grp_key = ['year', 'circuit', 'driver']

    # =====================================================================
    # DERIVE TOTAL LAPS PER RACE DYNAMICALLY
    # =====================================================================
    # Use the maximum lap number across all drivers in each race
    race_max_laps = (
        df.groupby(['year', 'circuit'])['lap_number']
        .transform('max')
    )
    # Use at least the max observed, default to 57 (median F1 race length)
    df['total_laps_race'] = race_max_laps.clip(lower=10)

    # =====================================================================
    # DERIVE CIRCUIT ENCODING DYNAMICALLY
    # =====================================================================
    circuit_labels = sorted(df['circuit'].unique())
    circuit_map = {name: idx for idx, name in enumerate(circuit_labels)}
    print(f"Circuit encoding: {len(circuit_map)} circuits")

    # =====================================================================
    # FEATURES
    # =====================================================================

    # -- 1. lap_time_delta -------------------------------------------------
    race_driver_median = (
        df.groupby(grp_key)['lap_time_seconds'].transform('median')
    )
    df['lap_time_delta'] = df['lap_time_seconds'] - race_driver_median

    # -- 2. tyre_age_normalised --------------------------------------------
    df['tyre_age_normalised'] = df.apply(
        lambda row: min(
            row['tyre_life'] / AVG_STINT_LENGTH.get(row['compound'], 28),
            TYRE_AGE_CAP
        ) if pd.notna(row['tyre_life']) and pd.notna(row['compound']) else np.nan,
        axis=1
    )

    # -- 3. relative_position ----------------------------------------------
    df['cum_time'] = df.groupby(grp_key)['lap_time_seconds'].cumsum()
    n_drivers_per_lap = df.groupby(['year', 'circuit', 'lap_number'])['driver'].transform('count')
    df['position_approx'] = (
        df.groupby(['year', 'circuit', 'lap_number'])['cum_time']
        .rank(method='min')
    )
    df['relative_position'] = df['position_approx'] / n_drivers_per_lap
    df.drop(columns=['cum_time', 'position_approx'], inplace=True)

    # -- 4. laps_remaining -------------------------------------------------
    df['laps_remaining'] = df['total_laps_race'] - df['lap_number']

    # -- 5. compound_encoded -----------------------------------------------
    df['compound_encoded'] = df['compound'].map(COMPOUND_MAP).fillna(-1).astype(int)

    # -- 6. stint_number ---------------------------------------------------
    df['stint_number'] = df.groupby(grp_key)['is_pit_lap'].cumsum() + 1

    # -- 7. pct_race_complete ----------------------------------------------
    df['pct_race_complete'] = df['lap_number'] / df['total_laps_race']

    # -- 8. fuel_corrected_delta -------------------------------------------
    df['fuel_corrected_delta'] = df['lap_time_delta'] + (FUEL_EFFECT_PER_LAP * df['lap_number'])

    # -- 9. tyre_life_squared (captures non-linear degradation) ------------
    df['tyre_life_squared'] = df['tyre_life'] ** 2

    # -- 10. degradation trend: rolling mean last 3 laps -------------------
    df['deg_trend_3'] = (
        df.groupby(grp_key)['lap_time_delta']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # -- 11. degradation trend: rolling mean last 5 laps -------------------
    df['deg_trend_5'] = (
        df.groupby(grp_key)['lap_time_delta']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # -- 12. lap time consistency: rolling std over last 3 laps ------------
    df['lap_time_std_3'] = (
        df.groupby(grp_key)['lap_time_seconds']
        .transform(lambda x: x.rolling(3, min_periods=2).std())
    )

    # -- 13. delta to race leader lap time ---------------------------------
    min_lap_time = (
        df.groupby(['year', 'circuit', 'lap_number'])['lap_time_seconds']
        .transform('min')
    )
    df['gap_to_leader_lap'] = df['lap_time_seconds'] - min_lap_time

    # -- 14. laps since last pit -------------------------------------------
    df['laps_since_pit'] = 0
    for _, grp in df.groupby(grp_key):
        idx = grp.index
        pit_flags = grp['is_pit_lap'].values
        laps_since = np.zeros(len(pit_flags), dtype=int)
        counter = 999
        for i in range(len(pit_flags)):
            if i > 0 and pit_flags[i - 1] == 1:
                counter = 0
            counter += 1
            laps_since[i] = counter
        df.loc[idx, 'laps_since_pit'] = laps_since

    # -- 15. track_status cleaned ------------------------------------------
    df['track_status_raw'] = pd.to_numeric(df['track_status'], errors='coerce').fillna(1)
    df['is_sc_vsc'] = (df['track_status_raw'] > 1).astype(int)

    # -- 16. circuit_encoded (dynamic) -------------------------------------
    df['circuit_encoded'] = df['circuit'].map(circuit_map).fillna(-1).astype(int)

    # -- 17. compound-tyre interaction features ----------------------------
    df['compound_x_tyrelife'] = df['compound_encoded'] * df['tyre_life']
    df['compound_x_lapsrem'] = df['compound_encoded'] * df['laps_remaining']

    # -- 18. position change from last lap ---------------------------------
    df['position_change'] = (
        df.groupby(grp_key)['relative_position']
        .transform(lambda x: x.diff())
    ).fillna(0)

    # -- 19. proportion of field that has pitted on this lap ---------------
    pit_count_per_lap = (
        df.groupby(['year', 'circuit', 'lap_number'])['is_pit_lap']
        .transform('sum')
    )
    df['field_pit_fraction'] = (pit_count_per_lap - df['is_pit_lap']) / (n_drivers_per_lap - 1).clip(lower=1)

    # -- 20. prev_is_pit_lap -----------------------------------------------
    df['prev_is_pit_lap'] = (
        df.groupby(grp_key)['is_pit_lap']
        .transform(lambda x: x.shift(1))
    ).fillna(0).astype(int)

    # -- 21. is_wet_condition -----------------------------------------------
    df['is_wet_condition'] = df['compound'].isin(['INTERMEDIATE', 'WET']).astype(int)

    # -- 22. deg_acceleration -----------------------------------------------
    df['deg_acceleration'] = (
        df.groupby(grp_key)['deg_trend_3']
        .transform(lambda x: x.diff())
    ).fillna(0)

    # Clean up helper columns
    df.drop(columns=['total_laps_race', 'track_status_raw'], inplace=True)

    # =====================================================================
    # LABELS
    # =====================================================================

    # -- pit_in_next_3_laps ------------------------------------------------
    df['pit_in_next_3_laps'] = 0
    for (year, circuit, driver), grp in df.groupby(grp_key):
        idx = grp.index
        pit_flags = grp['is_pit_lap'].values
        label = np.zeros(len(pit_flags), dtype=int)
        for offset in range(1, 4):
            shifted = np.roll(pit_flags, -offset)
            shifted[-offset:] = 0
            label = np.maximum(label, shifted)
        df.loc[idx, 'pit_in_next_3_laps'] = label

    # -- Save --------------------------------------------------------------
    df.to_csv('data/features.csv', index=False)
    print(f"\nSaved {len(df)} rows with {len(df.columns)} columns to data/features.csv")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nLabel distribution (pit_in_next_3_laps):")
    print(df['pit_in_next_3_laps'].value_counts())
    print(f"\nCompound distribution:")
    print(df['compound'].value_counts())
    print(f"\nRows per year:")
    print(df.groupby('year').size())


if __name__ == '__main__':
    engineer_features()
