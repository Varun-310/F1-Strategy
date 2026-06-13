import pandas as pd
import numpy as np

# Official total laps per circuit-year (sourced from FastF1 race data).
# Cancelled or shortened races are included with their actual lap count.
TOTAL_LAPS = {
    ('Canada', 2019): 70,
    ('Canada', 2022): 70,
    ('Canada', 2023): 70,
    ('Canada', 2024): 70,
    ('Silverstone', 2019): 52,
    ('Silverstone', 2020): 52,
    ('Silverstone', 2021): 52,
    ('Silverstone', 2022): 52,
    ('Silverstone', 2023): 52,
    ('Silverstone', 2024): 52,
    ('Spa', 2019): 44,
    ('Spa', 2020): 44,
    ('Spa', 2021): 3,   # rain-shortened 2021 Belgian GP
    ('Spa', 2022): 44,
    ('Spa', 2023): 44,
    ('Spa', 2024): 44,
    ('Monza', 2019): 53,
    ('Monza', 2020): 53,
    ('Monza', 2021): 53,
    ('Monza', 2022): 53,
    ('Monza', 2023): 53,
    ('Monza', 2024): 53,
    ('Japan', 2019): 53,
    ('Japan', 2022): 53,
    ('Japan', 2023): 53,
    ('Japan', 2024): 53,
}

DEFAULT_TOTAL_LAPS = 70

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

# Circuit encoding
CIRCUIT_MAP = {
    'Canada': 0,
    'Silverstone': 1,
    'Spa': 2,
    'Monza': 3,
    'Japan': 4,
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

    # -- Ensure correct types ----------------------------------------------
    df['lap_number'] = df['lap_number'].astype(int)
    df['is_pit_lap'] = df['is_pit_lap'].astype(int)

    # -- Sort consistently for rolling / shift operations ------------------
    df = df.sort_values(['year', 'circuit', 'driver', 'lap_number']).reset_index(drop=True)

    # Define group key for per-driver-per-race operations
    grp_key = ['year', 'circuit', 'driver']

    # =====================================================================
    # ORIGINAL FEATURES
    # =====================================================================

    # -- 1. lap_time_delta -------------------------------------------------
    #    Driver lap time minus their own median for that race
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
    df['position_approx'] = (
        df.groupby(['year', 'circuit', 'lap_number'])['cum_time']
        .rank(method='min')
    )
    n_drivers_per_lap = df.groupby(['year', 'circuit', 'lap_number'])['driver'].transform('count')
    df['relative_position'] = df['position_approx'] / n_drivers_per_lap
    df.drop(columns=['cum_time', 'position_approx'], inplace=True)

    # -- 4. laps_remaining -------------------------------------------------
    df['total_laps_race'] = df.apply(
        lambda row: TOTAL_LAPS.get((row['circuit'], row['year']), DEFAULT_TOTAL_LAPS),
        axis=1
    )
    df['laps_remaining'] = df['total_laps_race'] - df['lap_number']

    # -- 5. compound_encoded -----------------------------------------------
    df['compound_encoded'] = df['compound'].map(COMPOUND_MAP).fillna(-1).astype(int)

    # =====================================================================
    # NEW FEATURES (v2)
    # =====================================================================

    # -- 6. stint_number ---------------------------------------------------
    #    Count how many stints the driver has done so far (cumulative pit stops + 1)
    df['stint_number'] = df.groupby(grp_key)['is_pit_lap'].cumsum() + 1
    # Shift so that the lap AFTER the pit is the new stint
    # Actually the cumsum already does this correctly:
    # pit_lap gets counted in its own stint, next lap starts a new stint

    # -- 7. pct_race_complete ----------------------------------------------
    df['pct_race_complete'] = df['lap_number'] / df['total_laps_race']

    # -- 8. fuel_corrected_delta -------------------------------------------
    #    Approximate fuel correction: lap times naturally drop ~0.06s/lap
    df['fuel_corrected_delta'] = df['lap_time_delta'] + (FUEL_EFFECT_PER_LAP * df['lap_number'])

    # -- 9. tyre_life_squared (captures non-linear degradation) ------------
    df['tyre_life_squared'] = df['tyre_life'] ** 2

    # -- 10. degradation trend: rolling mean of lap_time_delta last 3 laps --
    df['deg_trend_3'] = (
        df.groupby(grp_key)['lap_time_delta']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # -- 11. degradation trend: rolling mean of lap_time_delta last 5 laps --
    df['deg_trend_5'] = (
        df.groupby(grp_key)['lap_time_delta']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # -- 12. lap time consistency: rolling std over last 3 laps ------------
    df['lap_time_std_3'] = (
        df.groupby(grp_key)['lap_time_seconds']
        .transform(lambda x: x.rolling(3, min_periods=2).std())
    )

    # -- 13. delta to race leader lap time (per lap) -----------------------
    min_lap_time = (
        df.groupby(['year', 'circuit', 'lap_number'])['lap_time_seconds']
        .transform('min')
    )
    df['gap_to_leader_lap'] = df['lap_time_seconds'] - min_lap_time

    # -- 14. laps since last pit -------------------------------------------
    #    Number of laps since the driver's most recent pit stop
    df['laps_since_pit'] = 0
    for _, grp in df.groupby(grp_key):
        idx = grp.index
        pit_flags = grp['is_pit_lap'].values
        laps_since = np.zeros(len(pit_flags), dtype=int)
        counter = 999  # large number for "never pitted yet"
        for i in range(len(pit_flags)):
            if i > 0 and pit_flags[i - 1] == 1:
                counter = 0
            counter += 1
            laps_since[i] = counter
        df.loc[idx, 'laps_since_pit'] = laps_since

    # -- 15. track_status cleaned ------------------------------------------
    #    1 = normal, 2+ = SC/VSC/red flag, encode as binary flags
    df['track_status_raw'] = pd.to_numeric(df['track_status'], errors='coerce').fillna(1)
    df['is_sc_vsc'] = (df['track_status_raw'] > 1).astype(int)

    # -- 16. circuit_encoded -----------------------------------------------
    df['circuit_encoded'] = df['circuit'].map(CIRCUIT_MAP).fillna(-1).astype(int)

    # -- 17. compound-tyre interaction features ----------------------------
    df['compound_x_tyrelife'] = df['compound_encoded'] * df['tyre_life']
    df['compound_x_lapsrem'] = df['compound_encoded'] * df['laps_remaining']

    # -- 18. position change from last lap ---------------------------------
    df['position_change'] = (
        df.groupby(grp_key)['relative_position']
        .transform(lambda x: x.diff())
    ).fillna(0)

    # -- 19. proportion of field that has pitted on this lap ---------------
    #    How many OTHER drivers pitted on the same lap
    pit_count_per_lap = (
        df.groupby(['year', 'circuit', 'lap_number'])['is_pit_lap']
        .transform('sum')
    )
    df['field_pit_fraction'] = (pit_count_per_lap - df['is_pit_lap']) / (n_drivers_per_lap - 1).clip(lower=1)

    # -- 20. prev_is_pit_lap -----------------------------------------------
    #    Lagged version of is_pit_lap: was the PREVIOUS lap a pit lap?
    #    Unlike is_pit_lap, this is known at prediction time (start of current lap)
    df['prev_is_pit_lap'] = (
        df.groupby(grp_key)['is_pit_lap']
        .transform(lambda x: x.shift(1))
    ).fillna(0).astype(int)

    # -- 21. is_wet_condition -----------------------------------------------
    #    Binary flag: is the driver currently on wet-weather tyres?
    df['is_wet_condition'] = df['compound'].isin(['INTERMEDIATE', 'WET']).astype(int)

    # -- 22. deg_acceleration -----------------------------------------------
    #    Change in degradation trend = 2nd derivative of lap time
    #    Captures whether degradation is accelerating (cliff approaching)
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
    print(f"\nSample rows (new features):")
    print(df[['driver', 'lap_number', 'stint_number', 'deg_trend_3',
              'fuel_corrected_delta', 'tyre_life_squared', 'laps_since_pit',
              'is_sc_vsc', 'field_pit_fraction', 'pit_in_next_3_laps']].head(10).to_string())


if __name__ == '__main__':
    engineer_features()
