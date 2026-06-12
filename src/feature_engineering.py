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
    ('Japan', 2022): 53,  # rain-shortened in reality but data shows 29 max laps
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


def engineer_features():
    """Load historical laps, compute derived features and the pit label, save to CSV."""
    df = pd.read_csv('data/historical_laps.csv')
    print(f"Loaded {len(df)} rows from data/historical_laps.csv")

    # ── Remove placeholder rows from cancelled races ──────────────────────
    df = df[df['driver'] != 'NONE'].copy()

    # ── Drop rows with null lap times ─────────────────────────────────────
    df = df.dropna(subset=['lap_time_seconds']).copy()
    print(f"After dropping null lap times: {len(df)} rows")

    # ── Ensure correct types ──────────────────────────────────────────────
    df['lap_number'] = df['lap_number'].astype(int)
    df['is_pit_lap'] = df['is_pit_lap'].astype(int)

    # ── Sort consistently for rolling / shift operations ──────────────────
    df = df.sort_values(['year', 'circuit', 'driver', 'lap_number']).reset_index(drop=True)

    # ── 1. lap_time_delta ─────────────────────────────────────────────────
    #    Driver lap time minus their own median for that race
    race_driver_median = (
        df.groupby(['year', 'circuit', 'driver'])['lap_time_seconds']
        .transform('median')
    )
    df['lap_time_delta'] = df['lap_time_seconds'] - race_driver_median

    # ── 2. tyre_age_normalised ────────────────────────────────────────────
    #    tyre_life / avg stint length for the compound, capped at 1.5
    df['tyre_age_normalised'] = df.apply(
        lambda row: min(
            row['tyre_life'] / AVG_STINT_LENGTH.get(row['compound'], 28),
            TYRE_AGE_CAP
        ) if pd.notna(row['tyre_life']) and pd.notna(row['compound']) else np.nan,
        axis=1
    )

    # ── 3. relative_position ──────────────────────────────────────────────
    #    We don't have a 'position' column in our dataset, so we derive it
    #    from lap times: rank drivers by cumulative time each lap.
    #    Approximate position as rank of cumulative lap time within each
    #    (year, circuit, lap_number) group.
    df['cum_time'] = df.groupby(['year', 'circuit', 'driver'])['lap_time_seconds'].cumsum()
    df['position_approx'] = (
        df.groupby(['year', 'circuit', 'lap_number'])['cum_time']
        .rank(method='min')
    )
    df['relative_position'] = df['position_approx'] / 20.0
    df.drop(columns=['cum_time', 'position_approx'], inplace=True)

    # ── 4. laps_remaining ─────────────────────────────────────────────────
    df['total_laps'] = df.apply(
        lambda row: TOTAL_LAPS.get((row['circuit'], row['year']), DEFAULT_TOTAL_LAPS),
        axis=1
    )
    df['laps_remaining'] = df['total_laps'] - df['lap_number']
    df.drop(columns=['total_laps'], inplace=True)

    # ── 5. compound_encoded ───────────────────────────────────────────────
    df['compound_encoded'] = df['compound'].map(COMPOUND_MAP)
    # Map unknowns to -1
    df['compound_encoded'] = df['compound_encoded'].fillna(-1).astype(int)

    # ── 6. pit_in_next_3_laps (label) ─────────────────────────────────────
    #    For each driver in each race, check if a pit stop happens on any
    #    of the next 3 laps (including current + 1, +2, +3).
    df['pit_in_next_3_laps'] = 0
    for (year, circuit, driver), grp in df.groupby(['year', 'circuit', 'driver']):
        idx = grp.index
        pit_flags = grp['is_pit_lap'].values
        label = np.zeros(len(pit_flags), dtype=int)
        for offset in range(1, 4):
            shifted = np.roll(pit_flags, -offset)
            # Zero out the wrapped-around values
            shifted[-offset:] = 0
            label = np.maximum(label, shifted)
        df.loc[idx, 'pit_in_next_3_laps'] = label

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv('data/features.csv', index=False)
    print(f"Saved {len(df)} rows with {len(df.columns)} columns to data/features.csv")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nLabel distribution (pit_in_next_3_laps):")
    print(df['pit_in_next_3_laps'].value_counts())
    print(f"\nSample rows:")
    print(df[['year', 'circuit', 'driver', 'lap_number', 'lap_time_delta',
              'tyre_age_normalised', 'relative_position', 'laps_remaining',
              'compound_encoded', 'pit_in_next_3_laps']].head(10).to_string())


if __name__ == '__main__':
    engineer_features()
