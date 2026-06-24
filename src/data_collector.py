import os
import pandas as pd
import fastf1


def collect_all_data():
    """Collect lap data for ALL races from 2019 to 2024 using FastF1 event schedules."""
    os.makedirs('cache', exist_ok=True)
    fastf1.Cache.enable_cache('cache')

    output_file = 'data/historical_laps.csv'
    os.makedirs('data', exist_ok=True)

    # Track completed sessions for resume support
    completed_sessions = set()
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            existing_df = pd.read_csv(output_file)
            if not existing_df.empty and 'year' in existing_df.columns and 'circuit' in existing_df.columns:
                unique_pairs = existing_df[['year', 'circuit']].drop_duplicates()
                for _, r in unique_pairs.iterrows():
                    completed_sessions.add((int(r['year']), str(r['circuit'])))
                print(f"Resuming. Found {len(completed_sessions)} sessions already collected.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Starting fresh.")

    years = range(2019, 2027)
    total_loaded = 0
    total_skipped = 0
    total_failed = 0

    for year in years:
        print(f"\n{'='*60}")
        print(f"  YEAR {year}")
        print(f"{'='*60}")

        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Failed to get schedule for {year}: {e}")
            continue

        # Filter to race events only (exclude testing)
        races = schedule[schedule['EventFormat'].isin([
            'conventional', 'sprint', 'sprint_shootout',
            'sprint_qualifying', 'testing'
        ]) == False]
        # Actually just get all non-testing events
        races = schedule[schedule['EventFormat'] != 'testing']

        # If the above filtering is too aggressive, just use all events
        # and let fastf1 handle it
        if len(races) == 0:
            races = schedule

        print(f"Found {len(races)} events in {year}")

        for _, event in races.iterrows():
            event_name = event.get('EventName', 'Unknown')
            location = event.get('Location', 'Unknown')
            round_num = event.get('RoundNumber', 0)

            # Skip pre-season testing
            if round_num == 0:
                continue

            # Use location as circuit identifier for consistency
            circuit_name = location

            if (year, circuit_name) in completed_sessions:
                print(f"  [{year} R{round_num}] {event_name} ({location}) - already collected, skipping")
                total_skipped += 1
                continue

            print(f"\n  [{year} R{round_num}] {event_name} ({location})...")

            try:
                session = fastf1.get_session(year, round_num, 'R')
                session.load()
                laps = session.laps

                if laps is None or len(laps) == 0:
                    print(f"    No laps found - race may have been cancelled")
                    # Save placeholder
                    dummy = pd.DataFrame([{
                        'year': year, 'circuit': circuit_name,
                        'driver': 'NONE', 'lap_number': None,
                        'lap_time_seconds': None, 'compound': None,
                        'tyre_life': None, 'is_pit_lap': None,
                        'track_status': None
                    }])
                    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
                    dummy.to_csv(output_file, mode='a', header=write_header, index=False)
                    completed_sessions.add((year, circuit_name))
                    continue

                # Extract lap data
                session_laps = []
                for _, row in laps.iterrows():
                    lap_time = row['LapTime']
                    lap_time_seconds = lap_time.total_seconds() if pd.notna(lap_time) else None
                    is_pit_lap = 1 if pd.notna(row['PitInTime']) else 0

                    session_laps.append({
                        'year': year,
                        'circuit': circuit_name,
                        'driver': row['Driver'],
                        'lap_number': row['LapNumber'],
                        'lap_time_seconds': lap_time_seconds,
                        'compound': row['Compound'],
                        'tyre_life': row['TyreLife'],
                        'is_pit_lap': is_pit_lap,
                        'track_status': row['TrackStatus']
                    })

                if session_laps:
                    session_df = pd.DataFrame(session_laps)
                    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
                    session_df.to_csv(output_file, mode='a', header=write_header, index=False)
                    print(f"    Saved {len(session_laps)} laps")
                    completed_sessions.add((year, circuit_name))
                    total_loaded += 1

            except Exception as e:
                print(f"    FAILED: {e}")
                total_failed += 1

    print(f"\n{'='*60}")
    print(f"  DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Loaded:  {total_loaded} races")
    print(f"  Skipped: {total_skipped} (already collected)")
    print(f"  Failed:  {total_failed}")

    # Print final summary
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        df_clean = df[df['driver'] != 'NONE']
        print(f"\n  Total rows in CSV: {len(df)}")
        print(f"  Valid lap rows:    {len(df_clean)}")
        print(f"  Unique circuits:   {df_clean['circuit'].nunique()}")
        print(f"  Unique races:      {len(df_clean.groupby(['year', 'circuit']))}")


if __name__ == '__main__':
    collect_all_data()
