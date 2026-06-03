import os
import pandas as pd
import fastf1

# Valid locations/event names for each circuit to filter out fastf1 auto-correction for cancelled races
CIRCUIT_VALIDATORS = {
    'Canada': {'location': 'montreal', 'keywords': ['canada', 'canadian']},
    'Silverstone': {'location': 'silverstone', 'keywords': ['british', '70th anniversary', 'silverstone']},
    'Spa': {'location': 'spa', 'keywords': ['belgian', 'belgium', 'spa']},
    'Monza': {'location': 'monza', 'keywords': ['italian', 'italy', 'monza']},
    'Japan': {'location': 'suzuka', 'keywords': ['japanese', 'japan']}
}

def collect_data():
    # Setup cache
    os.makedirs('cache', exist_ok=True)
    fastf1.Cache.enable_cache('cache')
    
    output_file = 'data/historical_laps.csv'
    os.makedirs('data', exist_ok=True)
    
    # Check which sessions have been completed already
    completed_sessions = set()
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            existing_df = pd.read_csv(output_file)
            if not existing_df.empty and 'year' in existing_df.columns and 'circuit' in existing_df.columns:
                unique_pairs = existing_df[['year', 'circuit']].drop_duplicates()
                for _, r in unique_pairs.iterrows():
                    completed_sessions.add((int(r['year']), str(r['circuit'])))
                print(f"Resuming data collection. Found {len(completed_sessions)} sessions already collected in {output_file}.")
        except Exception as e:
            print(f"Error reading existing CSV to resume: {e}. Will overwrite.")
    
    years = range(2019, 2025)
    circuits = {
        'Canada': 'Canada',
        'Silverstone': 'Silverstone',
        'Spa': 'Belgium',
        'Monza': 'Monza',
        'Japan': 'Japan'
    }
    
    for year in years:
        for circuit_name, query_name in circuits.items():
            if (year, circuit_name) in completed_sessions:
                print(f"Session {year} - {circuit_name} already collected. Skipping.")
                continue
                
            print(f"\n=========================================")
            print(f"Loading session: {year} - {circuit_name} ({query_name})...")
            try:
                session = fastf1.get_session(year, query_name, 'R')
                
                # Check if the loaded event matches the requested circuit
                event = session.event
                loc = event.get('Location', '').lower()
                name = event.get('EventName', '').lower()
                
                validator = CIRCUIT_VALIDATORS[circuit_name]
                is_valid_location = validator['location'] in loc
                is_valid_keyword = any(kw in name for kw in validator['keywords'])
                
                if not (is_valid_location or is_valid_keyword):
                    print(f"WARNING: Skipping {year} {circuit_name}. Loaded event '{event.get('EventName')}' at '{event.get('Location')}' does not match target circuit.")
                    # Write placeholder so we skip this invalid/cancelled session in future runs
                    dummy_df = pd.DataFrame([{
                        'year': year,
                        'circuit': circuit_name,
                        'driver': 'NONE',
                        'lap_number': None,
                        'lap_time_seconds': None,
                        'compound': None,
                        'tyre_life': None,
                        'is_pit_lap': None,
                        'track_status': None
                    }])
                    dummy_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                    completed_sessions.add((year, circuit_name))
                    continue
                
                session.load()
                laps = session.laps
                
                if laps is None or len(laps) == 0:
                    print(f"No laps loaded for {year} - {circuit_name}")
                    # Write placeholder so we skip in future runs
                    dummy_df = pd.DataFrame([{
                        'year': year,
                        'circuit': circuit_name,
                        'driver': 'NONE',
                        'lap_number': None,
                        'lap_time_seconds': None,
                        'compound': None,
                        'tyre_life': None,
                        'is_pit_lap': None,
                        'track_status': None
                    }])
                    dummy_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                    completed_sessions.add((year, circuit_name))
                    continue
                
                print(f"Loaded {len(laps)} laps for {year} - {circuit_name} (Event: {event.get('EventName')})")
                
                session_laps = []
                for _, row in laps.iterrows():
                    lap_time = row['LapTime']
                    if pd.notna(lap_time):
                        lap_time_seconds = lap_time.total_seconds()
                    else:
                        lap_time_seconds = None
                    
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
                
                # Append session laps to CSV
                if session_laps:
                    session_df = pd.DataFrame(session_laps)
                    session_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                    print(f"Saved {len(session_laps)} laps for {year} - {circuit_name} to {output_file}")
                    completed_sessions.add((year, circuit_name))
                    
            except Exception as e:
                print(f"Failed to load {year} - {circuit_name}: {e}")
                # We do NOT save a placeholder here so we can retry on next run if it was a network failure.
                
    print(f"\nData collection run completed.")

if __name__ == '__main__':
    collect_data()
