"""
race_simulator.py -- Replays a historical race lap-by-lap using FastF1 data.
"""

import os
import sys
import pandas as pd
import fastf1

# Ensure project root is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.race_state import RaceState
from src.strategy_engine import StrategyEngine
from src.llm_explainer import explain_decision
from src.weather import CIRCUITS

def simulate_race(circuit: str, year: int, driver: str) -> list:
    """Load a historical race from FastF1 and replay it lap-by-lap through the strategy engine.

    Args:
        circuit: Name of the circuit (e.g. 'Canada' or 'Montreal').
        year: Year of the Grand Prix (e.g. 2024).
        driver: Three-letter driver identifier (e.g. 'NOR').

    Returns:
        List of recommendation dictionaries.
    """
    print(f"Loading session for {year} {circuit} GP...")
    
    # 1. Enable FastF1 caching for speed
    os.makedirs('cache', exist_ok=True)
    fastf1.Cache.enable_cache('cache')

    # 2. Get and load the session
    try:
        session = fastf1.get_session(year, circuit, 'R')
        session.load()
    except Exception as e:
        print(f"Failed to load FastF1 session: {e}")
        return []

    # 3. Pick driver laps
    driver_laps = session.laps.pick_driver(driver).sort_values('LapNumber').copy()
    if driver_laps.empty:
        print(f"No laps found for driver {driver} in the {year} {circuit} GP.")
        return []

    # 4. Pre-compute leader's session times to calculate gap_to_leader exactly
    leader_times = {}
    for _, row in session.laps.iterrows():
        if row['Position'] == 1 and pd.notna(row['Time']):
            leader_times[int(row['LapNumber'])] = row['Time'].total_seconds()

    # 5. Resolve circuit coordinates
    coords = CIRCUITS.get(circuit, {'lat': 45.5006, 'lon': -73.5225}) # Fallback to Canada
    lat, lon = coords['lat'], coords['lon']

    # 6. Initialize StrategyEngine
    pit_model = "models/pit_model.json"
    compound_model = "models/compound_model.json"
    engine = StrategyEngine(pit_model, compound_model)

    # 7. Initialize RaceState
    total_laps = int(session.laps['LapNumber'].max()) if not session.laps.empty else 70
    starting_compound = driver_laps.iloc[0]['Compound'] if not driver_laps.empty else 'MEDIUM'
    
    state = RaceState(
        driver=driver,
        circuit=circuit,
        total_laps=total_laps,
        circuit_lat=lat,
        circuit_lon=lon,
        current_compound=starting_compound,
        tyre_age=0,
        current_position=int(driver_laps.iloc[0]['Position']) if pd.notna(driver_laps.iloc[0]['Position']) else 10
    )

    decisions = []
    
    # 8. Loop through each lap and replay
    print(f"\nReplaying {len(driver_laps)} laps for {driver}:")
    
    for idx, row in driver_laps.iterrows():
        lap_num = int(row['LapNumber'])
        lap_time = row['LapTime'].total_seconds() if pd.notna(row['LapTime']) else 0.0
        position = int(row['Position']) if pd.notna(row['Position']) else state.current_position

        # Compute gap_to_leader
        driver_time = row['Time'].total_seconds() if pd.notna(row['Time']) else 0.0
        leader_time = leader_times.get(lap_num, driver_time)
        gap_to_leader = max(0.0, driver_time - leader_time)

        # Parse track status
        track_status_val = 1
        if pd.notna(row['TrackStatus']) and str(row['TrackStatus']).strip():
            try:
                # TrackStatus can be a combined string like '24' - if it contains >1 status, treat as SC/VSC
                track_status_str = str(row['TrackStatus']).strip()
                if any(c in track_status_str for c in ['2', '4', '5', '6', '7']):
                    track_status_val = 2
            except ValueError:
                pass

        # Compute field pit fraction
        other_pitting = len(session.laps[(session.laps['LapNumber'] == lap_num) & 
                                         (session.laps['Driver'] != driver) & 
                                         pd.notna(session.laps['PitInTime'])])
        total_others = len(session.laps[(session.laps['LapNumber'] == lap_num) & 
                                       (session.laps['Driver'] != driver)])
        field_pit_fraction = other_pitting / total_others if total_others > 0 else 0.0

        # Construct updates
        lap_data = {
            'lap_time': lap_time,
            'position': position,
            'gap_to_leader': gap_to_leader,
            'track_status': track_status_val,
            'field_pit_fraction': field_pit_fraction
        }
        weather_data = {
            'rain_probability': 20.0,
            'temperature': 20.0
        }

        # Update current lap metrics in state
        state.update(lap_data, weather_data)

        # Call engine and explainer
        decision = engine.recommend(state, rain_risk=20.0, sc_prob=0.1)
        explanation = explain_decision(state, decision, rain_risk=20.0)

        # Print the requested output format: lap number, action, compound, explanation
        print(f"Lap {decision['lap']} | Action: {decision['action']} | Compound: {decision['compound']} | Briefing: {explanation}")

        # Append to choices
        decision['explanation'] = explanation
        decisions.append(decision)

        # Synchronize physical state if the actual driver pitted on this lap
        actual_pitted = pd.notna(row['PitInTime'])
        if actual_pitted:
            # Look up compound fitted for the next lap
            next_laps = driver_laps[driver_laps['LapNumber'] == lap_num + 1]
            next_compound = next_laps.iloc[0]['Compound'] if not next_laps.empty else row['Compound']
            state.record_pit(next_compound)

    return decisions
