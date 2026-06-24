"""
pre_race.py -- Generates a pre-race briefing including weather forecasts, tyres, pit windows, and safety car probability.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timezone
from src.race_state import RaceState
from src.llm_explainer import explain_decision
from src.weather import CIRCUITS, rain_risk_score
from src.safety_car import sc_probability

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def generate_briefing(circuit: str, race_date: str, grid_position: int, driver: str, api_key: Optional[str] = None) -> dict:
    """Prepare a pre-race briefing report for a driver before lights out.

    Args:
        circuit: Name of the circuit (e.g. 'Canada').
        race_date: The date of the race in YYYY-MM-DD format.
        grid_position: Starting grid slot (1 to 20).
        driver: Three-letter driver abbreviation (e.g. 'NOR').
        api_key: Optional Gemini API key.

    Returns:
        A dict with briefing details:
            - starting_compound: Recommended tyre compound (e.g. 'MEDIUM')
            - pit_window_start: Projected start lap of the pit window
            - pit_window_end: Projected end lap of the pit window
            - rain_risk: Weather rain probability score (0 to 100)
            - sc_risk: Estimated Safety Car probability
            - briefing_text: Custom AI explanation/briefing text
    """
    # 1. Resolve coordinates for circuit
    coords = CIRCUITS.get(circuit, None)
    if coords is None:
        # Check case-insensitive substring match
        circuit_lower = circuit.lower()
        for name, c_coords in CIRCUITS.items():
            if name.lower() in circuit_lower or circuit_lower in name.lower():
                coords = c_coords
                break
        if coords is None:
            # Default to Canada coordinates
            coords = {'lat': 45.5006, 'lon': -73.5225}

    lat, lon = coords['lat'], coords['lon']

    # 2. Get rain risk score for race start hour (defaulting to 14:00 UTC start)
    race_start_iso = f"{race_date}T14:00:00Z"
    try:
        rain_risk = float(rain_risk_score(lat, lon, race_start_iso))
    except Exception as e:
        print(f"Error computing rain risk: {e}")
        rain_risk = 20.0

    # 3. Load historical data and estimate tyre and pit parameters
    historical_file = 'data/historical_laps.csv'
    starting_compound = 'MEDIUM'
    pit_window_start = 20
    pit_window_end = 26
    median_pit_lap = 23

    if os.path.exists(historical_file):
        try:
            df = pd.read_csv(historical_file)
            # Filter by circuit name (ignoring casing)
            circuit_df = df[df['circuit'].str.lower().str.contains(circuit.lower(), na=False)].copy()
            
            # Find most common starting compound (compound on lap 1)
            lap_1_df = circuit_df[circuit_df['lap_number'] == 1]
            if not lap_1_df.empty:
                mode_comp = lap_1_df['compound'].mode()
                if not mode_comp.empty:
                    starting_compound = str(mode_comp.iloc[0])
            
            # Find median pit stop lap
            pit_stops = circuit_df[circuit_df['is_pit_lap'] == 1]
            if not pit_stops.empty:
                median_pit_lap = int(pit_stops['lap_number'].median())
            
            pit_window_start = max(1, median_pit_lap - 3)
            pit_window_end = median_pit_lap + 3
        except Exception as e:
            print(f"Error analyzing historical laps: {e}")

    # 4. Lookup Safety Car risk (pre-race start probability on lap 1 of a 57-lap race)
    sc_risk = float(sc_probability(circuit, current_lap=1, total_laps=57))

    # 5. Generate LLM Pre-race Briefing text using a fake pre-race state
    try:
        race_year = int(race_date.split('-')[0])
    except Exception:
        race_year = 2026

    fake_state = RaceState(
        driver=driver,
        circuit=circuit,
        total_laps=57,
        circuit_lat=lat,
        circuit_lon=lon,
        year=race_year,
        current_lap=0,  # Lap 0 designates pre-race status
        current_compound=starting_compound,
        tyre_age=0,
        current_position=grid_position,
        gap_to_leader=0.0,
        last_lap_time=0.0
    )

    fake_decision = {
        'action': 'STAY_OUT',
        'compound': starting_compound,
        'confidence': 1.0,
        'rain_risk': rain_risk,
        'lap': 0
    }

    # Fetch explanation briefing
    briefing_text = explain_decision(fake_state, fake_decision, rain_risk, api_key=api_key)

    # Intercept fallback message if LLM generated generic in-race fallback text
    if "Stay out" in briefing_text and fake_state.current_lap == 0:
        briefing_text = (
            f"Good morning, {driver}. We are starting from P{grid_position} today on the {starting_compound} compound. "
            f"Expected rain risk is stable at {rain_risk:.0f}%, and our planned pit window is open around lap {pit_window_start} to {pit_window_end}. "
            f"Focus on a clean start, keep temperature in the tyres, and we will monitor safety car opportunities early on."
        )

    return {
        'starting_compound': starting_compound,
        'pit_window_start': pit_window_start,
        'pit_window_end': pit_window_end,
        'rain_risk': rain_risk,
        'sc_risk': sc_risk,
        'briefing_text': briefing_text
    }
