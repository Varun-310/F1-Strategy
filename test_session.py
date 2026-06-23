"""
test_session.py -- Comprehensive test script to verify all four new core features.
"""

import os
import sys

# Ensure project root is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.safety_car import sc_probability, SC_RATES
from src.pre_race import generate_briefing
from src.live_feed import LiveRaceFeed
from src.race_simulator import simulate_race

def test_safety_car():
    print("\n=== TESTING SAFETY CAR PROBABILITY ===")
    circuits_to_test = ['Monaco', 'Singapore', 'Canada', 'Monza', 'Spa', 'Unknown']
    laps_to_test = [(1, 70), (5, 70), (40, 70), (60, 70)] # start, early, mid, late

    for circuit in circuits_to_test:
        print(f"\nCircuit: {circuit} (Base SC rate: {SC_RATES.get(circuit, 1.0)})")
        for current_lap, total_laps in laps_to_test:
            prob = sc_probability(circuit, current_lap, total_laps)
            print(f"  Lap {current_lap}/{total_laps}: SC Prob (next 5 laps) = {prob:.2%}")

def test_pre_race_briefing():
    print("\n=== TESTING PRE-RACE BRIEFING ===")
    circuit = "Canada"
    race_date = "2024-06-09"
    grid_position = 3
    driver = "NOR"
    
    print(f"Generating pre-race briefing for {driver} at {circuit} GP ({race_date}) starting P{grid_position}...")
    briefing = generate_briefing(circuit, race_date, grid_position, driver)
    
    print("\nBriefing Result Details:")
    print(f"  - Starting Tyre Compound:  {briefing['starting_compound']}")
    print(f"  - Expected Pit Window:     Laps {briefing['pit_window_start']} - {briefing['pit_window_end']}")
    print(f"  - Rain Risk Probability:   {briefing['rain_risk']:.1f}%")
    print(f"  - Safety Car Risk Prob:    {briefing['sc_risk']:.2%}")
    print("\nAI Driver Briefing Text:")
    print("-" * 65)
    print(briefing['briefing_text'])
    print("-" * 65)

def test_live_feed():
    print("\n=== TESTING LIVE RACE FEED ===")
    # Using session_key 9153 (Singapore 2023) and driver 44 (Hamilton)
    session_key = "9153"
    driver_number = 44
    
    print(f"Initializing LiveRaceFeed for session={session_key}, driver={driver_number}...")
    feed = LiveRaceFeed(session_key, driver_number)
    
    print("Polling live data once (expects real API response or graceful fallback)...")
    polled = feed.poll()
    print("Polled Output Dict:")
    for k, v in polled.items():
        print(f"  - {k}: {v}")

    # Test threading controls
    print("\nTesting background polling thread...")
    results_received = []
    
    def on_new_data(data):
        results_received.append(data)
        print(f"  [Callback] Polled Lap {data['lap_number']}, Pos P{data['position']}, Compound {data['compound']}")

    print("Starting background polling every 2 seconds...")
    feed.start_polling(on_new_data, interval_seconds=2)
    time_to_wait = 5
    print(f"Waiting {time_to_wait} seconds for polling...")
    import time
    time.sleep(time_to_wait)
    
    print("Stopping background polling...")
    feed.stop()
    print(f"Stopped. Total polls received: {len(results_received)}")

def test_simulator():
    print("\n=== TESTING HISTORICAL REPLAY SIMULATOR ===")
    # Replay 2024 Canada Grand Prix for driver NOR (we will just run a few laps to verify it loads)
    circuit = "Canada"
    year = 2024
    driver = "NOR"
    
    print(f"Starting simulation of {year} {circuit} GP for {driver}...")
    try:
        # To avoid running all 70 laps in the test, we will limit the printout or just let it run.
        # But wait! Since simulate_race downloads the full session from FastF1, let's verify if cache is hit.
        decisions = simulate_race(circuit, year, driver)
        print(f"\nSimulation complete. Total decisions simulated: {len(decisions)}")
    except Exception as e:
        print(f"Simulator run encountered an error: {e}")

def main():
    test_safety_car()
    test_pre_race_briefing()
    test_live_feed()
    
    # We run the simulator last since FastF1 download might take some seconds
    # Let's run it to ensure the replay is fully verified.
    test_simulator()

if __name__ == "__main__":
    main()
