"""
test_engine.py -- Integration test for RaceState, StrategyEngine, and LLMExplainer.
"""

import os
import sys

# Ensure project root is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.race_state import RaceState
from src.strategy_engine import StrategyEngine
from src.llm_explainer import explain_decision

def main():
    print("Initializing Integration Test...")

    # Define model paths
    pit_model_path = "models/pit_model.json"
    compound_model_path = "models/compound_model.json"

    # 1. Initialize the Strategy Engine
    try:
        engine = StrategyEngine(
            pit_model_path=pit_model_path,
            compound_model_path=compound_model_path
        )
        print("  - StrategyEngine loaded successfully.")
    except Exception as e:
        print(f"Error loading StrategyEngine: {e}")
        sys.exit(1)

    # 2. Create a RaceState for driver NOR at Canada Grand Prix (total 70 laps)
    driver = "NOR"
    circuit = "Canada"
    total_laps = 70
    lat = 45.5005
    lon = -73.5225

    state = RaceState(
        driver=driver,
        circuit=circuit,
        total_laps=total_laps,
        circuit_lat=lat,
        circuit_lon=lon,
        current_compound="MEDIUM",
        tyre_age=0,
        current_position=5,
        gap_to_leader=8.5,
        last_lap_time=76.2
    )
    print(f"  - RaceState created for {driver} at {circuit} (mapped to circuit_encoded={state._circuit_encoded}).")

    # 3. Simulate 5 laps of updates with dummy lap data
    print("\nSimulating 5 laps of updates:")
    dummy_laps = [
        # Lap 1
        {"lap_time": 76.5, "position": 5, "gap_to_leader": 8.8, "track_status": 1, "field_pit_fraction": 0.0},
        # Lap 2
        {"lap_time": 76.8, "position": 4, "gap_to_leader": 8.2, "track_status": 1, "field_pit_fraction": 0.0},
        # Lap 3
        {"lap_time": 77.2, "position": 4, "gap_to_leader": 8.9, "track_status": 1, "field_pit_fraction": 0.05},
        # Lap 4
        {"lap_time": 77.5, "position": 4, "gap_to_leader": 9.4, "track_status": 1, "field_pit_fraction": 0.10},
        # Lap 5
        {"lap_time": 77.9, "position": 4, "gap_to_leader": 10.1, "track_status": 1, "field_pit_fraction": 0.0},
    ]

    weather_data = {"rain_probability": 15.0, "temperature": 21.5}

    for i, lap_data in enumerate(dummy_laps, start=1):
        state.update(lap_data=lap_data, weather=weather_data)
        print(f"  Lap {state.current_lap} updated: tyre_age={state.tyre_age}, position={state.current_position}, last_time={state.last_lap_time:.2f}s")

    # 4. Run Strategy Recommendation
    print("\nRunning strategy engine recommendation...")
    rain_risk = weather_data["rain_probability"]
    sc_prob = 0.05

    decision = engine.recommend(
        state=state,
        rain_risk=rain_risk,
        sc_prob=sc_prob
    )

    print("\n--- Strategy Engine Output ---")
    print(f"Lap Number: {decision['lap']}")
    print(f"Action:     {decision['action']}")
    print(f"Compound:   {decision['compound']}")
    print(f"Confidence: {decision['confidence']:.4%}")

    # 5. Generate LLM explanation of the decision
    print("\nGenerating LLM explanation...")
    explanation = explain_decision(
        state=state,
        decision=decision,
        rain_risk=rain_risk
    )

    print("\n--- Explanation briefing to driver ---")
    print(explanation)
    print("--------------------------------------")

    # 6. Test Rain Override Scenario
    print("\n======================================")
    print("TESTING RAIN OVERRIDE SCENARIO")
    print("======================================")
    
    # Update state with high rain probability
    rainy_weather = {"rain_probability": 45.0, "temperature": 18.0}
    rainy_lap = {"lap_time": 79.5, "position": 4, "gap_to_leader": 12.0, "track_status": 1, "field_pit_fraction": 0.20}
    
    state.update(lap_data=rainy_lap, weather=rainy_weather)
    print(f"  Lap {state.current_lap} updated: tyre_age={state.tyre_age}, position={state.current_position}, rain={rainy_weather['rain_probability']}%")
    
    rain_risk_rainy = rainy_weather["rain_probability"]
    decision_rainy = engine.recommend(
        state=state,
        rain_risk=rain_risk_rainy,
        sc_prob=0.10
    )
    
    print("\n--- Strategy Engine Output (Rainy) ---")
    print(f"Lap Number: {decision_rainy['lap']}")
    print(f"Action:     {decision_rainy['action']}")
    print(f"Compound:   {decision_rainy['compound']}")
    print(f"Confidence: {decision_rainy['confidence']:.4%}")
    print(f"Override:   {decision_rainy['_rain_override']}")
    
    print("\nGenerating LLM explanation (Rainy)...")
    explanation_rainy = explain_decision(
        state=state,
        decision=decision_rainy,
        rain_risk=rain_risk_rainy
    )
    
    print("\n--- Explanation briefing to driver (Rainy) ---")
    print(explanation_rainy)
    print("--------------------------------------")

if __name__ == "__main__":
    main()
