"""
safety_car.py -- Computes safety car probability based on historical rates and race progress.
"""

SC_RATES = {
    'Monaco': 2.5,
    'Singapore': 2.0,
    'Baku': 1.8,
    'Spa': 1.5,
    'Canada': 1.2,
    'Silverstone': 1.0,
    'Monza': 0.8,
    'Bahrain': 0.7,
    'Japan': 0.6
}

def sc_probability(circuit: str, current_lap: int, total_laps: int) -> float:
    """Estimate the probability of a safety car occurring in the next 5 laps.

    Args:
        circuit: Name of the circuit (e.g. 'Monaco').
        current_lap: The current lap number of the race.
        total_laps: Total laps in the race.

    Returns:
        Probability float between 0.0 and 1.0.
    """
    # 1. Resolve circuit rate (default to 1.0 if not listed)
    base_rate = 1.0
    circuit_lower = circuit.lower()
    for name, rate in SC_RATES.items():
        if name.lower() in circuit_lower or circuit_lower in name.lower():
            base_rate = rate
            break

    # 2. Compute base probability per single lap
    # F1 races are typically around 50 to 70 laps. Avoid division by zero.
    laps_denominator = max(total_laps, 1)
    single_lap_rate = base_rate / laps_denominator

    # 3. Apply stage of race multiplier
    # Multiplier is higher in the first 10 laps (starts/congestion) and last 15 laps (driver fatigue/tension)
    multiplier = 1.0
    if current_lap <= 10:
        multiplier = 1.5
    elif (total_laps - current_lap) <= 15:
        multiplier = 1.3

    adjusted_lap_rate = single_lap_rate * multiplier

    # 4. Compute probability of at least one SC in the next 5 laps: 1 - (1 - rate)^5
    # Cap single lap rate to 1.0 to avoid complex numbers
    adjusted_lap_rate = min(max(adjusted_lap_rate, 0.0), 1.0)
    prob_next_5_laps = 1.0 - (1.0 - adjusted_lap_rate) ** 5

    return round(float(prob_next_5_laps), 4)
