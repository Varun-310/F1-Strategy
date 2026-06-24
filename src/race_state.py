"""
race_state.py -- RaceState dataclass that tracks everything happening in a race.

Maintains a running memory of lap times, weather, pit stops, and decisions
for a single driver throughout a race.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Average stint lengths per compound (laps) -- matches feature_engineering.py
AVG_STINT_LENGTH = {
    'SOFT': 18,
    'MEDIUM': 28,
    'HARD': 38,
    'INTERMEDIATE': 20,
    'INTER': 20,
    'WET': 15,
}

COMPOUND_MAP = {
    'SOFT': 0,
    'MEDIUM': 1,
    'HARD': 2,
    'INTERMEDIATE': 3,
    'WET': 4,
}

# Circuit name to encoded ID mapping from training dataset
CIRCUIT_MAP = {
    'Austin': 0, 'Baku': 1, 'Barcelona': 2, 'Budapest': 3, 'Hockenheim': 4,
    'Imola': 5, 'Istanbul': 6, 'Jeddah': 7, 'Las Vegas': 8, 'Le Castellet': 9,
    'Lusail': 10, 'Marina Bay': 11, 'Melbourne': 12, 'Mexico City': 13, 'Miami': 14,
    'Monaco': 15, 'Monte Carlo': 16, 'Montreal': 17, 'Montréal': 17, 'Montral': 17,
    'Canada': 17, 'Monza': 18, 'Mugello': 19, 'Nürburgring': 20, 'Nrburgring': 20,
    'Portimao': 21, 'Portimão': 21, 'Portimo': 21, 'Sakhir': 22, 'Shanghai': 23,
    'Silverstone': 24, 'Singapore': 25, 'Sochi': 26, 'Spa-Francorchamps': 27, 'Spa': 27,
    'Spielberg': 28, 'Austria': 28, 'Red Bull Ring': 28, 'Suzuka': 29, 'São Paulo': 30,
    'Sao Paulo': 30, 'So Paulo': 30, 'Brazil': 30, 'Yas Island': 31, 'Abu Dhabi': 31,
    'Yas Marina': 31, 'Zandvoort': 32, 'Dutch': 32
}

# Approximate fuel-effect per lap (seconds)
FUEL_EFFECT_PER_LAP = 0.06



@dataclass
class RaceState:
    """Tracks the full state of a single driver's race."""

    # -- Identity --
    driver: str
    circuit: str
    total_laps: int
    circuit_lat: float
    circuit_lon: float
    year: int = 2026

    # -- Current state (updated each lap) --
    current_lap: int = 0
    current_compound: str = 'MEDIUM'
    tyre_age: int = 0
    current_position: int = 10
    gap_to_leader: float = 0.0
    last_lap_time: float = 0.0

    # -- History --
    weather_history: List[Dict] = field(default_factory=list)
    decisions_made: List[Dict] = field(default_factory=list)
    pit_stops: List[Dict] = field(default_factory=list)

    # -- Internal tracking for feature engineering --
    _lap_times: List[float] = field(default_factory=list, repr=False)
    _stint_number: int = field(default=1, repr=False)
    _prev_is_pit_lap: int = field(default=0, repr=False)
    _field_pit_fraction: float = field(default=0.0, repr=False)
    _position_history: List[int] = field(default_factory=list, repr=False)
    _is_sc_vsc: int = field(default=0, repr=False)
    _circuit_encoded: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Resolve circuit name to the encoded ID used by the XGBoost models."""
        circuit_lower = self.circuit.lower()
        mapped = 0
        for name, idx in CIRCUIT_MAP.items():
            if name.lower() in circuit_lower or circuit_lower in name.lower():
                mapped = idx
                break
        self._circuit_encoded = mapped

    def update(self, lap_data: Dict, weather: Dict) -> None:
        """Update state with new lap data and weather.

        Args:
            lap_data: dict with keys like 'lap_time', 'position',
                      'gap_to_leader', 'track_status', 'field_pit_fraction'.
            weather:  dict with keys like 'rain_probability', 'temperature'.
        """
        self.current_lap += 1
        self.tyre_age += 1

        # Update from lap_data
        self.last_lap_time = lap_data.get('lap_time', self.last_lap_time)
        self.current_position = lap_data.get('position', self.current_position)
        self.gap_to_leader = lap_data.get('gap_to_leader', self.gap_to_leader)

        # Track status (1=normal, >1=SC/VSC)
        track_status = lap_data.get('track_status', 1)
        self._is_sc_vsc = 1 if track_status > 1 else 0

        # Field pit fraction (proportion of other drivers pitting this lap)
        self._field_pit_fraction = lap_data.get('field_pit_fraction', 0.0)

        # Record lap time for rolling calculations
        if self.last_lap_time > 0:
            self._lap_times.append(self.last_lap_time)

        # Position history for position change calc
        self._position_history.append(self.current_position)

        # Update prev_is_pit_lap (was the PREVIOUS lap a pit lap?)
        # This gets set by record_pit, stays 0 otherwise
        # After recording, reset it for next lap
        if self.current_lap > 1:
            # Check if last lap was a pit
            if self.pit_stops and self.pit_stops[-1].get('lap') == self.current_lap - 1:
                self._prev_is_pit_lap = 1
            else:
                self._prev_is_pit_lap = 0

        # Append weather
        self.weather_history.append({
            'lap': self.current_lap,
            **weather
        })

    def record_pit(self, compound_in: str) -> None:
        """Record a pit stop on the current lap.

        Args:
            compound_in: The new tyre compound being fitted (e.g. 'HARD').
        """
        self.pit_stops.append({
            'lap': self.current_lap,
            'old_compound': self.current_compound,
            'new_compound': compound_in,
            'tyre_age_at_stop': self.tyre_age,
        })
        self.current_compound = compound_in
        self.tyre_age = 0
        self._stint_number += 1

    def add_decision(self, decision: Dict) -> None:
        """Record a strategy decision for history."""
        self.decisions_made.append(decision)

    def laps_since_last_pit(self) -> int:
        """Return laps since the last pit stop, or current_lap if no pits yet."""
        if not self.pit_stops:
            return self.current_lap
        return self.current_lap - self.pit_stops[-1]['lap']

    def get_feature_vector(self, rain_risk: float, sc_prob: float) -> Dict:
        """Build a feature dict matching the 25 features expected by the models.

        This produces the same feature names as train_model.PIT_FEATURES / COMPOUND_FEATURES
        so that the XGBoost models can consume it directly.

        Args:
            rain_risk:  0-100 rain risk score from weather module.
            sc_prob:    0-1 safety car probability estimate.
        """
        compound_enc = COMPOUND_MAP.get(self.current_compound, 1)
        avg_stint = AVG_STINT_LENGTH.get(self.current_compound, 28)

        # Live Calibration:
        # Compare actual pace loss against historical baseline.
        # If pace is dropping off faster than the historical baseline of ~0.06s/lap,
        # we scale the normalized tyre age to make the model perceive the tyre as older.
        deg_factor = 1.0
        if self.tyre_age >= 5 and len(self._lap_times) >= 5:
            recent_times = self._lap_times[-5:]
            slope = (sum(recent_times[-2:]) - sum(recent_times[:2])) / 3.0
            expected_slope = 0.06
            if slope > expected_slope:
                deg_factor = min(slope / expected_slope, 2.0)

        calibrated_tyre_age = int(round(self.tyre_age * deg_factor))
        tyre_age_norm = min(calibrated_tyre_age / avg_stint, 1.5)

        laps_remaining = max(self.total_laps - self.current_lap, 0)
        pct_race = self.current_lap / self.total_laps if self.total_laps > 0 else 0.0
        relative_pos = self.current_position / 20.0

        # Lap time delta: difference from driver's own median
        if len(self._lap_times) >= 3:
            sorted_times = sorted(self._lap_times)
            median_time = sorted_times[len(sorted_times) // 2]
            lap_time_delta = self.last_lap_time - median_time
        elif len(self._lap_times) > 0:
            median_time = sum(self._lap_times) / len(self._lap_times)
            lap_time_delta = self.last_lap_time - median_time
        else:
            lap_time_delta = 0.0

        # Fuel-corrected delta
        fuel_corrected_delta = lap_time_delta + (FUEL_EFFECT_PER_LAP * self.current_lap)

        # Degradation trends (rolling means of delta)
        if len(self._lap_times) >= 3:
            recent_3 = self._lap_times[-3:]
            med = median_time if len(self._lap_times) >= 3 else sum(self._lap_times) / len(self._lap_times)
            deg_trend_3 = sum(t - med for t in recent_3) / len(recent_3)
        else:
            deg_trend_3 = lap_time_delta

        if len(self._lap_times) >= 5:
            recent_5 = self._lap_times[-5:]
            deg_trend_5 = sum(t - median_time for t in recent_5) / len(recent_5)
        else:
            deg_trend_5 = deg_trend_3

        # Lap time consistency (std of last 3 laps)
        if len(self._lap_times) >= 3:
            recent = self._lap_times[-3:]
            mean_r = sum(recent) / len(recent)
            lap_time_std_3 = (sum((t - mean_r) ** 2 for t in recent) / len(recent)) ** 0.5
        else:
            lap_time_std_3 = 0.0

        # Gap to leader lap time (approximated from gap_to_leader per-lap metric)
        gap_to_leader_lap = max(self.gap_to_leader, 0.0)

        # Position change from last lap
        if len(self._position_history) >= 2:
            position_change = (self._position_history[-1] - self._position_history[-2]) / 20.0
        else:
            position_change = 0.0

        # Deg acceleration (change in deg_trend_3)
        # Approximate from recent lap time changes
        deg_acceleration = 0.0
        if len(self._lap_times) >= 4:
            prev_3 = self._lap_times[-4:-1]
            curr_3 = self._lap_times[-3:]
            med = median_time
            prev_deg = sum(t - med for t in prev_3) / 3
            curr_deg = sum(t - med for t in curr_3) / 3
            deg_acceleration = curr_deg - prev_deg

        # Is wet condition
        is_wet = 1 if self.current_compound in ('INTERMEDIATE', 'WET') else 0

        return {
            'lap_time_delta': lap_time_delta,
            'tyre_age_normalised': tyre_age_norm,
            'relative_position': relative_pos,
            'laps_remaining': laps_remaining,
            'compound_encoded': compound_enc,
            'tyre_life': calibrated_tyre_age,
            'lap_number': self.current_lap,
            'is_pit_lap': 0,  # at prediction time, current lap is not a pit lap yet
            'stint_number': self._stint_number,
            'pct_race_complete': pct_race,
            'fuel_corrected_delta': fuel_corrected_delta,
            'tyre_life_squared': calibrated_tyre_age ** 2,
            'deg_trend_3': deg_trend_3,
            'deg_trend_5': deg_trend_5,
            'lap_time_std_3': lap_time_std_3,
            'gap_to_leader_lap': gap_to_leader_lap,
            'laps_since_pit': self.laps_since_last_pit(),
            'is_sc_vsc': self._is_sc_vsc,
            'circuit_encoded': self._circuit_encoded,
            'compound_x_tyrelife': compound_enc * calibrated_tyre_age,
            'compound_x_lapsrem': compound_enc * laps_remaining,
            'position_change': position_change,
            'field_pit_fraction': self._field_pit_fraction,
            'prev_is_pit_lap': self._prev_is_pit_lap,
            'is_wet_condition': is_wet,
            'deg_acceleration': deg_acceleration,
            # Extra context (not consumed by XGBoost but useful for explainer)
            '_rain_risk_score': rain_risk,
            '_sc_probability': sc_prob,
        }
