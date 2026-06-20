"""
strategy_engine.py -- Integrates machine learning models and rain rules to make pit decisions.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.race_state import RaceState

COMPOUND_LABELS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']

class StrategyEngine:
    """Combines pit stop and compound classification models with strategic rules."""

    def __init__(self, pit_model_path: str, compound_model_path: str):
        """Load the trained XGBoost models.

        Args:
            pit_model_path: Path to the JSON/UBJ file of the pit stop model.
            compound_model_path: Path to the JSON/UBJ file of the next compound model.
        """
        if not os.path.exists(pit_model_path):
            raise FileNotFoundError(f"Pit model file not found: {pit_model_path}")
        if not os.path.exists(compound_model_path):
            raise FileNotFoundError(f"Compound model file not found: {compound_model_path}")

        self.pit_model = XGBClassifier()
        self.pit_model.load_model(pit_model_path)

        self.compound_model = XGBClassifier()
        self.compound_model.load_model(compound_model_path)

        # Feature lists matching train_model.py exactly
        self.pit_features = [
            'lap_time_delta', 'tyre_age_normalised', 'relative_position', 'laps_remaining',
            'compound_encoded', 'tyre_life', 'lap_number', 'stint_number', 'pct_race_complete',
            'fuel_corrected_delta', 'tyre_life_squared', 'deg_trend_3', 'deg_trend_5',
            'lap_time_std_3', 'gap_to_leader_lap', 'laps_since_pit', 'is_sc_vsc',
            'circuit_encoded', 'compound_x_tyrelife', 'compound_x_lapsrem',
            'position_change', 'field_pit_fraction', 'prev_is_pit_lap',
            'is_wet_condition', 'deg_acceleration'
        ]

        self.compound_features = [
            'lap_time_delta', 'tyre_age_normalised', 'relative_position', 'laps_remaining',
            'compound_encoded', 'tyre_life', 'lap_number', 'is_pit_lap', 'stint_number',
            'pct_race_complete', 'fuel_corrected_delta', 'tyre_life_squared', 'deg_trend_3',
            'deg_trend_5', 'lap_time_std_3', 'gap_to_leader_lap', 'laps_since_pit',
            'is_sc_vsc', 'circuit_encoded', 'compound_x_tyrelife', 'compound_x_lapsrem',
            'position_change', 'field_pit_fraction', 'prev_is_pit_lap',
            'is_wet_condition', 'deg_acceleration'
        ]

        self.compound_labels = COMPOUND_LABELS

    def recommend(self, state: RaceState, rain_risk: float, sc_prob: float) -> dict:
        """Evaluate race state and make a strategic recommendation.

        Args:
            state: The current RaceState memory object.
            rain_risk: The current rain probability (0 to 100).
            sc_prob: The current safety car probability (0.0 to 1.0).

        Returns:
            A dict containing:
                - action: 'PIT' or 'STAY_OUT'
                - compound: Recommended compound (e.g. 'HARD') if pitting, else current compound
                - confidence: Probability float of the decision
                - rain_risk: Input rain risk
                - lap: Current lap number
        """
        # 1. Retrieve the engineered feature vector
        features = state.get_feature_vector(rain_risk, sc_prob)

        # 2. Predict pit probability
        df_pit = pd.DataFrame([features])[self.pit_features]
        pit_probs = self.pit_model.predict_proba(df_pit)[0]
        pit_probability = float(pit_probs[1])

        # 3. Determine base ML pit decision
        should_pit = (pit_probability >= 0.65) and (state.laps_since_last_pit() >= 5)

        # 4. Predict the recommended compound
        # If we decide to pit, evaluate compound model assuming is_pit_lap=1
        features_comp = features.copy()
        features_comp['is_pit_lap'] = 1 if should_pit else 0
        df_comp = pd.DataFrame([features_comp])[self.compound_features]

        compound_probs = self.compound_model.predict_proba(df_comp)[0]
        recommended_idx = np.argmax(compound_probs)
        recommended_compound = self.compound_labels[recommended_idx]
        compound_conf = float(compound_probs[recommended_idx])

        # 5. Rule-based rain override
        # Weather data is critical; if rain risk is high, force a pit for wets/inters
        rain_override = False
        if rain_risk >= 30.0:
            if rain_risk >= 60.0:
                target_compound = 'WET'
            else:
                target_compound = 'INTERMEDIATE'

            if state.current_compound != target_compound:
                should_pit = True
                recommended_compound = target_compound
                pit_probability = max(pit_probability, rain_risk / 100.0)
                compound_conf = 1.0
                rain_override = True

        # 6. Format recommendation output
        if should_pit:
            action = 'PIT'
            chosen_compound = recommended_compound
            confidence = pit_probability
        else:
            action = 'STAY_OUT'
            chosen_compound = state.current_compound
            confidence = 1.0 - pit_probability

        decision = {
            'action': action,
            'compound': chosen_compound,
            'confidence': round(confidence, 4),
            'rain_risk': rain_risk,
            'lap': state.current_lap,
            # Additional metadata for explanation
            '_pit_prob': round(pit_probability, 4),
            '_compound_conf': round(compound_conf, 4),
            '_rain_override': rain_override,
            '_tyre_age': state.tyre_age,
        }

        # Record decision in state memory
        state.add_decision(decision)

        return decision
