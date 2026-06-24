"""
llm_explainer.py -- Explains and refines race strategy decisions using Google GenAI SDK.
"""

import os
from typing import Optional, Dict
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from src.race_state import RaceState

class StrategicDecision(BaseModel):
    action: str = Field(description="The recommended strategy action: 'PIT' or 'STAY_OUT'")
    compound: str = Field(description="The recommended tyre compound to fit: 'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', or 'WET'")
    confidence: float = Field(description="The confidence of the decision between 0.0 and 1.0")
    reasoning: str = Field(description="A concise 2-3 sentence strategic explanation briefing to the driver. Speak directly to them as their race engineer (e.g. 'Box, box, NOR. We are pitting for Mediums...')")
    override_applied: bool = Field(description="Whether the LLM overrode the machine learning model's recommendation")

def get_genai_client(api_key: Optional[str] = None) -> Optional[genai.Client]:
    """Helper to initialize the google-genai client."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception as e:
        print(f"Error initializing GenAI client: {e}")
        return None

def explain_decision(state: RaceState, decision: dict, rain_risk: float, api_key: Optional[str] = None) -> str:
    """Build a prompt describing the race state and explain the strategy choice.

    Args:
        state: The current RaceState object.
        decision: The recommendation dictionary from StrategyEngine.recommend().
        rain_risk: The current rain probability (0 to 100).
        api_key: Optional Gemini API key.

    Returns:
        A 2-3 sentence string briefing the driver.
    """
    action = decision.get('action', 'STAY_OUT')
    target_compound = decision.get('compound', state.current_compound)
    confidence = decision.get('confidence', 1.0)
    tyre_age = state.tyre_age
    is_sc = state._is_sc_vsc

    client = get_genai_client(api_key)

    # Local fallback briefing if client is not available
    if not client:
        if action == 'PIT':
            if decision.get('_rain_override'):
                return (f"Box, box, {state.driver}. Heavy rain is coming in, and the risk has spiked to {rain_risk:.0f}%. "
                        f"We are pitting immediately for {target_compound} tyres. Watch the white line on exit.")
            if decision.get('_stint_override'):
                return (f"Box, box, {state.driver}. We have hit the regulation stint limit on the {state.current_compound} tyres. "
                        f"Pitting now to fit a fresh set of {target_compound} tyres. Push hard on entry.")
            return (f"Box, box, {state.driver}. We are pitting this lap for a fresh set of {target_compound} tyres. "
                    f"Your current stint is at {tyre_age} laps and the degradation is catching up. Push hard on pit entry.")
        else:
            if rain_risk >= 20.0:
                return (f"Stay out, {state.driver}. We are monitoring the rain risk at {rain_risk:.0f}%, but the track is still dry enough. "
                        f"Keep managing the current {state.current_compound} tyres and maintain the gap.")
            return (f"Stay out, {state.driver}. The pace is strong and tyre life is stable. "
                    f"We will extend this stint on the {state.current_compound} tyres. Maintain your current rhythm.")

    # Detailed strategic prompt for briefing
    prompt = f"""You are an elite Formula 1 Race Engineer briefing your driver {state.driver} during the race at {state.circuit}.
Analyze the current race state and explain the strategic decision made by our AI models.

Current Race State:
- Driver: {state.driver}
- Circuit: {state.circuit}
- Current Lap: {state.current_lap} of {state.total_laps}
- Current Position: P{state.current_position}
- Gap to Leader: {state.gap_to_leader:.1f}s
- Last Lap Time: {state.last_lap_time:.3f}s
- Current Tyre: {state.current_compound} ({state.tyre_age} laps old)
- Rain Risk: {rain_risk:.1f}%
- Safety Car / VSC Active: {"Yes" if is_sc else "No"}

AI Strategic Decision:
- Action: {action}
- Next Tyre Compound: {target_compound}
- Decision Confidence: {confidence * 100:.1f}%
- Rain Override: {"Yes" if decision.get('_rain_override') else "No"}
- Stint Limit Override: {"Yes" if decision.get('_stint_override') else "No"}

Provide a concise, professional briefing to the driver in exactly 2-3 sentences. Speak directly to {state.driver} as their race engineer (e.g. "Box this lap, {state.driver}..."). Do not include any markdown format, quotation marks, or conversational filler outside the briefing. Focus on tyre wear, safety car opportunities, or rain risks if applicable.
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        text = response.text.strip()
        # Clean up any potential markdown wrapper quotes or prefix if the LLM produced them
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text
    except Exception as e:
        print(f"Error in explain_decision Gemini call: {e}")
        # Fallback to local briefing if API call fails
        if action == 'PIT':
            if decision.get('_rain_override'):
                return (f"Box, box, {state.driver}. Rain risk is at {rain_risk:.0f}%. We are pitting now for {target_compound}s. "
                        f"Watch the grip levels on pit entry.")
            if decision.get('_stint_override'):
                return (f"Box, box, {state.driver}. Stint limit reached. We are pitting now to fit {target_compound} tyres.")
            return (f"Box, box, {state.driver}. We are pitting this lap for {target_compound} tyres. "
                    f"Let's cover the cars behind and make this stop count.")
        else:
            return (f"Stay out, {state.driver}. The tyres are holding up well and our pace is stable. "
                    f"Keep extending this stint on the {state.current_compound}s.")

def refine_decision_with_llm(state: RaceState, ml_decision: dict, rain_risk: float, api_key: Optional[str] = None) -> dict:
    """Uses Gemini to evaluate, refine, and explain the XGBoost strategy decision.

    Args:
        state: The current RaceState object.
        ml_decision: The recommendation dictionary from StrategyEngine.recommend().
        rain_risk: The current rain probability (0 to 100).
        api_key: Optional Gemini API key.

    Returns:
        A dict matching the StrategicDecision schema with compatibility keys:
            - action: 'PIT' or 'STAY_OUT'
            - compound: Recommended compound
            - confidence: Confidence probability
            - explanation: The briefing text explanation
            - override_applied: Boolean indicating if LLM overrode ML
    """
    client = get_genai_client(api_key)

    # Local fallback decision if client is not available
    if not client:
        return {
            'action': ml_decision.get('action', 'STAY_OUT'),
            'compound': ml_decision.get('compound', state.current_compound),
            'confidence': ml_decision.get('confidence', 1.0),
            'explanation': explain_decision(state, ml_decision, rain_risk, api_key=None),
            'override_applied': False
        }

    # Detailed strategic prompt for F1 Chief Strategist
    prompt = f"""You are the elite Formula 1 Chief Race Strategist for driver {state.driver} at the {state.circuit} Grand Prix.
Your job is to evaluate the machine learning model's recommendation and either confirm it or override it based on critical safety, regulations, and strategic rules.

Current Race State:
- Driver: {state.driver}
- Circuit: {state.circuit}
- Current Lap: {state.current_lap} of {state.total_laps}
- Track Position: P{state.current_position}
- Gap to Leader: {state.gap_to_leader:.1f}s
- Last Lap Time: {state.last_lap_time:.3f}s
- Current Tyre: {state.current_compound} ({state.tyre_age} laps old)
- Rain Risk: {rain_risk:.1f}%
- Safety Car / VSC Active: {"Yes" if state._is_sc_vsc else "No"}

XGBoost ML Model Recommendations:
- Recommended Action: {ml_decision.get('action')}
- Recommended Tyre Compound: {ml_decision.get('compound')}
- ML Confidence: {ml_decision.get('confidence') * 100:.1f}%
- Rain Override Flag: {"Yes" if ml_decision.get('_rain_override') else "No"}
- Stint Limit Override Flag: {"Yes" if ml_decision.get('_stint_override') else "No"}

Key Strategy Rules:
1. WEATHER: Wet tyres (WET) are for heavy rain (>60% rain risk). Intermediates (INTERMEDIATE) are for damp/light rain (30-60% rain risk). Dry slicks (SOFT/MEDIUM/HARD) are for dry tracks (<30% rain risk).
   CRITICAL OVERRIDE: If rain risk is high but dropping off rapidly (e.g. dropping to <15% within 5 laps), do NOT fit intermediates/wets. Avoid starting on intermediates if the rain is clearing.
2. STINT LIMITS (2026 regulations): SOFT max 15 laps, MEDIUM max 25 laps, HARD max 35 laps. Pitting is mandatory if these limits are reached.
3. SAFETY CAR / VSC: Pitting under VSC/SC saves ~10-15 seconds. If tyre age is close to the pit window (e.g. within 5-6 laps of typical tyre life) and VSC/SC is active, PIT immediately.
4. OVERRIDES: If the ML model recommends STAY_OUT but tyres are extremely degraded or rain is coming, you can override to PIT. If it recommends PIT for intermediates but weather is clearing (e.g. Canada 2026 GP where rain risk drops), override to STAY_OUT on dry slicks.

If you choose to override, set `override_applied` to true, update the `action` and/or `compound`, adjust the `confidence` to reflect your decision, and explain the strategic reasoning to the driver.
Otherwise, confirm the ML recommendation, set `override_applied` to false, and generate a reassuring explanation.
"""

    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StrategicDecision,
            system_instruction="You are a precise, data-driven F1 Chief Race Strategist. You must follow the strategy rules strictly and produce structured, JSON-conforming strategic outputs."
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config,
        )
        # Parse the structured response
        import json
        result = json.loads(response.text)
        return {
            'action': result.get('action', ml_decision.get('action', 'STAY_OUT')),
            'compound': result.get('compound', ml_decision.get('compound', state.current_compound)),
            'confidence': float(result.get('confidence', ml_decision.get('confidence', 1.0))),
            'explanation': result.get('reasoning', ''),
            'override_applied': bool(result.get('override_applied', False))
        }
    except Exception as e:
        print(f"Error in refine_decision_with_llm Gemini call: {e}")
        return {
            'action': ml_decision.get('action', 'STAY_OUT'),
            'compound': ml_decision.get('compound', state.current_compound),
            'confidence': ml_decision.get('confidence', 1.0),
            'explanation': explain_decision(state, ml_decision, rain_risk, api_key=api_key),
            'override_applied': False
        }
