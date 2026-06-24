"""
llm_explainer.py -- Explains race strategy decisions using Gemini.
"""

import os
import google.generativeai as genai
from src.race_state import RaceState

def explain_decision(state: RaceState, decision: dict, rain_risk: float) -> str:
    """Build a prompt describing the race state and explain the strategy choice.

    Args:
        state: The current RaceState object.
        decision: The recommendation dictionary from StrategyEngine.recommend().
        rain_risk: The current rain probability (0 to 100).

    Returns:
        A 2-3 sentence string briefing the driver.
    """
    action = decision.get('action', 'STAY_OUT')
    target_compound = decision.get('compound', state.current_compound)
    confidence = decision.get('confidence', 1.0)
    tyre_age = state.tyre_age
    is_sc = state._is_sc_vsc

    # API Key retrieval
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # Local fallback briefing if API key is not available
    if not api_key:
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

    # Configure genai
    genai.configure(api_key=api_key)

    # Detailed strategic prompt
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
        # Use gemini-1.5-flash which is widely available and fast
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Clean up any potential markdown wrapper quotes or prefix if the LLM produced them
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text
    except Exception as e:
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
