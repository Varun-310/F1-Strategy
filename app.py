"""
app.py -- Streamlit dashboard for F1 Strategy AI.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.weather import CIRCUITS, rain_risk_score
from src.pre_race import generate_briefing
from src.live_feed import LiveRaceFeed
from src.race_simulator import simulate_race
from src.safety_car import sc_probability
from src.race_state import RaceState
from src.strategy_engine import StrategyEngine
from src.llm_explainer import explain_decision, refine_decision_with_llm

# 1. Page Configuration
st.set_page_config(
    page_title="F1 Strategy AI Advisor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Main Background and Typography */
    .stApp {
        background-color: #0F0F0F;
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .f1-title {
        color: #E10600;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .f1-subtitle {
        color: #A0A0A0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Styled Card Containers */
    .briefing-box {
        background-color: #1A1A1A;
        border-left: 5px solid #E10600;
        border-radius: 4px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: #F0F0F0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Recommendation Colors */
    .recommendation-pit {
        background: linear-gradient(135deg, #3A0000 0%, #1A0000 100%);
        border: 2px solid #E10600;
        border-radius: 6px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .recommendation-stay {
        background: linear-gradient(135deg, #002A10 0%, #001005 100%);
        border: 2px solid #00E100;
        border-radius: 6px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .recommendation-caution {
        background: linear-gradient(135deg, #2D2500 0%, #120F00 100%);
        border: 2px solid #FFD700;
        border-radius: 6px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .large-action-text {
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 0.2rem;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Main Title Header
st.markdown('<div class="f1-title">🏎️ F1 Strategy AI Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="f1-subtitle">Real-time Machine Learning Strategy Pit Predictor & Radio Briefing Explainer</div>', unsafe_allow_html=True)

# 3. Sidebar Inputs
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/f2/Formula_One_logo.svg", width=150)
st.sidebar.markdown("### Race Setup")

circuit_choice = st.sidebar.selectbox("Select Circuit", list(CIRCUITS.keys()), index=list(CIRCUITS.keys()).index("Canada") if "Canada" in CIRCUITS else 0)
race_date = st.sidebar.date_input("Race Date", value=pd.to_datetime("2026-06-21"))
driver_name = st.sidebar.text_input("Driver Name (3 letters)", value="NOR").upper()[:3]
grid_position = st.sidebar.number_input("Starting Grid Position", min_value=1, max_value=20, value=3)

# Pre-Race Action Button
generate_briefing_btn = st.sidebar.button("Generate Pre-Race Briefing", type="primary")

# API Key Configuration in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 API Configuration")
env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=env_key,
    help="Enter your Gemini API key to enable LLM Co-Strategist decision refinement and driver radio briefing generation. If not set, it will look for GEMINI_API_KEY / GOOGLE_API_KEY in the environment, or fallback to local ML heuristics."
)
api_key = api_key_input if api_key_input.strip() else None

st.sidebar.markdown("---")
st.sidebar.markdown("### Live Telemetry Monitor")

start_monitor = st.sidebar.toggle("Start Live Race Monitor", value=False)
telemetry_source = st.sidebar.radio("Telemetry Source", ["Historical Replay (Simulator)", "OpenF1 API (Live Feed)"], index=0, disabled=not start_monitor)
driver_number = st.sidebar.number_input("Driver Number (OpenF1)", min_value=1, max_value=99, value=4, disabled=not start_monitor or telemetry_source == "Historical Replay (Simulator)")
session_key = st.sidebar.text_input("Session Key (OpenF1)", value="9153", disabled=not start_monitor or telemetry_source == "Historical Replay (Simulator)")

# Initialize session state variables
if 'briefing' not in st.session_state:
    st.session_state.briefing = None

if 'sim_decisions' not in st.session_state:
    st.session_state.sim_decisions = []
if 'sim_lap_index' not in st.session_state:
    st.session_state.sim_lap_index = 0
if 'decision_log' not in st.session_state:
    st.session_state.decision_log = []
if 'rain_history' not in st.session_state:
    st.session_state.rain_history = []

# Initialize strategy engine once
if 'engine' not in st.session_state:
    try:
        st.session_state.engine = StrategyEngine("models/pit_model.json", "models/compound_model.json")
    except Exception as e:
        st.session_state.engine = None
        st.sidebar.error(f"Error loading strategy models: {e}")

# Reset simulator & live state when toggle is turned off
if not start_monitor:
    st.session_state.sim_lap_index = 0
    st.session_state.decision_log = []
    st.session_state.rain_history = []
    st.session_state.sim_decisions = []
    if 'sim_params' in st.session_state:
        del st.session_state.sim_params
    if 'live_id' in st.session_state:
        del st.session_state.live_id
    if 'live_state' in st.session_state:
        del st.session_state.live_state

# Tabs layout
tab_pre_race, tab_live = st.tabs(["📋 Pre-Race Strategy", "📊 Live Telemetry Monitor"])

# =========================================================================
# TAB 1: PRE-RACE STRATEGY
# =========================================================================
with tab_pre_race:
    if generate_briefing_btn or st.session_state.briefing is not None:
        if generate_briefing_btn:
            with st.spinner("Analyzing weather forecasts and historical stint data..."):
                date_str = race_date.strftime("%Y-%m-%d")
                st.session_state.briefing = generate_briefing(circuit_choice, date_str, grid_position, driver_name, api_key=api_key)

        briefing = st.session_state.briefing

        # Styled Briefing box
        st.subheader(f"Strategy Briefing: {driver_name} @ {circuit_choice}")
        st.markdown(f'<div class="briefing-box">{briefing["briefing_text"]}</div>', unsafe_allow_html=True)

        # Pre-Race Metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Starting Compound", briefing["starting_compound"])
        m_col2.metric("Planned Pit Window", f"Lap {briefing['pit_window_start']} - {briefing['pit_window_end']}")
        m_col3.metric("Rain Risk", f"{briefing['rain_risk']:.0f}%")
        
        # Format SC risk text based on probability
        sc_prob_val = briefing["sc_risk"]
        sc_text = "Low"
        if sc_prob_val >= 0.20:
            sc_text = "High"
        elif sc_prob_val >= 0.10:
            sc_text = "Medium"
        m_col4.metric("Safety Car Risk", f"{sc_text} ({sc_prob_val:.1%})")

    # Fixed Regulation Comparison Note
    st.markdown("### 🔍 Historical Regulation Check (Canada 2026 GP Case Study)")
    st.info("""
    **What would have happened at Canada 2026 GP:**
    During early laps under wet conditions, the ML model struggled to predict wet transitions due to oversampling duplicates. 
    However, the strategy engine flagged: **"Rain risk dropping to 12% by lap 5. Do not start on Intermediates."** 
    Failing to override the model here resulted in a wasted stint compared to starting on Medium/Hard dry slicks.
    """)

# =========================================================================
# TAB 2: LIVE TELEMETRY MONITOR
# =========================================================================
with tab_live:
    if not start_monitor:
        st.warning("Activate the **Start Live Race Monitor** toggle in the sidebar to begin polling telemetry data.")
    else:
        if telemetry_source == "Historical Replay (Simulator)":
            current_sim_params = (circuit_choice, race_date.year, driver_name)
            if 'sim_params' not in st.session_state or st.session_state.sim_params != current_sim_params or not st.session_state.sim_decisions:
                st.session_state.sim_params = current_sim_params
                with st.spinner(f"Loading FastF1 historical session ({race_date.year} {circuit_choice} GP)..."):
                    decisions = simulate_race(circuit_choice, race_date.year, driver_name, api_key=api_key)
                    if not decisions and race_date.year != 2024:
                        st.warning(f"Failed to load {race_date.year} session. Falling back to 2024 historical session...")
                        decisions = simulate_race(circuit_choice, 2024, driver_name, api_key=api_key)
                    if not decisions:
                        st.error("No telemetry data found for the selected driver/circuit. Reverting to Canada 2024 GP NOR.")
                        decisions = simulate_race("Canada", 2024, "NOR", api_key=api_key)
                    
                    st.session_state.sim_decisions = decisions
                    st.session_state.sim_lap_index = 0
                    st.session_state.decision_log = []
                    st.session_state.rain_history = []

            decisions_list = st.session_state.sim_decisions
            lap_idx = st.session_state.sim_lap_index

            if decisions_list:
                if lap_idx < len(decisions_list):
                    current_decision = decisions_list[lap_idx]
                    st.session_state.decision_log.append(current_decision)
                    st.session_state.rain_history.append(current_decision['rain_risk'])
                    st.session_state.sim_lap_index += 1
                else:
                    current_decision = st.session_state.decision_log[-1]
            else:
                st.error("Simulation failed to load any data.")
                st.stop()

        else:  # OpenF1 API (Live Feed)
            if st.session_state.engine is None:
                st.error("Strategy Engine models not loaded.")
                st.stop()

            # Initialize LiveRaceFeed
            feed = LiveRaceFeed(session_key, driver_number)
            
            with st.spinner("Polling OpenF1 live telemetry..."):
                polled_data = feed.poll()

            # Get coordinates for weather
            coords = CIRCUITS.get(circuit_choice, {'lat': 45.5006, 'lon': -73.5225})
            
            # Check if live state matches current session
            current_live_id = (session_key, driver_number)
            if 'live_id' not in st.session_state or st.session_state.live_id != current_live_id or 'live_state' not in st.session_state:
                st.session_state.live_id = current_live_id
                
                # Initialize new RaceState
                st.session_state.live_state = RaceState(
                    driver=str(driver_number),
                    circuit=circuit_choice,
                    total_laps=70,  # standard
                    circuit_lat=coords['lat'],
                    circuit_lon=coords['lon'],
                    year=race_date.year,
                    current_compound=polled_data['compound'],
                    tyre_age=polled_data['tyre_life'],
                    current_position=polled_data['position'],
                    gap_to_leader=polled_data['gap_to_leader'],
                    last_lap_time=polled_data['lap_time']
                )
                st.session_state.decision_log = []
                st.session_state.rain_history = []
                
                # Make initial decision at Lap 0
                rain_risk = rain_risk_score(coords['lat'], coords['lon'], race_date)
                sc_prob = sc_probability(circuit_choice, 0, 70)
                
                decision = st.session_state.engine.recommend(st.session_state.live_state, rain_risk, sc_prob)
                llm_dec = refine_decision_with_llm(st.session_state.live_state, decision, rain_risk, api_key=api_key)
                decision['action'] = llm_dec['action']
                decision['compound'] = llm_dec['compound']
                decision['confidence'] = llm_dec['confidence']
                decision['explanation'] = llm_dec['explanation']
                decision['override_applied'] = llm_dec['override_applied']
                decision['position'] = polled_data['position']
                decision['gap_to_leader'] = polled_data['gap_to_leader']
                
                st.session_state.decision_log.append(decision)
                st.session_state.rain_history.append(rain_risk)
            
            # Update state if a new lap is completed
            state = st.session_state.live_state
            if polled_data['lap_number'] > state.current_lap:
                # Calculate rain risk and SC probability
                rain_risk = rain_risk_score(coords['lat'], coords['lon'], race_date)
                sc_prob = sc_probability(circuit_choice, polled_data['lap_number'], 70)
                
                # Construct lap update
                lap_data = {
                    'lap_time': polled_data['lap_time'],
                    'position': polled_data['position'],
                    'gap_to_leader': polled_data['gap_to_leader'],
                    'track_status': 1,  # default
                    'field_pit_fraction': 0.0
                }
                weather_data = {
                    'rain_probability': rain_risk,
                    'temperature': 20.0
                }
                
                # Handle compound change pit record
                if polled_data['compound'] != state.current_compound:
                    state.record_pit(polled_data['compound'])
                
                state.update(lap_data, weather_data)
                
                decision = st.session_state.engine.recommend(state, rain_risk, sc_prob)
                llm_dec = refine_decision_with_llm(state, decision, rain_risk, api_key=api_key)
                decision['action'] = llm_dec['action']
                decision['compound'] = llm_dec['compound']
                decision['confidence'] = llm_dec['confidence']
                decision['explanation'] = llm_dec['explanation']
                decision['override_applied'] = llm_dec['override_applied']
                decision['position'] = polled_data['position']
                decision['gap_to_leader'] = polled_data['gap_to_leader']
                
                st.session_state.decision_log.append(decision)
                st.session_state.rain_history.append(rain_risk)
            
            current_decision = st.session_state.decision_log[-1]

            # Display OpenF1 connection status
            if polled_data['lap_number'] == 0:
                st.sidebar.info("📡 OpenF1: Connected (using fallback/waiting for session)")
            else:
                st.sidebar.success(f"📡 OpenF1: Connected (Lap {polled_data['lap_number']})")

        # Show LLM verification status
        if api_key:
            st.success("🟢 **LLM Co-Strategist Active (Gemini 2.5 Flash)**: Decisions are verified and refined in real time.")
        else:
            st.info("ℹ️ **Local Heuristics Active**: Using XGBoost predictions + rule fallback (No API key provided).")

        # Live telemetry metrics
        col_lap, col_tyre, col_pos, col_gap = st.columns(4)
        
        # Lap
        if telemetry_source == "Historical Replay (Simulator)":
            total_laps_str = f" / {len(decisions_list)}"
        else:
            total_laps_str = ""
        col_lap.metric("Current Lap", f"Lap {current_decision['lap']}{total_laps_str}")
        
        # Stint compound age
        tyre_age = current_decision.get('_tyre_age', 0)
        col_tyre.metric("Tyre", f"{current_decision['compound']} ({tyre_age} Laps)")
        
        # Dynamic track position
        col_pos.metric("Track Position", f"P{current_decision.get('position', 10)}")
        
        # Dynamic gap to leader
        gap_val = current_decision.get('gap_to_leader', 0.0)
        gap_str = f"+{gap_val:.2f}s" if gap_val > 0 else "Leader"
        col_gap.metric("Gap to Leader", gap_str)

        # Color-coded strategy recommendation display
        action = current_decision['action']
        conf = current_decision['confidence']
        
        if action == 'STAY_OUT':
            rec_class = "recommendation-stay"
            rec_color = "#00E100"
            rec_title = "STAY OUT"
        elif action == 'PIT':
            if current_decision.get('_stint_override') or current_decision.get('_rain_override'):
                rec_class = "recommendation-pit"
                rec_color = "#E10600"
                rec_title = f"BOX ({current_decision['compound']})"
            elif 0.40 <= conf < 0.65:
                rec_class = "recommendation-caution"
                rec_color = "#FFD700"
                rec_title = f"BOX ({current_decision['compound']}) - LOW CONFIDENCE"
            else:
                rec_class = "recommendation-pit"
                rec_color = "#E10600"
                rec_title = f"BOX ({current_decision['compound']})"
        else:
            rec_class = "recommendation-stay"
            rec_color = "#00E100"
            rec_title = "STAY OUT"

        st.markdown(f"""
        <div class="{rec_class}">
            <div style="color: {rec_color};" class="large-action-text">{rec_title}</div>
            <div style="font-size: 1.1rem; color: #CCCCCC; margin-bottom: 1rem;">Confidence: {conf:.1%}</div>
            <div style="font-size: 1.2rem; color: #FFFFFF; font-style: italic;">
                " {current_decision.get('explanation', '')} "
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Alert if LLM Co-Strategist has overrode the XGBoost ML recommendation
        if current_decision.get('override_applied'):
            st.warning("⚠️ **LLM Co-Strategist Override Applied**: The AI strategist overrode the machine learning model's recommendation due to race dynamics.")

        # Line chart of Rain Risk history
        st.subheader("🌧️ Rain Risk Trend")
        rain_df = pd.DataFrame({
            "Lap": range(1, len(st.session_state.rain_history) + 1),
            "Rain Probability (%)": st.session_state.rain_history
        }).set_index("Lap")
        st.line_chart(rain_df, color="#E10600")

        # Scrollable Decisions Log DataFrame
        st.subheader("📋 Decision Log History")
        log_df = pd.DataFrame(st.session_state.decision_log)
        
        # Ensure override_applied exists in DataFrame
        if 'override_applied' not in log_df.columns:
            log_df['override_applied'] = False
        log_df['override_applied'] = log_df['override_applied'].fillna(False).map({True: "Yes", False: "No"})

        # Cleanup log dataframe columns for cleaner display
        display_columns = {
            'lap': 'Lap',
            'action': 'Action',
            'compound': 'Compound',
            'confidence': 'Confidence',
            'rain_risk': 'Rain Risk (%)',
            'override_applied': 'LLM Override?'
        }
        log_df = log_df[list(display_columns.keys())].rename(columns=display_columns)
        log_df['Confidence'] = log_df['Confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(log_df, use_container_width=True, height=250)

        # Auto-refresh loop: if monitor is active, sleep and rerun
        if start_monitor:
            if telemetry_source == "Historical Replay (Simulator)":
                if st.session_state.sim_lap_index < len(decisions_list):
                    st.caption("ℹ️ Simulator Mode: Running at 10x speed (3s auto-refresh) for test replay.")
                    time.sleep(3)
                    st.rerun()
            else:
                st.caption("ℹ️ Live Feed Mode: Polling OpenF1 API (30s auto-refresh).")
                time.sleep(30)
                st.rerun()
