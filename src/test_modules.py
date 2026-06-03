import sys
from weather import get_forecast, rain_risk_score, CIRCUITS
from openf1 import fetch_stints, fetch_weather, fetch_position

print("=== Weather Module Verification ===")
for circuit, coords in CIRCUITS.items():
    print(f"Testing {circuit} ({coords['lat']}, {coords['lon']})...")
    forecast = get_forecast(coords['lat'], coords['lon'])
    print(f"  Forecast length (expecting 24): {len(forecast)}")
    if forecast:
        print(f"  First hour time: {forecast[0]['time']}")
        print(f"  First hour temp: {forecast[0]['temperature_2m']} C")
        print(f"  First hour precip prob: {forecast[0]['precipitation_probability']}%")
    
    # Test rain risk score for current hour (e.g., as an integer)
    risk = rain_risk_score(coords['lat'], coords['lon'], 14)
    print(f"  Rain risk score at 14:00 (UTC): {risk}")

print("\n=== OpenF1 Module Verification ===")
# Let's test with a public historical session key (e.g. 7953, which is a session key from the 2023 Bahrain GP Race)
session_key = 7953
driver_number = 1 # Max Verstappen

print(f"Testing stints for session {session_key}, driver {driver_number}...")
stints = fetch_stints(session_key, driver_number)
print(f"  Stints: {stints}")

print(f"Testing weather for session {session_key}...")
latest_weather = fetch_weather(session_key)
print(f"  Latest weather: {latest_weather}")

print(f"Testing position/gap for session {session_key}, driver {driver_number}...")
pos_gap = fetch_position(session_key, driver_number)
print(f"  Position/Gap: {pos_gap}")
