import requests
from datetime import datetime, timezone, timedelta

# Lat/Lon coordinates for the circuits
CIRCUITS = {
    'Canada': {'lat': 45.5006, 'lon': -73.5225},
    'Silverstone': {'lat': 52.0710, 'lon': -1.0160},
    'Spa': {'lat': 50.4372, 'lon': 5.9714},
    'Monza': {'lat': 45.6206, 'lon': 9.2894},
    'Japan': {'lat': 34.8417, 'lon': 136.5389}
}

def get_forecast(lat, lon):
    """
    Calls the Open-Meteo API and returns hourly precipitation_probability and
    temperature_2m for the next 24 hours.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation_probability,temperature_2m&timezone=UTC"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly_data = data.get('hourly', {})
        times = hourly_data.get('time', [])
        precip_probs = hourly_data.get('precipitation_probability', [])
        temps = hourly_data.get('temperature_2m', [])
        
        now_utc = datetime.now(timezone.utc)
        forecast_24h = []
        
        for t_str, prob, temp in zip(times, precip_probs, temps):
            # Parse the time string (Open-Meteo returns ISO strings like 2026-06-03T12:00)
            t_dt = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)
            
            # We want the next 24 hours starting from the current hour.
            # Allow items starting from 30 mins ago up to 24 hours from now.
            if now_utc - timedelta(minutes=30) <= t_dt < now_utc + timedelta(hours=24):
                forecast_24h.append({
                    'time': t_dt.isoformat(),
                    'precipitation_probability': prob,
                    'temperature_2m': temp
                })
        
        # Ensure we return at most 24 hours of forecast
        return forecast_24h[:24]
    except Exception as e:
        print(f"Error fetching forecast for lat={lat}, lon={lon}: {e}")
        return []

def rain_risk_score(lat, lon, race_hour):
    """
    Returns a 0-100 score for rain likelihood in the 2 hours around race_hour.
    The race_hour can be:
      - a datetime object (naive or aware)
      - an ISO 8601 string (e.g. '2026-06-03T15:00:00Z')
      - an integer/float representing the hour of today (UTC)
    """
    forecast = get_forecast(lat, lon)
    if not forecast:
        return 0
    
    # Resolve race_hour to a datetime object in UTC
    if isinstance(race_hour, str):
        try:
            # Handle Z suffix
            clean_str = race_hour.replace("Z", "+00:00")
            race_dt = datetime.fromisoformat(clean_str)
        except ValueError:
            # Try parsing simple hour format like '14:00'
            try:
                parts = list(map(int, race_hour.split(':')))
                race_dt = datetime.now(timezone.utc).replace(hour=parts[0], minute=parts[1], second=0, microsecond=0)
            except Exception:
                print(f"Failed to parse race_hour string: {race_hour}")
                return 0
    elif isinstance(race_hour, (int, float)):
        race_dt = datetime.now(timezone.utc).replace(hour=int(race_hour), minute=0, second=0, microsecond=0)
    elif isinstance(race_hour, datetime):
        race_dt = race_hour
    else:
        print(f"Unsupported race_hour type: {type(race_hour)}")
        return 0
        
    if race_dt.tzinfo is None:
        race_dt = race_dt.replace(tzinfo=timezone.utc)
        
    matched_probs = []
    for item in forecast:
        item_dt = datetime.fromisoformat(item['time'])
        if item_dt.tzinfo is None:
            item_dt = item_dt.replace(tzinfo=timezone.utc)
        
        # Calculate time difference in seconds
        time_diff = abs((item_dt - race_dt).total_seconds())
        # Within the 2-hour window around race_hour (e.g., +/- 1 hour from the target hour)
        if time_diff <= 3600:
            matched_probs.append(item['precipitation_probability'])
            
    if not matched_probs:
        # Fallback: if race_hour is outside the next 24 hours, we might not have matches in the 24-hour sliced forecast.
        # In this case, calculate from the full API response if possible, or return 0.
        return 0
        
    # Return the maximum probability in the window as the risk score
    return max(matched_probs)
