import requests

BASE_URL = "https://api.openf1.org/v1"

def fetch_stints(session_key, driver_number):
    """
    Fetches stints for a specific driver in a session and returns the tyre compound per stint.
    Returns:
        List of dicts, e.g., [{'stint_number': 1, 'compound': 'MEDIUM'}, ...]
    """
    url = f"{BASE_URL}/stints"
    params = {
        'session_key': session_key,
        'driver_number': driver_number
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        stints = []
        if isinstance(data, list):
            # Sort by stint_number to keep them in order
            data.sort(key=lambda x: x.get('stint_number', 0))
            for item in data:
                stints.append({
                    'stint_number': item.get('stint_number'),
                    'compound': item.get('compound')
                })
        return stints
    except Exception as e:
        print(f"Error fetching stints from OpenF1 for session={session_key}, driver={driver_number}: {e}")
        return []

def fetch_weather(session_key):
    """
    Fetches weather data for a session and returns the latest weather row.
    Returns:
        Dict representing the latest weather row, or empty dict if not found.
    """
    url = f"{BASE_URL}/weather"
    params = {
        'session_key': session_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            # Sort by date/timestamp to get the latest record
            data.sort(key=lambda x: x.get('date', ''))
            return data[-1]
        return {}
    except Exception as e:
        print(f"Error fetching weather from OpenF1 for session={session_key}: {e}")
        return {}

def fetch_position(session_key, driver_number):
    """
    Fetches the current position and gap_to_leader for a driver in a session.
    Returns:
        Dict: {'position': int/None, 'gap_to_leader': float/str/None}
    """
    pos_url = f"{BASE_URL}/position"
    int_url = f"{BASE_URL}/intervals"
    params = {
        'session_key': session_key,
        'driver_number': driver_number
    }
    
    position = None
    gap_to_leader = None
    
    # 1. Fetch current position
    try:
        response = requests.get(pos_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            data.sort(key=lambda x: x.get('date', ''))
            position = data[-1].get('position')
    except Exception as e:
        print(f"Error fetching position from OpenF1 for session={session_key}, driver={driver_number}: {e}")
        
    # 2. Fetch gap to leader
    try:
        response = requests.get(int_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            data.sort(key=lambda x: x.get('date', ''))
            gap_to_leader = data[-1].get('gap_to_leader')
    except Exception as e:
        print(f"Error fetching intervals from OpenF1 for session={session_key}, driver={driver_number}: {e}")
        
    return {
        'position': position,
        'gap_to_leader': gap_to_leader
    }
