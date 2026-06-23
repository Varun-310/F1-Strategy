"""
live_feed.py -- Polls real-time race data from the OpenF1 API.
"""

import time
import requests
import threading

class LiveRaceFeed:
    """Polls OpenF1 API to track driver race state in real-time."""

    def __init__(self, session_key: str, driver_number: int):
        """Initialize with session_key and driver_number.

        Args:
            session_key: The OpenF1 session identifier (e.g. '9153').
            driver_number: The driver's car number (e.g. 44).
        """
        self.session_key = str(session_key)
        self.driver_number = int(driver_number)
        self.base_url = "https://api.openf1.org/v1"
        self._stop_event = threading.Event()
        self._thread = None

        # Fallback dictionary to return last known values if any API call fails
        self.last_known = {
            'lap_number': 0,
            'compound': 'MEDIUM',
            'tyre_life': 0,
            'position': 10,
            'gap_to_leader': 0.0,
            'lap_time': 0.0
        }

    def poll(self) -> dict:
        """Call OpenF1 endpoints and construct the latest driver state.

        Returns:
            A dictionary containing:
                - lap_number: Latest completed lap number
                - compound: Current tyre compound name (e.g., 'MEDIUM')
                - tyre_life: Estimated laps run on this compound
                - position: Current track position (1-20)
                - gap_to_leader: Gap in seconds to P1
                - lap_time: Duration of the last lap in seconds
        """
        params = {
            'session_key': self.session_key,
            'driver_number': self.driver_number
        }

        # 1. Poll laps
        try:
            r = requests.get(f"{self.base_url}/laps", params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                data.sort(key=lambda x: x.get('lap_number', 0))
                latest_lap = data[-1]
                self.last_known['lap_number'] = latest_lap.get('lap_number', self.last_known['lap_number'])
                # OpenF1 uses lap_duration for the lap time in seconds
                if latest_lap.get('lap_duration') is not None:
                    self.last_known['lap_time'] = float(latest_lap['lap_duration'])
        except Exception as e:
            print(f"Error polling OpenF1 /laps: {e}")

        # 2. Poll stints (to get current compound and starting tyre age)
        try:
            r = requests.get(f"{self.base_url}/stints", params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                data.sort(key=lambda x: x.get('stint_number', 0))
                latest_stint = data[-1]
                self.last_known['compound'] = latest_stint.get('compound', self.last_known['compound'])

                lap_start = latest_stint.get('lap_start')
                tyre_age_start = latest_stint.get('tyre_age_at_start', 0)
                # tyre_life = age_at_start + (current_lap - start_of_stint)
                if lap_start is not None and self.last_known['lap_number'] > 0:
                    current_lap = self.last_known['lap_number']
                    self.last_known['tyre_life'] = int(tyre_age_start + max(current_lap - lap_start, 0))
                else:
                    self.last_known['tyre_life'] = int(tyre_age_start)
        except Exception as e:
            print(f"Error polling OpenF1 /stints: {e}")

        # 3. Poll position
        try:
            r = requests.get(f"{self.base_url}/position", params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                data.sort(key=lambda x: x.get('date', ''))
                latest_pos = data[-1]
                self.last_known['position'] = int(latest_pos.get('position', self.last_known['position']))
        except Exception as e:
            print(f"Error polling OpenF1 /position: {e}")

        # 4. Poll intervals (for gap to leader)
        try:
            r = requests.get(f"{self.base_url}/intervals", params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                data.sort(key=lambda x: x.get('date', ''))
                latest_int = data[-1]
                val = latest_int.get('gap_to_leader')
                if val is not None:
                    try:
                        self.last_known['gap_to_leader'] = float(val)
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Error polling OpenF1 /intervals: {e}")

        return self.last_known.copy()

    def start_polling(self, callback, interval_seconds: int = 30) -> None:
        """Spawn a background thread that polls every interval_seconds.

        Args:
            callback: Function to receive the polled dict.
            interval_seconds: Delay between poll attempts.
        """
        self._stop_event.clear()

        def _polling_loop():
            while not self._stop_event.is_set():
                polled_data = self.poll()
                callback(polled_data)
                
                # Check for stop event periodically during sleep
                sleep_chunks = max(1, int(interval_seconds * 10))
                for _ in range(sleep_chunks):
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.1)

        self._thread = threading.Thread(target=_polling_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
