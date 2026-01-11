import time
import csv
import random
from datetime import datetime, timedelta

def collect_moisture_data(duration_days=2, interval_minutes=30):
    """
    Simulate soil moisture data collection (2 days, 30-min interval)
    Generates realistic sensor data with moisture/temperature/light trends
    """
    filename = "soil_moisture.csv"
    start_time = datetime.now()
    
    # CSV headers (match project requirements)
    header = [
        "timestamp", 
        "soil_moisture_percent", 
        "temperature_c", 
        "light_intensity_lux", 
        "plant_health_status"
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Calculate total data points (2 days = 96 points)
        total_intervals = int(duration_days * 24 * 60 / interval_minutes)
        
        for i in range(total_intervals):
            # Simulate time progression (start from 6 AM)
            current_time = start_time.replace(hour=6, minute=0, second=0, microsecond=0)
            current_time += timedelta(minutes=interval_minutes * i)
            
            # Realistic moisture trend simulation (with watering event)
            if i < 12:          # Morning: gradual decrease
                moisture = 48.2 - i * 0.3
            elif i == 12:       # Noon: watering event (moisture spike)
                moisture = 55.6
            elif i < 20:        # Afternoon: faster decrease
                moisture = 55.6 - (i-12) * 1.2
            else:               # Evening: slow recovery
                moisture = 47.7 + (i-20) * 0.45
            
            # Temperature (daily cycle: 18-28°C)
            hour = 6 + i * 0.5  # 0.5h per interval
            temperature = 18 + 10 * abs(12 - hour) / 6
            
            # Light intensity (peak at noon: 10000 lux)
            light = int(10000 * abs(1 - abs(12 - hour) / 6))
            
            # Plant health status
            if i == 12:
                status = "post_watering"
            elif 44 <= moisture <= 55:
                status = "optimal"
            else:
                status = "normal"
            
            # Add realistic sensor noise
            moisture += random.uniform(-0.5, 0.5)
            temperature += random.uniform(-0.3, 0.3)
            light = max(0, light + random.randint(-100, 100))  # Ensure non-negative
            
            # Write row to CSV
            row = [
                current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                round(moisture, 1),
                round(temperature, 1),
                light,
                status
            ]
            writer.writerow(row)
    
    print(f"✅ Simulated data generated: {total_intervals} points saved to {filename}")

# Run data generation
if __name__ == "__main__":
    collect_moisture_data(duration_days=2, interval_minutes=30)
