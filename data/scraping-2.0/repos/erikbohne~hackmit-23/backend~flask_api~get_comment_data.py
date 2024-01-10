"""
Returns a comment based on the data in the gpx file
"""
import pandas as pd
import openai
import json
import math

# Haversine distance formula for calculating distance
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    R = 6371.0  # Earth radius in kilometers
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2)**2 + 
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def get_comment_data(df):
    
    # convert df to float
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)
    df['Elevation (m)'] = df['Elevation (m)'].astype(float)
    df['Heart Rate'] = df['Heart Rate'].astype(float)
    df['Power'] = df['Power'].astype(float)

    # Calculate distance between consecutive points and sum for total distance
    df['Distance'] = df.apply(lambda row: haversine_distance(row['Latitude'], row['Longitude'], 
                                                            df.at[row.name - 1, 'Latitude'] if row.name > 0 else row['Latitude'], 
                                                            df.at[row.name - 1, 'Longitude'] if row.name > 0 else row['Longitude']), axis=1)
    distance = df['Distance'].sum()
    duration = df['timestamp'].count() * 1  # Assuming each row represents a 1 second interval
    avg_speed = distance / (duration / 3600)  # Convert duration to hours
    total_elevation_gain = df['Elevation (m)'].diff().where(df['Elevation (m)'].diff() > 0).sum()
    total_elevation_loss = -df['Elevation (m)'].diff().where(df['Elevation (m)'].diff() < 0).sum()
    avg_heart_rate = df['Heart Rate'].mean()
    max_heart_rate = df['Heart Rate'].max()
    variablilty_hr = df['Heart Rate'].std() / avg_heart_rate
    avg_power = df['Power'].mean()
    variability_index = df['Power'].std() / avg_power

    # Calculate rolling averages for Power and Heart Rate
    df['Rolling Power'] = df['Power'].rolling(window=5).mean()
    df['Rolling Heart Rate'] = df['Heart Rate'].rolling(window=5).mean()

    # Identify intervals of low power (below 70% of average) as potential struggle points
    df['Low Power Interval'] = df['Rolling Power'] < 0.7 * avg_power

    # Identify intervals of high heart rate (above 130% of average) as potential intense efforts
    df['High Heart Rate Interval'] = df['Rolling Heart Rate'] > 1.3 * avg_heart_rate

    # 3. Calculate Intervals, Zones, and Elevation Impact
    zones = {'Zone 1': (0, 140), 'Zone 2': (140, 160), 'Zone 3': (160, 220)}
    zone_times = {}
    for zone, (min_hr, max_hr) in zones.items():
        time_in_zone = df[(df['Heart Rate'] >= min_hr) & (df['Heart Rate'] < max_hr)]['Heart Rate'].count()
        zone_times[zone] = time_in_zone

    power_zones = {'Low': (0, 100), 'Medium': (100, 250), 'High': (250, 400)}
    power_zone_times = {}
    for zone, (min_power, max_power) in power_zones.items():
        time_in_zone = df[(df['Power'] >= min_power) & (df['Power'] < max_power)]['Power'].count()
        power_zone_times[zone] = time_in_zone

    # 4. Extract summary details
    struggle_intervals = df[df['Low Power Interval']]
    intense_intervals = df[df['High Heart Rate Interval']]

    num_struggle_intervals = len(struggle_intervals)
    avg_struggle_elevation = struggle_intervals['Elevation (m)'].mean()

    num_intense_intervals = len(intense_intervals)
    avg_intense_elevation = intense_intervals['Elevation (m)'].mean()
    
    # 4.5 Create a JSON object
    run_data = {
        "duration": f"{duration//3600}:{duration%3600//60}:{duration%60}",
        "distance": distance,
        "average_pace": (duration / 60) / distance, # in minutes per kilometer
        "segments": [], 
        "max_heart_rate": max_heart_rate,
        "min_heart_rate": df['Heart Rate'].min(),
        "average_heart_rate": avg_heart_rate,
        "cadence": None, # You don't seem to have this metric
        "weather": None, # You don't seem to have this metric
    }
    
    # Calculate segments (splits every 1km)
    segment_distance = 0
    segment_duration = 0
    segment_elevation_gain = 0
    segment_elevation_loss = 0
    segment_heart_rates = []
    segment_power_readings = []
    
    for i, row in df.iterrows():
        segment_distance += row['Distance']
        segment_duration += 1  # Assuming each row is 1 second
        if i > 0:
            elevation_diff = row['Elevation (m)'] - df.at[i - 1, 'Elevation (m)']
            if elevation_diff > 0:
                segment_elevation_gain += elevation_diff
            else:
                segment_elevation_loss -= elevation_diff
        
        segment_heart_rates.append(row['Heart Rate'])
        
        segment_power_readings.append(row['Power'])
        
        # If this segment has reached 1km or if this is the last data point
        if segment_distance >= 1 or i == len(df) - 1:
            run_data["segments"].append({
                "segment_id": len(run_data["segments"]) + 1,
                "distance": segment_distance,
                "duration": f"{segment_duration//3600}:{segment_duration%3600//60}:{segment_duration%60}",
                "pace": segment_duration / 60 / segment_distance,
                "elevation_gain": segment_elevation_gain,
                "elevation_loss": segment_elevation_loss,
                "average_heart_rate": sum(segment_heart_rates) / len(segment_heart_rates),
                "max_heart_rate": max(segment_heart_rates),
                "min_heart_rate": min(segment_heart_rates),
                "average_power": sum(segment_power_readings) / len(segment_power_readings),
            })
            
            # Reset segment data
            segment_distance = 0
            segment_duration = 0
            segment_elevation_gain = 0
            segment_elevation_loss = 0
            segment_heart_rates = []

    # 5. Use OpenAI API to get insights
    with open('config.json', 'r') as file:
        config = json.load(file)
        openai.api_key = config["OPENAI_API_KEY"]

    prompt_text = f"""
    Based on the following metrics for this run, provide a user-friendly analysis in a maximum of three sentences, like a coach would do:
    {run_data}
    Objective: Offer encouraging, concise, and specific feedback that's easily understandable, without restating the metrics.
    
    Here are two examples of feedback:
    
    You had great progress during this run as your pace kept improving throughout the run without a significant increase in heart rate. Keep up the good work, consistency is key!
    
    Do you feel tired? You heart rate was increasing throughout the run at the same pace, which is not a good sign. You should consider taking a break.
    
    We are interested in getting feedback for the following metrics:
    - Was the session too easy or too hard? (e.g. bad progression in pace and heart rate)
    - Did the user struggle at any point? (e.g. low power, high heart rate, etc.)
    - Did the user have any intense efforts? 
    - What type of run was this? (e.g. long run, tempo run, interval run, etc.)
    - Any notable achievements? (e.g. really fast pace, really high heart rate, etc.)
    
    Try to keep the feedback as personal as possible, and refer to the session with examples of what the user did well and what they could improve.
    
    Only provide the comment.
    """

    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt_text, max_tokens=500)
    insights = response.choices[0].text.strip()

    return insights
