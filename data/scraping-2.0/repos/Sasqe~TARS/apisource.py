import requests
import datetime
from dotenv import load_dotenv
import os

load_dotenv()
# MODULE: apisource.py
# LAST UPDATED: 03/25/2023
# AUTHOR: CHRIS KING
# FUNCTION : Query external API's and return data

class apiQuery():
    
    # Function to query GPT-3 Model
    # Using GPT-3.5 Turbo model
    # Using ChatCompletion and passing dict array of messages
    # PARAMETERS: messages (dict array)
    # SOURCE MODULE: tars.py
    # RETURNS: response text from GPT-3 model
    def queryGPT(self, messages):
        import openai  # Import openAI library
        # Import openAI key
        openai.api_key = os.getenv("AI_KEY")
         # Configure GPT-3 Neural Network
        print("Res")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages # Array Dict of messages formatted as {"role": "user"/"assistant"/"system", "content": message}
        )
        
        res = response["choices"][0]["message"]["content"] # Selecting first choice's content
        return res
    
    # Function to query weather network API for forecast data
    # API for endpoint coordinates, we are using Phoenix AZ
    # Forecasting endpoint excludes current data and alerts, units set to imperial
    # PARAMETERS: intent, day (datetime object)
    # SOURCE MODULE: tars.py
    # RETURNS: data returned from API endpoint
    def queryForecast(self, intent, day):
        try:
            data = ""
            # First, we use API endpoint to generate latitude and longitude based on city and state.
            georeq = f'http://api.openweathermap.org/geo/1.0/direct?q=Phoenix,AZ,US&appid={os.getenv("WEATHER_KEY")}'
            georesponse = requests.get(georeq)
            geojson = georesponse.json()[0]
            lat = geojson["lat"]
            lon = geojson["lon"]
            # Next, we pass latitude and longitude into weather API endpoint
            endpoint = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=currently,alerts&units=imperial&appid={os.getenv("WEATHER_KEY")}'
            response = requests.get(endpoint)  # Send query to API endpoint
            responsejson = response.json()  # Convert to JSON
            
            if intent == "uviForecast":  # <-- If intent is uviForecast
                
                for daily_data in responsejson['daily']: # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        
                        uv_index = daily_data['uvi'] # <-- If day matches, set uv_index to the uvi value
                        # Create data dictionary with the day, date, and uv_index
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "uvi": uv_index
                        }
                        break
                    
            elif intent == "windForecast": # <-- If intent is uviForecast
                
                for daily_data in responsejson['daily']:  # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        
                        wind_speed = daily_data['wind_speed'] # <-- If day matches, set wind speed to the wind_speed value
                        # Create data dictionary with the day, date, and wind_speed
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "wind_speed": wind_speed
                        }
                        break
                    
            elif intent == "rainForecast": # <-- If intent is rainForecast
                
                for daily_data in responsejson['daily']: # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        
                        try: # <-- Try rain at on daily forecast
                            rain = daily_data['rain']
                            pop = daily_data['pop']
                        except: # <-- If no rain, return 0
                            rain = None 
                            pop = 0 
                        # Create data dictionary with day, date, and rain
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "pop": pop,
                            "rain": rain
                        }
                        break
            
            elif intent == "highTempForecast": # <-- If intent is highTempForecast
                for daily_data in responsejson['daily']: # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        
                        temp = daily_data['temp']['max'] # <-- If day matches, set temp to the max temp value
                        # Create data dictionary with the day, date, and high temp
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "high_temp": temp
                        }
                        break
                    
            elif intent == "lowTempForecast": # <-- If intent is lowTempForecast
                for daily_data in responsejson['daily']: # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        
                        temp = daily_data['temp']['min'] # <-- If day matches, set temp to the min temp value
                        # Create data dictionary with the day, date, and low temp
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "low_temp": temp
                        }
                        break
            elif intent == "tempForecast": # <-- If intent is lowTempForecast
                for daily_data in responsejson['daily']: # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        
                        avg_temp = daily_data['temp']['day'] # <-- If day matches, set temp to the avg temp value
                        temp_min = daily_data['temp']['min']
                        temp_max = daily_data['temp']['max']
                        # Create data dictionary with the day, date, and low temp
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "avg_temp": avg_temp,
                            "temp_min": temp_min,
                            "temp_max": temp_max
                        }
                        break
            elif intent == "dewForecast": # <-- If intent is lowTempForecast
                for daily_data in responsejson['daily']: # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        
                        dew_point = daily_data['dew_point'] # <-- If day matches, set dew point
                        # Create data dictionary with the day, date, and low temp
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "dew_point": dew_point,
                        }
                        break
            elif intent == "weatherForecast": # <-- If intent is weatherForecast
                for daily_data in responsejson['daily']: # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        # Set variables to the values in the daily forecast: avg_temp, min_temp, max_temp, wind_speed, wind_gust, clouds, pop
                        avg_temp = daily_data['temp']['day']
                        min_temp = daily_data['temp']['min']
                        max_temp = daily_data['temp']['max']
                        
                        wind_speed = daily_data['wind_speed']
                        wind_gust = daily_data['wind_gust']
                        
                        clouds = daily_data['clouds']
                        
                        pop = daily_data['pop']
                        rain_desc = "rain"
                        for weather in daily_data['weather']:
                            if weather['main'] == "Rain":
                                rain_desc = weather['description']
                        
                        # Create data dictionary with the day, date, and all the variables
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "avg_temp": avg_temp,
                            "min_temp": min_temp,
                            "max_temp": max_temp,
                            
                            "wind_speed": wind_speed,
                            "wind_gust": wind_gust,
                            
                            "clouds": clouds,
                            
                            "pop": pop,
                            "rain_desc": rain_desc
                        }
                        break
                    
            elif intent == "humidityForecast": # <-- If intent is humidityForecast
                
                for daily_data in responsejson['daily']: # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        
                        humidity = daily_data['humidity'] # <-- If day matches, set humidity to the humidity value
                        # Create data dictionary with the day, date, and humidity
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "humidity": humidity
                        }
                        break
                    
            elif intent == "pressureForecast": # <-- If intent is pressureForecast
                
                for daily_data in responsejson['daily']: # <-- Iterate over daily forecasts and find matching timestamp to the day parameter
                    if datetime.datetime.utcfromtimestamp(daily_data['dt']).strftime('%Y-%m-%d') == day.strftime('%Y-%m-%d'):
                        
                        pressure = daily_data['pressure'] # <-- If day matches, set pressure to the pressure value ( returns in hPa )
                        # Create data dictionary with the day, date, and pressure
                        data = {
                            "day": datetime.datetime.strftime(day, '%A'),
                            "date": day,
                            "hPa": pressure
                        }
                        break
                    
            # Error handling error if intent is not found        
            else:
                data = "Error"
        except:
            data = "My apologies. I encountered an error when trying to get that data."
            # Return data
        return data
                
    # Function to query weather network API for current data
    # API for endpoint coordinates, we are using Phoenix AZ
    # Forecasting endpoint excludes minutely data and alerts, units set to imperial
    # PARAMETERS: intent
    # SOURCE MODULE: tars.py
    # RETURNS: data from api endpoint
    def queryWeather(self, intent):
        # try:
            # First, we use API endpoint to generate latitude and longitude based on city and state.
            georeq = f'http://api.openweathermap.org/geo/1.0/direct?q=Phoenix,AZ,US&appid={os.getenv("WEATHER_KEY")}'
            georesponse = requests.get(georeq)
            geojson = georesponse.json()[0]
            lat = geojson["lat"]
            lon = geojson["lon"]
            # Next, we pass latitude and longitude into weather API endpoint
            endpoint = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,alerts&units=imperial&appid={os.getenv("WEATHER_KEY")}'
            response = requests.get(endpoint) # Send query to API endpoint
            responsejson = response.json() # Convert to JSON
            
            # Switch case to get the data we need based on the intent.
            if intent == "currentTemperature": # <-- If intent is currentTemperature
                # Set variables to the values in the current forecast: temp, feels_like
                current_temp = responsejson['current']['temp']
                feels_like = responsejson['current']['feels_like']
                # Set data dictionary with variables current_temp and feels_like
                data = {
                    "current_temp": current_temp,
                    "feels_like": feels_like
                }
                
            elif intent == "currentWeather": # <-- If intent is dailySunset
                # Set variables to the values in the current forecast: temp, uvi, wind_speed, wind_deg, clouds
                current_temp = responsejson['current']['temp']
                current_uvi = responsejson['current']['uvi']
                current_windspeed = responsejson['current']['wind_speed']
                current_winddeg = responsejson['current']['wind_deg']
                current_clouds = responsejson['current']['clouds']
                # Set data to dictionary with variables current_temp, current_uvi, current_windspeed, current_winddeg, current_clouds
                data = {
                    "current_temp": current_temp,
                    "current_uvi": current_uvi,
                    "current_windspeed": current_windspeed,
                    "current_winddeg": current_winddeg,
                    "current_clouds": current_clouds
                }
                
            elif intent == "currentDewPoint": # <-- If intent is currentDewPoint
                # Set data to the value in the current forecast: dew_point
                data = responsejson['current']['dew_point']
                
            elif intent == "currentUvi": # <-- If intent is currentUvi
                # Set data to the value in the current forecast: uvi
                data = responsejson['current']['uvi']
                
            elif intent == "currentWind": # <-- If intent is currentWind
                # Set data to dictionary with the values in the current forecast: wind_speed, wind_deg
                data = {                                    
                    "wind_speed": responsejson['current']['wind_speed'],
                    "wind_deg": responsejson['current']['wind_deg']
                }
                
            elif intent == "currentHumidity": # <-- If intent is currentHumidity
                # Set data to the value in the current forecast: humidity
                data = responsejson['current']['humidity']
                
            elif intent == "currentPressure": # <-- If intent is currentPressure
                # Set data to the value in the current forecast: pressure
                data = responsejson['current']['pressure']
            
            elif intent == "currentVisibility": # <-- If intent is curerntVisibility
                # Set data to the value in the current forecast: visibility
                data = responsejson['current']['visibility']
            
            elif intent == "currentRain": # <-- If intent is dailyRain
                data = next((weather['description'] for weather in responsejson['current']['weather'] if weather['main'] == 'Rain'), None)
            
            elif intent == "dailySunset": # <-- If intent is dailySunset
                # Set data to the value in the current forecast: sunset
                data = responsejson['daily'][0]['sunset']
                timestamp = datetime.datetime.fromtimestamp(data) # Convert Unix timestamp to Human-Readable datetime
                data = timestamp.strftime("%H:%M") # Format datetime to hour:minute
                
            elif intent == "dailySunrise": # <-- If intent is dailySunrise
                # Set day0 to the value in the current forecast: sunrise
                day0 = responsejson['daily'][0]['sunrise']
                # Set day1 to the value in the next day's forecast: sunrise
                day1 = responsejson['daily'][1]['sunrise']
                # Convert data to timestamps
                timestamp0 = datetime.datetime.fromtimestamp(day0) # Convert Unix timestamp to Human-Readable datetime
                sunrise_0 = timestamp0.strftime("%H:%M")
                timestamp1 = datetime.datetime.fromtimestamp(day1) # Convert Unix timestamp to Human-Readable datetime
                sunrise_1 = timestamp1.strftime("%H:%M")
                # Set data to dictionary with today and tomorrow's sunrise.
                data = {
                    "today_sunrise": sunrise_0,
                    "tomorrow_sunrise": sunrise_1
                }
                
            
            # FORECASTING: First check if it will rain today, then check for rain in the next 7 days.
            elif intent == "rainWeek":
                days_with_rain = {} # <-- Dictionary to store days with rain
                now = datetime.datetime.now() # <-- Get current time
                rain_times = [] # <-- List to store times it will rain today
                
                # FIRST DAY CHECK
                for forecast in responsejson['hourly']: # <-- Loop through hourly forecast
                    dt = datetime.datetime.fromtimestamp(forecast['dt']) # <-- Convert Unix timestamp to datetime for index
                    if 'rain' in forecast and forecast['rain']['1h'] > 0 and dt.date() == now.date(): # <-- If it will rain today
                        
                        rain_pop = forecast['pop'] # <-- Get probability of precipitation
                        rain_dt = datetime.datetime.fromtimestamp(forecast['dt']) # <-- Convert Unix timestamp to datetime
                        rain_time = rain_dt.strftime("%I:%M %p %Z").strip() # <-- Format datetime to hour:minute AM/PM
                        
                        rain_times.append((rain_time, rain_pop)) # <-- Append time and probability of precipitation to list
                        
                if rain_times: # <-- If list is not empty store rain times in dictionary at index 'Today'
                    days_with_rain['Today'] = rain_times
                    

                # NEXT 7 DAYS FORECAST
                for forecast in responsejson['daily'][1:8]: # <-- Loop through daily forecast
                    dt = datetime.datetime.fromtimestamp(forecast['dt']) # <-- Convert Unix timestamp to datetime for index
                    if 'rain' in forecast and forecast['rain'] > 0: # <-- If it will rain on indexed day
                        
                        days_with_rain[dt.strftime("%A")] = forecast['rain'] # <-- Store rain amount in dictionary at index of day
                # Set data to dictionary with days of rain and times it will rain
                data = days_with_rain
            
            elif intent == "weatherWeek":
                days = {}
                now = datetime.datetime.now()
                  # NEXT 7 DAYS FORECAST
                print(len(responsejson['daily']))
                for forecast in responsejson['daily']: # <-- Loop through daily forecast
                    
                    dt = datetime.datetime.fromtimestamp(forecast['dt']) # <-- Convert Unix timestamp to datetime for index
                    day = dt.strftime("%A")
                    if dt.strftime('%Y-%m-%d') == now.strftime('%Y-%m-%d'):
                        day = "Today"
                    # Compile weather
                    if 'rain' in forecast and forecast['rain'] > 0: # <-- If it will rain on indexed day
                        rain = forecast['rain'] # <-- Store rain amount in dictionary at index of day
                    avg_temp = forecast['temp']['day']
                    min_temp = forecast['temp']['min']
                    max_temp = forecast['temp']['max']
                    uvi = forecast['uvi']
                    
                    wind_speed = forecast['wind_speed']
                    wind_gust = forecast['wind_gust']
                    
                    clouds = forecast['clouds']
                    
                    rain_pop = forecast['pop']
                    rain_desc = "rain"
                    for weather in forecast['weather']:
                        if weather['main'] == "Rain":
                            rain_desc = weather['description']
                        

                    days[day] = {
                        "day": day,
                        "avg_temp": avg_temp,
                        "min_temp": min_temp,
                        "max_temp": max_temp,
                        "uvi": uvi,
                        
                        "wind_speed": wind_speed,
                        "wind_gust": wind_gust,
                        "clouds": clouds,
                        
                        "rain_pop": rain_pop,
                        "rain_desc": rain_desc 
                    }
                # Set data to dictionary with days of rain and times it will rain
                data = days
            elif intent == "tempWeek":
                days = {}
                now = datetime.datetime.now()
                  # NEXT 7 DAYS FORECAST
                print(len(responsejson['daily']))
                for forecast in responsejson['daily']: # <-- Loop through daily forecast
                    
                    dt = datetime.datetime.fromtimestamp(forecast['dt']) # <-- Convert Unix timestamp to datetime for index
                    day = dt.strftime("%A")
                    if dt.strftime('%Y-%m-%d') == now.strftime('%Y-%m-%d'):
                        day = "Today"
                    # Compile weather
                    avg_temp = forecast['temp']['day']
                    min_temp = forecast['temp']['min']
                    max_temp = forecast['temp']['max']
                    days[day] = {
                        "day": day,
                        "avg_temp": avg_temp,
                        "min_temp": min_temp,
                        "max_temp": max_temp
                    }
                # Set data to dictionary with days of rain and times it will rain
                data = days   
                
            elif intent == "humidityWeek":
                days = {}
                now = datetime.datetime.now()
                  # NEXT 7 DAYS FORECAST
                print(len(responsejson['daily']))
                for forecast in responsejson['daily']: # <-- Loop through daily forecast
                    
                    dt = datetime.datetime.fromtimestamp(forecast['dt']) # <-- Convert Unix timestamp to datetime for index
                    day = dt.strftime("%A")
                    if dt.strftime('%Y-%m-%d') == now.strftime('%Y-%m-%d'):
                        day = "Today"
                    # Compile weather
                    humidity = forecast['humidity']
                    days[day] = {
                        "day": day,
                        "humidity": humidity,
                    }
                # Set data to dictionary with days of rain and times it will rain
                data = days  
 

            elif intent == "uviWeek":
                days = {}
                now = datetime.datetime.now()
                  # NEXT 7 DAYS FORECAST
                print(len(responsejson['daily']))
                for forecast in responsejson['daily']: # <-- Loop through daily forecast
                    
                    dt = datetime.datetime.fromtimestamp(forecast['dt']) # <-- Convert Unix timestamp to datetime for index
                    day = dt.strftime("%A")
                    if dt.strftime('%Y-%m-%d') == now.strftime('%Y-%m-%d'):
                        day = "Today"
                    # Compile weather
                    uvi = forecast['uvi']
                    days[day] = {
                        "day": day,
                        "uvi": uvi,
                    }
                # Set data to dictionary with days of rain and times it will rain
                data = days
               
            elif intent == "windWeek":
                days = {}
                now = datetime.datetime.now()
                  # NEXT 7 DAYS FORECAST
                print(len(responsejson['daily']))
                for forecast in responsejson['daily']: # <-- Loop through daily forecast
                    
                    dt = datetime.datetime.fromtimestamp(forecast['dt']) # <-- Convert Unix timestamp to datetime for index
                    day = dt.strftime("%A")
                    if dt.strftime('%Y-%m-%d') == now.strftime('%Y-%m-%d'):
                        day = "Today"
                    # Compile weather
                    wind_speed = forecast['wind_speed']
                    wind_gust = forecast['wind_gust']
                    days[day] = {
                        "day": day,
                        "wind_speed": wind_speed,
                        "wind_gust": wind_gust
                    }
                # Set data to dictionary with days of rain and times it will rain
                data = days          
            # Error handling error if intent is not found
            else:
                data = "Error"
                
            # RETURN data
            return data
        