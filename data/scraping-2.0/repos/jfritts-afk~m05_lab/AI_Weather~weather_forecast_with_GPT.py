import requests
from tabulate import tabulate
import os
from datetime import datetime
import openai

# Define your API key here
api_key = 'ee9673ea49f34c0cba902931231909'  # Replace with your actual API key
# Define your OpenAI API key here
openai.api_key = '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'


# Get the current time and date
now = datetime.now()

def clear_screen():
    # Clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()


def generate_weather_description(day, unit, location, time_period):
    # Extract relevant weather data
    max_temp = day['day']['maxtemp_' + unit.lower()]
    condition = day['day']['condition']['text'].lower()
    humidity = day['day']['avghumidity']

    # Extract times of the day
    if time_period == 'morning':
        temp = day['hour'][0]['temp_c'] if unit == 'C' else day['hour'][0]['temp_f']
    elif time_period == 'day':
        temp = day['hour'][6]['temp_c'] if unit == 'C' else day['hour'][6]['temp_f']
    elif time_period == 'evening':
        temp = day['hour'][18]['temp_c'] if unit == 'C' else day['hour'][18]['temp_f']
    else:
        temp = None

    # Compose a prompt for GPT-3
    prompt = f"Generate a weather description for {location} on {day['date']} ({time_period.capitalize()}):\n\n" \
             f"The maximum temperature is {max_temp}°{unit} with {condition} conditions. " \
             f"In the {time_period}, it's {temp}°{unit}."

    # Use GPT-3 to generate the description
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150  # Adjust the max_tokens as needed
    )

    # Extract and return the generated description from GPT-3
    generated_description = response.choices[0].text.strip()
    return generated_description


def get_hour_from_time(time_string):
    # Extract the hour from a time string in the format "yyyy-MM-dd HH:mm AM/PM"
    try:
        parts = time_string.split(' ')
        if len(parts) == 3:
            time = parts[1]
            hour, _ = time.split(':')
            return int(hour)
    except ValueError:
        pass
    return 12  # Default to 12 if parsing fails

def get_time_of_day(hour):
    if 5 <= hour < 10:
        return "Morning"  # Morning (5 AM to 9:59 AM)
    elif 19 <= hour <= 23 or (0 <= hour < 5):
        return "Evening"  # Evening (7 PM to 11:59 PM and 12 AM to 4:59 AM)
    else:
        return "Day"  # Day (10 AM to 6:59 PM)

def calculate_probabilities(forecast_data, unit):
    for day in forecast_data:
        try:
            condition = day['day']['condition']['text'].lower()
        except KeyError:
            condition = "N/A"  # If condition data is not available
        max_temp = day['day']['maxtemp_' + unit.lower()]

        # Initialize probabilities for each time period
        probability_umbrella_morning = "No"
        probability_umbrella_day = "No"
        probability_umbrella_evening = "No"
        
        probability_shorts_morning = "No"
        probability_shorts_day = "No"
        probability_shorts_evening = "No"
        
        probability_jacket_morning = "No"
        probability_jacket_day = "No"
        probability_jacket_evening = "No"

        time_of_day_list = day['hour']

        for time_info in time_of_day_list:
            time = time_info['time']
            hour = get_hour_from_time(time)
            time_of_day = get_time_of_day(hour)
            
            # Determine if you need an umbrella, shorts, or a jacket based on time and weather condition
            if condition in ["rain", "showers"]:
                if time_of_day == "Morning":
                    probability_umbrella_morning = "Yes"
                elif time_of_day == "Day":
                    probability_umbrella_day = "Yes"
                elif time_of_day == "Evening":
                    probability_umbrella_evening = "Yes"
            
            if max_temp >= 70:
                if time_of_day == "Morning":
                    probability_shorts_morning = "Yes"
                elif time_of_day == "Day":
                    probability_shorts_day = "Yes"
                elif time_of_day == "Evening":
                    probability_shorts_evening = "Yes"
            
            if max_temp < 70:
                if time_of_day == "Morning":
                    probability_jacket_morning = "Yes"
                elif time_of_day == "Day":
                    probability_jacket_day = "Yes"
                elif time_of_day == "Evening":
                    probability_jacket_evening = "Yes"

        # Store the calculated probabilities in the day dictionary
        day['probability_umbrella_morning'] = probability_umbrella_morning
        day['probability_umbrella_day'] = probability_umbrella_day
        day['probability_umbrella_evening'] = probability_umbrella_evening
        
        day['probability_shorts_morning'] = probability_shorts_morning
        day['probability_shorts_day'] = probability_shorts_day
        day['probability_shorts_evening'] = probability_shorts_evening
        
        day['probability_jacket_morning'] = probability_jacket_morning
        day['probability_jacket_day'] = probability_jacket_day
        day['probability_jacket_evening'] = probability_jacket_evening

def display_single_day_forecast(day, unit, time_periods, location):
    # Extract data for the current day
    date = day['date']
    max_temp = day['day']['maxtemp_' + unit.lower()]
    min_temp = day['day']['mintemp_' + unit.lower()]
    try:
        condition = day['day']['condition']['text']
    except KeyError:
        condition = "N/A"  # If condition data is not available

    # Generate the weather descriptions
    morning_description = generate_weather_description(day, unit, location, 'morning')
    day_description = generate_weather_description(day, unit, location, 'day')
    evening_description = generate_weather_description(day, unit, location, 'evening')

    # Initialize lists to store relevant time periods
    umbrella_periods = []
    jacket_periods = []
    shorts_periods = []

    # Determine relevant time periods for each probability
    if day['probability_umbrella_morning'] == "Yes":
        umbrella_periods.append("Morning")
    if day['probability_umbrella_day'] == "Yes":
        umbrella_periods.append("Day")
    if day['probability_umbrella_evening'] == "Yes":
        umbrella_periods.append("Evening")

    if day['probability_jacket_morning'] == "Yes":
        jacket_periods.append("Morning")
    if day['probability_jacket_day'] == "Yes":
        jacket_periods.append("Day")
    if day['probability_jacket_evening'] == "Yes":
        jacket_periods.append("Evening")

    if day['probability_shorts_morning'] == "Yes":
        shorts_periods.append("Morning")
    if day['probability_shorts_day'] == "Yes":
        shorts_periods.append("Day")
    if day['probability_shorts_evening'] == "Yes":
        shorts_periods.append("Evening")

    # Format the time periods as strings or display "Not today" if none
    umbrella_text = "\n".join(umbrella_periods) if umbrella_periods else "Not today"
    jacket_text = "\n".join(jacket_periods) if jacket_periods else "Not today"
    shorts_text = "\n".join(shorts_periods) if shorts_periods else "Not today"

    # Create a table for the current day's forecast
    table_headers = ['Date', 'Max Temp (°{})'.format(unit), 'Min Temp (°{})'.format(unit), 'Condition', 'Umbrella', 'Jacket', 'Shorts']
    table_data = [[date, max_temp, min_temp, condition, umbrella_text, jacket_text, shorts_text]]

    # Display the current day's forecast
    clarification_line = "Weather Forecast for {} in {}:".format(date, location)
    print(f"{clarification_line:^{len(table_headers) * 20 + 2}}")
    table = tabulate(table_data, headers=table_headers, tablefmt='pretty')
    print(table)

    # Display additional weather information and the weather descriptions
    humidity = day['day']['avghumidity']

    print("\nAdditional Information:")
    # Check if 'pop' key exists before accessing it
    if 'pop' in day['day']:
        chance_of_rain = day['day']['pop']
        print(f"Chance of Rain: {chance_of_rain}%")
    else:
        print("Chance of Rain: N/A")

    print(f"Humidity: {humidity}%")

    # Check if 'cloud' key exists before accessing it
    if 'cloud' in day['day']:
        cloud_cover = day['day']['cloud']
        print(f"Cloud Cover: {cloud_cover}%")

    # Display the weather descriptions
    print("\nWeather Description (Morning):")
    print(morning_description)
    
    print("\nWeather Description (Day):")
    print(day_description)
    
    print("\nWeather Description (Evening):")
    print(evening_description)


def display_weather_forecast(forecast_data, unit, time_periods, location):
    for day in forecast_data:
        clear_screen()  # Clear the screen before displaying each day's forecast
        display_single_day_forecast(day, unit, time_periods, location)
        input("Press Enter to see the next day's forecast...")


def get_weather_forecast(location, num_days, unit, time_periods):
    # Replace 'YOUR_API_KEY' with your weatherapi.com API key
    api_key = 'ee9673ea49f34c0cba902931231909'
    base_url = 'http://api.weatherapi.com/v1/forecast.json'
    
    # Define the parameters for the API request
    params = {
        'q': location,
        'key': api_key,
        'days': num_days,
        'units': 'metric' if unit == 'C' else 'imperial'  # Use 'metric' for Celsius and 'imperial' for Fahrenheit
    }

    # Send a GET request to weatherapi
    response = requests.get(base_url, params=params)
    data = response.json()

    if 'error' not in data:
        forecast_data = data['forecast']['forecastday']
        calculate_probabilities(forecast_data, unit)
        display_weather_forecast(forecast_data, unit, time_periods, location)
    else:
        print("Error fetching weather data. Please check the location and API key.")

def get_time_periods():
    # Allow the user to customize time periods or use defaults
    time_periods = []
    skip_customization = input("Enter 's' to use defaults or press Enter to customize time periods: ").strip()
    
    if skip_customization.lower() == 's':
        # Use defaults if the user pressed 's'
        return ["Morning", "Day", "Evening"]
    else:
        morning = input("Morning (e.g., 6 AM - 10 AM): ").strip()
        day = input("Day (e.g., 10 AM - 6 PM): ").strip()
        evening = input("Evening (e.g., 6 PM - 10 PM): ").strip()

        if morning.lower() == 's' or day.lower() == 's' or evening.lower() == 's':
            # Use defaults if any input is 's'
            return ["Morning", "Day", "Evening"]
        else:
            return [morning, day, evening]


def display_weather_description(day, time_periods):
    # Generate a text description of the weather
    description = "Weather for {}: ".format(day['date'])
    
    morning_text = "Morning: " + get_description_for_time_period(day, 'morning', time_periods)
    day_text = "Day: " + get_description_for_time_period(day, 'day', time_periods)
    evening_text = "Evening: " + get_description_for_time_period(day, 'evening', time_periods)
    
    description += morning_text + ", " + day_text + ", " + evening_text

    # Display the weather description
    print(description)

def get_description_for_time_period(day, time_period, time_periods):
    description = ""
    
    if time_period in time_periods:
        if day['probability_umbrella_' + time_period] == "Yes":
            description += "Umbrella needed. "
        if day['probability_jacket_' + time_period] == "Yes":
            description += "Jacket needed. "
        if day['probability_shorts_' + time_period] == "Yes":
            description += "Shorts recommended. "
    
    if not description:
        description = "No significant weather."

    return description

if __name__ == "__main__":
    # Clear the screen at the beginning
    clear_screen()

    # Display the ASCII art message
    print(f"""
          .
          |                    
 .                /                
              I      Welcome to your     
              /    Personalized Weather
    \  ,g88R_            Forecast!
      d888(`  ).                   _
-  --==  888(     ).=--           .+(`  )`.
)         Y8P(       '`.          :(   .    )
    .+(`(      .   )     .--  `.  (    ) )
   ((    (..__.:'-'   .=(   )   ` _`  ) )
`.     `(       ) )       (   .  )     (   )  ._
)  )  ( )       --'       `- __.'         :(      ))
.-'  (_.'          .')                    `(    )  ))
              (_  )                     ` __.:'
                                  
--..,___.--,--'`,---..-.--+--.,,-,,..._.--..-._.-a:f--.
The Current Time is: {now.strftime("%H:%M:%S")}
Today's Date is: {now.strftime("%d/%m/%Y")}
--..,___.--,--'`,---..-.--+--.,,-,,..._.--..-._.-a:f--.
    """)

    # Wait for user input before continuing
    input("Press Enter to continue...")
    clear_screen()

    # Take user inputs
    location = input("Enter a location (e.g., city name or ZIP code): ")
    clear_screen()
    num_days = int(input("Enter the number of days for the forecast: "))
    clear_screen()
    unit = input("Enter 'C' for Celsius or 'F' for Fahrenheit: ").upper()
    clear_screen()

    if unit not in ['C', 'F']:
        print("Invalid unit. Please enter 'C' for Celsius or 'F' for Fahrenheit.")
    else:
        # Fetch current weather condition and temperature
        current_weather_url = 'http://api.weatherapi.com/v1/current.json'
        current_weather_params = {
            'q': location,
            'key': api_key,
            'units': 'metric' if unit == 'C' else 'imperial'
        }
        current_weather_response = requests.get(current_weather_url, params=current_weather_params)
        current_weather_data = current_weather_response.json()

        # Extract current weather condition and temperature
        try:
            current_condition = current_weather_data['current']['condition']['text']
            current_temperature = current_weather_data['current']['temp_{}'.format(unit.lower())]
        except KeyError:
            current_condition = "N/A"
            current_temperature = "N/A"

        # Display current time, date, temperature, and weather condition
        clear_screen()
        print(f"The Current Time is: {now.strftime('%H:%M:%S')}")
        print(f"Today's Date is: {now.strftime('%d/%m/%Y')}")
        print(f"Currently the weather is: {current_condition}")
        print(f"The current temperature is: {current_temperature}°{unit}")
        # Wait for user input before continuing
        input("Press Enter to continue...")
        clear_screen()

        # Continue with the rest of your code
        time_periods = get_time_periods()
        get_weather_forecast(location, num_days, unit, time_periods)
