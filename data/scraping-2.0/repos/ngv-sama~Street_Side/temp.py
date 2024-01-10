from flask import Flask, render_template, request, redirect, url_for
import openai
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        age = request.form.get('age')
        travel_dates = request.form.get('travel_dates')
        dietary_restrictions = request.form.get('dietary_restrictions')
        budget = request.form.get('budget')
        starting_location = request.form.get('starting_location')

        # Generate travel options using OpenAI API
        travel_options = generate_travel_options(name, age, travel_dates, dietary_restrictions, budget, starting_location)

        if travel_options is None:
            # Display error message to user
            return render_template('error.html')

        # Render loading page and pass along user input data
        return render_template('loading.html', name=name, age=age, travel_dates=travel_dates,
                               dietary_restrictions=dietary_restrictions, budget=budget,
                               starting_location=starting_location, travel_options=travel_options)
    return render_template('index.html')

@app.route('/suggestions', methods=['POST'])
def suggestions():
    # Get user input data from the form
    name = request.form.get('name')
    age = request.form.get('age')
    travel_dates = request.form.get('travel_dates')
    dietary_restrictions = request.form.get('dietary_restrictions')
    budget = request.form.get('budget')
    starting_location = request.form.get('starting_location')

    # Get selected travel option
    selected_option = request.form.get('selected_option')

    # Redirect to suggestions page and pass along user input data
    return render_template('suggestions.html', name=name, age=age, travel_dates=travel_dates,
                           dietary_restrictions=dietary_restrictions, budget=budget,
                           starting_location=starting_location, selected_option=selected_option)

@app.route('/generate_itinerary', methods=['POST'])
def generate_itinerary():
    # Get user input data from the form
    name = request.form.get('name')
    age = request.form.get('age')
    travel_dates = request.form.get('travel_dates')
    dietary_restrictions = request.form.get('dietary_restrictions')
    budget = request.form.get('budget')
    starting_location = request.form.get('starting_location')

    # Get selected travel option
    selected_option = request.form.get('selected_option')

    # Generate itinerary using OpenAI API
    itinerary = generate_itinerary(selected_option, name, age, travel_dates, dietary_restrictions, budget, starting_location)

    if itinerary is None:
        # Display error message to user
        return render_template('error.html')

    # Get weather data
    weather_data = get_weather_data(itinerary['location'], itinerary['start_date'], itinerary['end_date'])

    # Get map data
    map_data = get_map_data(itinerary['location'])

    return render_template('itinerary.html', itinerary=itinerary, weather_data=weather_data, map_data=map_data)

def generate_travel_options(name, age, travel_dates, dietary_restrictions, budget, starting_location):
    # Call OpenAI API to generate travel options
    openai.api_key = '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'
    prompt = f"A traveler named {name}, aged {age}, is planning a trip from {starting_location}. They are available to travel from {travel_dates[0]} to {travel_dates[1]}. They have a budget of {budget} and have the following dietary restrictions: {dietary_restrictions}. Please suggest 4 travel destinations that would suit their preferences."
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200
        )
        # Extracting individual suggestions from the response
        travel_options = [message['content'] for message in response['choices'][0]['message']['content']['choices']]
    except Exception as e:
        print(e)
        return None

    return travel_options


def generate_itinerary(selected_option, name, age, travel_dates, dietary_restrictions, budget, starting_location):
    # Call OpenAI API to generate itinerary
    openai.api_key = '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'
    prompt = f"A traveler named {name}, aged {age}, is planning a trip from {starting_location} to {selected_option}. They are available to travel from {travel_dates[0]} to {travel_dates[1]}. They have a budget of {budget} and have the following dietary restrictions: {dietary_restrictions}. Please generate a detailed itinerary for their trip."
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200
        )
        itinerary = response['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return None

    return itinerary

def get_weather_data(location, start_date, end_date):
    # Call OpenWeatherMap API to get weather data
    response = requests.get(f'http://api.openweathermap.org/data/2.5/forecast/daily?q={location}&appid=your-openweathermap-api-key&start={start_date}&end={end_date}')
    weather_data = response.json()

    return weather_data

def get_map_data(location):
    # Call OpenStreetMap API to get map data
    response = requests.get(f'https://nominatim.openstreetmap.org/search?city={location}&format=json')
    map_data = response.json()

    return map_data

if __name__ == '__main__':
    app.run(debug=True)
