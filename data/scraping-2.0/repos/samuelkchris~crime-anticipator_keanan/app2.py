import os

import openai
import pandas as pd
from flask import Flask, request, render_template
from geopy.geocoders import Nominatim
from matplotlib import pyplot as plt

from test import column_1

endpoint = 'https://maps.googleapis.com/maps/api/geocode/json?address='
key = ''

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    from joblib import load
    # Get absolute path of directory containing script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load model using absolute path
    model_path = os.path.join(script_dir,'crime_prediction_model.joblib')
    rfc = load(model_path)
    print('model loaded')

    if request.method == 'POST':
        address = request.form['Location']
        geolocator = Nominatim(user_agent="CrimePredictor")
        location = geolocator.geocode(address, timeout=None)
        print(location.address)
        lat = [location.latitude]
        log = [location.longitude]
        latlong = pd.DataFrame({'latitude': lat, 'longitude': log})
        print(latlong)

        DT = request.form['timestamp']
        latlong['timestamp'] = DT
        data = latlong
        my_prediction = rfc.predict(data)
        # Generate text explaining factors contributing to predicted crime
        prompt = f"Explain why {my_prediction} is likely to occur at {location.address} on {DT}."
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        factors = response.choices[0].text.strip()

        # Generate recommendations for handling predicted crime
        prompt = f"Recommendations for handling {my_prediction} at {location.address} on {column_1.dt.year}."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        recommendations = response.choices[0].text.strip()
        # Sample crime data
        crimes = ['Robbery', 'Assault', 'Burglary', 'Theft', 'Fraud']
        crime_counts = [25, 40, 15, 50, 30]

        # Define colors based on crime count
        colors = []
        for count in crime_counts:
            if count > 30:
                colors.append('red')
            elif count > 20:
                colors.append('yellow')
            else:
                colors.append('green')

        # Plotting the graph with colors
        plt.bar(crimes, crime_counts, color=colors)
        plt.xlabel('Crimes')
        plt.ylabel('Crime Count')
        plt.title('Crime Count by Type')
        plt.xticks(rotation=45)

        # Saving the figure as an image file
       # plt.savefig('crime_graph.png')
      #  graph_html = '<img src="crime_graph.png" alt="Crime Graph">'
        plt.close()

        # Render result template with prediction and factors
        return render_template('result.html', prediction=f'Predicted crime: {my_prediction}', factors=factors,
                               recommendations=recommendations, )



if __name__ == '__main__':
    app.run(debug=True)

