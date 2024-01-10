import requests
import json
import openai
from statistics import mean

# Ensure you have set the OPENAI_API_KEY in your environment variables
openai.api_key = 'OPENAI_API_KEY'

def get_eia_data(start_year, end_year):
    api_key = "EIA_API_KEY"
    base_url = "https://api.eia.gov/v2/petroleum/pri/dfp1/data/"
    url = f"{base_url}?api_key={api_key}&frequency=monthly&data[0]=value&start={start_year}-01&end={end_year}-12&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"

    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None


def calculate_average_price(data):
    # Extract the 'value' field from each data point and ignore data points without a value
    prices = [float(item['value']) for item in data['response']['data'] if 'value' in item and item['value'] is not None]

    # Return the average price
    return mean(prices)

def ask_gpt(question, average_price):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that knows a lot about oil prices."
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": f"The average oil price was ${average_price:.2f}"
            }
        ]
    )

    # Extract the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']
    
    return assistant_message

# Retrieve the data from the EIA API
data = get_eia_data(1980, 1989)  # The 80's
if data is not None:
    # Calculate the average price
    average_price = calculate_average_price(data)

    # Feed the average price to GPT-4
    response = ask_gpt("What was the average oil price in the 80's?", average_price)
    print(response)
else:
    print("Failed to fetch data")
