import openai
import requests
import json

openai.api_key = 'OPENAI_API_KEY'

def get_eia_data():
    eia_api_key = 'EIA_API_KEY'
    series_id = 'PET.RWTC.D'
    url = f'http://api.eia.gov/series/?api_key={eia_api_key}&series_id={series_id}'
    response = requests.get(url)
    data = json.loads(response.text)
    if data.get('series') is None:
        print("Error retrieving data from EIA API. Check your API key and series ID.")
        return None
    else:
        prices = data.get('series')[0].get('data')
        return prices

def calculate_average_price(prices):
    total_price = 0
    count = 0
    for price in prices:
        total_price += price[1]
        count += 1
    average_price = total_price / count
    return average_price

def ask_gpt(average_price, question):
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "The average price of crude oil is currently at " + str(average_price) + " dollars per barrel."},
            {"role": "user", "content": question}
        ]
    )
    return gpt_response.choices[0].message['content']

def main():
    prices = get_eia_data()
    if prices is None:
        print("Unable to continue due to error retrieving EIA data.")
    else:
        average_price = calculate_average_price(prices)
        question = input("What would you like to ask about the oil market? ")
        gpt_answer = ask_gpt(average_price, question)
        print(gpt_answer)

if __name__ == "__main__":
    main()
