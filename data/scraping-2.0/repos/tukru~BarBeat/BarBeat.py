import requests
import json
import os
import statistics
import openai

def search_bars(api_key, location):
    try:
        url = "https://api.yelp.com/v3/businesses/search"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        params = {
            "term": "bars",
            "location": location
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from Yelp: {e}")
        return None

def analyze_market_competition(bars):
    low_rated_bars = [bar for bar in bars if bar['rating'] <= 3.5]
    suggestions = """
    1. Make sure that your bar is well-rated on popular rating sites.
    2. Offer competitive pricing for your drinks and food.
    3. Make sure that your bar is clean and inviting.
    4. Offer a variety of drinks and food items.
    5. Make sure that your bar is located in a convenient location.
    """
    analysis_prompt = f"There are {len(low_rated_bars)} bars with a rating of 3.5 or below in the area. Analyze the market competition and offer suggestions for upgrades to these bars."
    try:
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=analysis_prompt,
            max_tokens=2024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        analysis_text = completion.choices[0].text + suggestions
        return analysis_text
    except openai.error.OpenAIError as e:
        print(f"Error fetching data from OpenAI: {e}")
        return None

def save_bar_details(bars, analysis_text):
    if not os.path.exists('html'):
        os.makedirs('html')
    sorted_bars = sorted(bars, key=lambda x: x['rating'], reverse=True)
    html_content = "<html><head><title>BarBeat - Bars in Area</title></head><body><h1>Bars in Area</h1><table border='1'>"
    html_content += "<tr><th>Name</th><th>Rating</th><th>Address</th><th>Phone</th><th>Price</th><th>Categories</th></tr>"

    for bar in sorted_bars:
        html_content += f"<tr><td>{bar['name']}</td><td>{bar['rating']}</td><td>{bar['address']}</td><td>{bar['phone']}</td><td>{bar['price']}</td><td>{', '.join(bar['categories'])}</td></tr>"

    html_content += "</table><h2>Analysis</h2>" + analysis_text + "</body></html>"

    with open('html/bars.html', 'w') as file:
        file.write(html_content)

openai.api_key = "ADD YOUR OPENAI KEY"
api_key = "ADD YOUR YELP API KEY (FUSION)"

# Prompt for location to search for bars
location = input("Enter a zip code or address to search for bars: ")

# Search for bars
bars_data = search_bars(api_key, location)

if bars_data:
    # Extract relevant information
    bars = []
    for bar in bars_data['businesses']:
        bar_info = {
            'name': bar['name'],
            'rating': bar['rating'],
            'address': bar['location']['address1'],
            'phone': bar.get('phone', ''),
            'price': bar.get('price', ''),
            'categories': [category['title'] for category in bar['categories']]
        }
        bars.append(bar_info)

    # Analyze the market competition
    analysis_text = analyze_market_competition(bars)

    if analysis_text:
        # Save bar details
        save_bar_details(bars, analysis_text)
        print("Analysis completed and saved to html/bars.html.")
    else:
        print("Failed to analyze the market competition.")
else:
    print("Failed to fetch bar data.")
