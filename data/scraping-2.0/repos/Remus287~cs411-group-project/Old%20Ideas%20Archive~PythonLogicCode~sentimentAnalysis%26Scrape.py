import requests
import openai
from keys import GPT_KEY # keys.py holds all api keys, make your own for testing


def get_google_news_and_analyze_sentiment(query, date):
    openai.api_key = GPT_KEY
    # Set up Google News search URL with the specified query and date
    search_url = f"https://www.google.com/search?q={query}&tbm=nws&as_ar=y&tbs=cdr:1,cd_min:{date},cd_max:{date}"
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&sca_esv=575285569&biw=1920&bih=923&sxsrf=AM9HkKlRCWhG1GPoVhdrzzZg-v2FxsTwXg%3A1697833060981&source=lnt&tbs=cdr%3A1%2Ccd_min%3A{date}%2Ccd_max%3A{date}&tbm=nws"

    # Send a GET request to Google News and parse the HTML
    response = requests.get(search_url)
    #print(search_url)
    if response.status_code == 200:
        html_content = response.text
    else:
        print(f"Error: Unable to fetch news. Status code {response.status_code}")
        return

    # Extract and process news headlines and descriptions
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    headlines = soup.find_all("div", class_="n0jPhd ynAwRc MBeuO nDgy9d")
    print(headlines)
    
    #print(news_text)

    """
    # Perform sentiment analysis using GPT-3
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": "You will be provided with text, and your task is to classify its sentiment as positive, neutral, or negative."
        },
        {
          "role": "user",
          "content": news_text
        }
      ],
      temperature=0,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    sentiment = response.choices[0].message["content"]

    return sentiment
    """

# Example usage
query = "Apple stock"  # Replace with your desired query
date = "10/10/2023"  # Replace with your desired date

result = get_google_news_and_analyze_sentiment(query, date)
#print(f"Sentiment analysis result: {result}")
print(result)










