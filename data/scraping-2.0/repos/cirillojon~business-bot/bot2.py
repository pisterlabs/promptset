import openai
import schedule
import time
import itertools
import requests
from bs4 import BeautifulSoup

# Initialize OpenAI's GPT-4 with your API key
openai.api_key = 'sk-'

# Topics for posts
topics = ["sustainable business practices", "leadership", "entrepreneurship", "financial analysis", 
          "global market trends", "innovative business strategies", "corporate responsibility", 
          "technology in business", "business ethics", "diversity and inclusion in the workplace", 
          "business and the environment", "data-driven decision making", "strategic planning", 
          "risk management", "product development", "branding strategies", "customer relationship management", 
          "organizational culture", "business law", "e-commerce trends", "sales strategies", 
          "digital marketing", "business innovation", "negotiation skills", "networking strategies"]
topics_cycle = itertools.cycle(topics)

# Function to search for an article
def search_article(query):
    # Replace spaces in the query with '+' for the Google search URL
    query = query.replace(' ', '+')
    
    # Perform a Google search
    response = requests.get(f'https://www.google.com/search?q={query}')

    # Parse the HTML response
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the first search result and return its URL
    for result in soup.find_all('a'):
        url = result.get('href')
        if 'url?q=' in url and 'google.com' not in url:
            return url.split('url?q=')[1].split('&')[0]
            
    return None

# Function to generate posts
def generate_post():
    profile = """""" 

    # Get next topic from the cycle
    topic = next(topics_cycle)
    
    # Search for an article on the topic
    link = search_article(topic)

    prompt = profile + "\n\nBased on this profile, write a LinkedIn post that they might write about " + topic + ", that could inspire, educate, or inform others in their network."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )

    post = response.choices[0].text.strip()
    # Append the article link to the post
    post += f"\n\nCheck out this article for more information: {link}"
    
    return post


# Function to write message to a text file
def write_message():
    message = generate_post()

    # Write the message to a text file
    with open('linkedin_posts.txt', 'a') as file:
        file.write(message + '\n\n')

# Schedule the task
schedule.every(10).seconds.do(write_message)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)
