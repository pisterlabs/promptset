import openai
import requests
import time
from config import OPENAI_API_KEY, OPENAI_ENGINE_ID

# Authenticate with OpenAI API
openai.api_key = OPENAI_API_KEY

# Define function to generate random values for each parameter
def generate_redaction_type():
    prompt = "Generate a random redaction type: news, blog, technology, social, ads, legal, medical, academic, code (Answer me only the result without extra text or data use maximum 1 word)"
    response = openai.Completion.create(
        engine=OPENAI_ENGINE_ID,
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.5,
    )    
    print('redaction type')
    time.sleep(10)
    
    return response.choices[0].text.strip()

def generate_language():
    prompt = "Generate a random language: english, spanish, french, german, italian, portuguese, dutch (Answer me only the result without extra text or data use maximum 1 word)"
    response = openai.Completion.create(
        engine=OPENAI_ENGINE_ID,
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.5,
    )
    print('language')
    time.sleep(10)
    return response.choices[0].text.strip()

def generate_audience(language):
    prompt = f"Generate a random audience: latam, us, uk, au, ca, de, es, fr, it, nl, pt... needs to be related to {language}language (Answer me only the result without extra text or data use maximum 1 word)"
    response = openai.Completion.create(
        engine=OPENAI_ENGINE_ID,
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.5,
    )
    print('audience')
    time.sleep(10)
    return response.choices[0].text.strip()

def generate_industry():
    prompt = "Generate a random industry: medical, legal, finance, technology, education, government, marketing, media, retail, travel, hospitality, automotive, real estate, energy, manufacturing, agriculture, construction, transportation, logistics, sports, entertainment, gaming, fashion, beauty, fitness, wellness, food, beverage, alcohol, cannabis, pets, home, garden, family, parenting, dating, relationships, religion, spirituality, astrology, gaming, gambling, adult, other (Answer me only the result without extra text or data use maximum 1 word)"
    response = openai.Completion.create(
        engine=OPENAI_ENGINE_ID,
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.5,
    )
    print('industry')
    time.sleep(10)
    return response.choices[0].text.strip()

def generate_topic(redaction_type, language, audience, industry):
    prompt = f"Generate a topic for a {redaction_type} article in {language} for {audience} audience in the {industry} industry. (use this params as metadata u dont need to write it in the text, just use it as context, Answer me only the result without extra text or data the topic must be 1 sentence max 10 words long)"
    response = openai.Completion.create(
        engine= OPENAI_ENGINE_ID,
        prompt=prompt,
        max_tokens=30,
        n=1,
        stop=None,
        temperature=0.7,
    )
    print('topic')
    time.sleep(10)
    return response.choices[0].text.strip()

# Define variables to store the generated values
for i in range(1):
    redaction_type = generate_redaction_type()
    #language = generate_language()
    language = 'español'
    audience = generate_audience(language)
    industry = generate_industry()
    topic = generate_topic(redaction_type, language, audience, industry)
    # Fill in the API request parameters and JSON data with the generated values 
    url = 'http://127.0.0.1:8000/generate'
    params = {
        'redaction_type': redaction_type,
        'language': language,
        'audience': audience,
        'industry': industry
    }
    json_data = {
        'text': topic
    }

    # Send the API request and print the response
    response = requests.post(url, params=params, json=json_data)
    print(topic + "\n" + str(params) + "\n" + str(json_data) + "\n" + "\n" + "\n")
    print(response.json())
