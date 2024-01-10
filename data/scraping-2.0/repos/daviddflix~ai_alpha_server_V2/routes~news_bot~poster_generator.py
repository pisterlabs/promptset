from openai import OpenAI
import requests
import json
import os




from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_KEY')

client = OpenAI(
   
    api_key=OPENAI_KEY,
)

def generate_poster_prompt(article):
    
    prompt=f'Please generate a DALL-E prompt exactly related to this {article}, no more than 1 line longer'
    response = client.chat.completions.create(
            model="gpt-4",
            messages=[ {"role": "system", "content": prompt},
                       {"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1024,
        )
    final_prompt = response.choices[0].message.content
    print(final_prompt)
    generate_article_poster(final_prompt)
    return final_prompt

def generate_article_poster(final_prompt): 
    api_url = 'https://api.openai.com/v1/images/generations'
    
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_KEY}'
    }
    data = {
        "model": "dall-e-3",
        "prompt": f'{final_prompt} - using an anime style',
        "n": 1,
        "size": "1024x1024"
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data))


    if response.status_code == 200:
        result = response.json()
        print("Generated image URL:", result['data'][0]['url'])
    else:
        print("Error:", response.status_code, response.text)
        
        
prompt=""" Anticipation Rises for 2024 Bitcoin ETF Approval
- The likelihood of a spot Bitcoin ETF approval before the deadline of January 10, 2024, is growing, resulting in a surge of Bitcoin's price to as high as $44,000.
- ETF filers, including investment manager Blackrock, are having ongoing discussions with the U.S. Securities and Exchange Commission (SEC).
- Matrixport Research predicts that the Bitcoin price could cross the $50,000 mark in January 2024 if the US SEC approves ETFs. They currently estimate a 95% chance of Bitcoin ETF approval in January 2024.
- Grayscale had a meeting with US SEC officials on December 19, 2023, regarding the potential conversion of the Grayscale Bitcoin Trust (GBTC) into a Bitcoin ETF. The discussion revolved around the proposed rule change to list and trade shares of the Grayscale Bitcoin Trust (BTC).
- Fox Business journalist Charles Gasparino stated that there is optimism among firms that the Commission will approve Bitcoin ETFs after January 8, 2024, potentially with conditions to prevent money laundering-related violations.
Additional Points:
- The SEC has a narrow window between January 8 and 10 to decide on the ETF proposals.
- It is possible that the agency might approve filings from a select few companies initially and later follow up with the remaining firms, or all the ETF filers could be given the same launch date for the ETFs."""
# generate_poster_prompt(prompt)

