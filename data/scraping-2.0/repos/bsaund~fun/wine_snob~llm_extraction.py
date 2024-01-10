import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_wine_from_entry(entry_text):
    prompt = f"""Each examples references a wine. What is the wine?
    Example: The 2017 Chateau de ste Michelle Chardonnay wine was excellent. I had it at a garrison keilor concert in Washington state. I believe I had the wine at the winery. It was about 2013.
    Answer: Chardonnay, 2017, Chateau de ste Michelle
    
    Example: The pinot noir from the San Carlos diner is awful. It is a Chateau de Cochran 1956. Never get this again
    Answer: Pinot Noir, 1956, Chateau de Cochran
    
    Example: Donna has put on a wonderful garden party and is serving a bunch of great wines. I am trying the Chateau Clairac Blaye Cotes de Bordeau, 2018
    Answer: Bordeau, 2018, Chateau Clairac Blaye Cotes
    
    Example: At Donna\'s party, she served a 2017 Chateu Haut-Pezat that I thought was excellent with salmon. I give it 91
    Answer: [Unknown], 2017, Chateau Haut-Pezat
    
    Example: {entry_text}
    Answer: 
    """
    # print(prompt)
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"user", "content": prompt}])
    # print(completion)
    wine = completion["choices"][0]["message"]["content"]
    print(wine)
    return wine


def extract_wine_from_query(entry_text):
    prompt = f"""Each examples references a wine. What is the wine?
    Example: Have I had the 2017 Chateau de ste Michelle Chardonnay before?
    Answer: Chardonnay, 2017, Chateau de ste Michelle

    Example: I'm looking at a pinot noir. It is a Chateau de Cochran 1956. Have I had this?
    Answer: Pinot Noir, 1956, Chateau de Cochran

    Example: I think I've had the Chateau Clairac Blaye Cotes de Bordeau from 2018 before. Tell me about it.
    Answer: Bordeau, 2018, Chateau Clairac Blaye Cotes

    Example: I see this 2017 Chateu Haut-Pezat. Have I rated it?
    Answer: [Unknown], 2017, Chateau Haut-Pezat

    Example: {entry_text}
    Answer: 
    """
    # print(prompt)
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    # print(completion)
    wine = completion["choices"][0]["message"]["content"]
    print(wine)
    return wine


if __name__ == "__main__":
    print("hello")
    extract_wine_from_entry("I hated the Cabernet from Barefoot. 2019")
