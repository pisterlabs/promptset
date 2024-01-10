import openai
import requests
import json

API_KEY = "***" 
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
FolderPath = "F:\Git\hackyeah3\zawody"


# Write data to the file


def generate_chat_completion(messages, model="gpt-4", temperature=1, max_tokens=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

skillset = """{"Cierpliwosc": "4"
,"Umiejetnosci Komunikacyjne":   "3
,"Ciagle Uczenie sie": "1"
,"Kreatywnosc": "3"
,"Myslenie Analityczne": "2"  
,"Dostosowawczosc":  "5"
,"Zarzadzanie Czasem": "5"
,"Wspolpraca":  "2"
,"Odpornosc": "3"
,"Umiejetnosci Organizacyjne" : "4"
,"Ciekawosc": "2"
,"Swiadomosc Etyczna:": "2"
,"Dokumentacja: "5" }"""

zawod = "prawnik"                

for x in range(20):

    messages = [

            {"role": "user", "content": f"""To jest opis jak dobre muszą być wskazane umiejętności potrzebne dla {zawod}. {skillset} Wygeneruj podobny dla tego zawodu zachowując te umiejętności , ale nie ich wagę którę są wskazane. Zastosuj dla tych wag pewien poziom odchyleń między zestawami , ale 
             niezbyt duży ponieważ to jest ten sam zawód. Nie dodawaj dodatkowych komenatrzy"""}
    ]

    print("\n")
    response_text = generate_chat_completion(messages)
    print(response_text)
    with open(f"{FolderPath}\{zawod}\{zawod}_{x}.json", "w") as f:
         json.dump(response_text, f)



#save response_text to json file with name skill[0]
print(response_text)
print("\n")


