import openai
import requests
import json

API_KEY = "" 
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
FolderPath = "F:\Git\hackyeah3\osoby"

print("Podaj imię")
imie = input()
#create json file with name imie
data = {}

# Write data to the file
with open(f"{FolderPath}\{imie}.json", "w") as f:
    json.dump(data, f)


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

skillset =     [[1 ,"Cierpliwosc", "testujące cierpliwość"],
                [2 ,"Umiejętnosci Komunikacyjne", "sprawdzające umiejętności komunikacyjne"],
                [3 ,"Ciagle Uczenie sie", "sprawdzające ciągłe uczenie się"],
                [4 ,"Kreatywnosc", "sprawdzające kreatywność"],
                [5 ,"Myslenie Analityczne",  "sprawdzające myślenie analityczne"],
                [6 ,"Dostosowawczosc", "sprawdzające dostosowawczość"],
                [7 ,"Zarzadzanie Czasem", "umiejętność zarządzania czasem"],
                [8 ,"Wspolpraca", "sprawdzające umiejętność współpracy"],
                [9,"Odpornosc", "sprawdzające odporność psychiczną"],
                [10 ,"Umiejętnosci Organizacyjne", "sprawdzające umiejętności organizacyjne"],
                [11 ,"Ciekawosc", "sprawdzające ciekawość"],
                [12 ,"Empatia", "sprawdzające empatie"]
                ]
for skill in skillset[0:1]:
    messages = [

        {"role": "user", "content": f"\n Stwórz pytanie {skill[2]} dla nastolatka'"}
    ]
    print(skill[1])
    print("\n")
    response_text = generate_chat_completion(messages)
    print(response_text)
    odpowiedz = input()
    messages = [
        {"role": "user", "content": "\n Jak w skali 1-5 oceniasz cechę osoby, która na pytanie : " + response_text + " odpowiada: " + odpowiedz + ". Odpowiedz podając jedynie liczbę."}
    ]
    
    print(messages[0]["content"])
    response_text = generate_chat_completion(messages)
    with open(f"{FolderPath}\{imie}.json", "r") as f:
     data = json.load(f)
     new_data = {skill[1]: response_text}
     data.update(new_data)
    with open(f"{FolderPath}\{imie}.json", "w") as f:
     json.dump(data, f)

#save response_text to json file with name skill[0]
    print(response_text)
    print("\n")



