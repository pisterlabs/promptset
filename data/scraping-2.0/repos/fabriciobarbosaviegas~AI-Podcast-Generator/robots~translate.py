import openai



def translate(text):
    searchTerms = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": text},
            {"role": "user", "content": "Traduza esse texto para o português brasileiro mantendo fidelidade total ao seu tema"},
        ]
    )
    response = searchTerms.choices[0].message["content"]

    return response

def parseScript(script, male, female):
    script = script.split(":")
    gender = script[0].lower()
    print(gender)
    if gender.find("masculino") >= 0 or gender.find("male") >= 0 or gender.find("homem") >= 0:
        print("masculino")
        return {"presenter":male, "text":f"{script[1]}..."}
    else:
        print("feminino")
        return {"presenter":female, "text":f"{script[1]}..."}
