import openai
import json
openai.api_key="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"

messages = [
    {"role": "system", "content": "You are a kind helpful assistant."},
]
def GetSuggestions(weather = "sunny",location = "Pune"):
        jsonFormat = "{\"PlacesToVisit\":[],\"Eateries\":[],\"Recipes\":[]\}."
        prompt = f"The weather in {location} shows {weather}. Based on this weather type and location information, return a list of places to visit, eateries to dineout & recipes to cook in following json format - {jsonFormat}.  Give the response only in the provided json format and do not add any extra words or sentences in your response."
        #weather = input("What is the weather outside : ")
        #location = input("What is your current city : ")
        message = f"{prompt} {weather} {location}"
        if message:
            messages.append(
                {"role": "user", "content": message},
            )
        chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
        
        reply = chat.choices[0].message.content

        jsonReply = json.loads(reply)
        placesToVisit, placesToEat, recepiesToCook = '', '', ''
        for place in jsonReply['PlacesToVisit']:
            placesToVisit += f"-{place}\n"
        for restaurant in jsonReply['Eateries']:
            placesToEat += f"-{restaurant}\n"
        for recepie in jsonReply['Recipes']:
            recepiesToCook += f"-{recepie}\n"

        return jsonReply, placesToVisit, placesToEat, recepiesToCook
        #messages.append({"role": "assistant", "content": reply})