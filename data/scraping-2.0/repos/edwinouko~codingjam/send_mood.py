# this file receives the input from input_mood.js and uses a model to predict the mood of the user then sends it as a json to the server
import openai
import os
import json
import random

openai.api_key = os.getenv("OPENAI_API_KEY")
def main():
    # read mood.txt
    #print filepath
    try:
        f = open("./mood.txt", "r")
    except:
        f = open("../mood.txt", "r")
    mood = f.read()
    f.close()
    if not mood:
        return
    # empty mood.txt
    # f = open("mood.txt", "w")
    # f.write("")
    # f.close()

    mood = mood.replace("\n", "").replace("\r", "").replace("'", "").replace('"', "").replace("\t", "")
    content = "Classify the sentiment of this input in one word as either SAD, HAPPY or NEUTRAL: " + mood
    full_query = "curl https://api.openai.com/v1/chat/completions -H \"Content-Type: application/json\" -H \"Authorization: Bearer " + openai.api_key + "\"  -d '{\
    \"model\": \"gpt-3.5-turbo\", \"messages\": [{\"role\": \"user\", \"content\": \"" + content + "\"}], \"temperature\": 0.7}'"
    # send the query to the openai api
    response = os.popen(full_query).read()
    # convert response json to dictionary
    response = json.loads(response)
    response_text = response["choices"][0]["message"]["content"]
    # API to send get request to the server
    response_api = "https://emotionplushie.azurewebsites.net/api/moodRes?mood=" + response_text.lower()
    response = os.popen("curl " + response_api).read()
    print("response: " + response)
    #write response to response.txt
    f = open("response.txt", "w")
    response = json.loads(response)
    song = response["song"]
    joke = response["joke"]
    quote = response["quote"]
    f.write(random.choice([song, joke, quote]))
    f.close()
if __name__ == "__main__":
    main()