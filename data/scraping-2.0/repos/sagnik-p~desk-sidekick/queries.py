import os
import openai

file_path_apikey = "S:/key.txt"
if os.path.isfile(file_path_apikey):
    text_file = open(file_path_apikey, "r")
    data = text_file.read()
    text_file.close()
else:
    data = input("File not found, Enter API key")
openai.api_key= data
def getGPTAnswer(query,type="short"):
    if(type == "explain" or type == "long"):
        query = " in detail, tell me " + query
    elif(type == "short"):
        query = " in a few words, tell me " + query
    print(query)
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": query}],stream = True)
    print(str(chat))
    return chat.choices[0].message.content
def getDaVinviAnswer(prompt,type = "short"):
    if (type == "explain"):
        query = " in simple words, explain in details " + prompt
    elif (type == "short"):
        query = " in short, explain " + prompt
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=4000
    )
    response_text = str(response['choices'][0]['text']).strip('\n\n')
    return response_text