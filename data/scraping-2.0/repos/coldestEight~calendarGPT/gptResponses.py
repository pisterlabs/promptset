from os import environ as env
import openai
import dotenv

dotenv.load_dotenv()
openai.api_key = env["OPENAI_API_KEY"]

#Makes call to GPT API and formats response
def getMsgResponse(messageLog):
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", messages = messageLog, temperature= 1.0,max_tokens=1000)
    rawOut = response.choices[0].message.content

    punctuationList = [".","?","!"]
    cutoffPos = 0

    #Check for last punctuation point and split the string at that point
    #Prevents incomplete sentences from being sent back to AI
    for i in range(len(rawOut)):
        if rawOut[i] in punctuationList:
            cutoffPos = i

    return rawOut[0:cutoffPos+1]

