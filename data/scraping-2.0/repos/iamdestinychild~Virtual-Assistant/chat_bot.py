import openai
from dotenv import dotenv_values


openai.api_key = dotenv_values(".env")["API_KEY"]


msg_logs = []

def bot(msg):

    while True:

        if msg == '':
           result = 'did not get that!'
        else:
            msg_logs.append({"role":"user", "content":msg})
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=msg_logs
                )

                result = response["choices"][0]["message"]["content"].strip("\n")
                # .strip("\n").strip()

            except openai.error.APIError as e:
                #Handle API error here, e.g. retry or log
                result = "OpenAI API returned an API Error"
                return result
            
            except openai.error.APIConnectionError as e:
                #Handle connection error here
                result = "Kindly check you internet connection"
                return result
            
            except openai.error.RateLimitError as e:
                #Handle rate limit error (we recommend using exponential backoff)
                result = "OpenAI API request exceeded rate limit"
                return  result
            
        return result
    





    