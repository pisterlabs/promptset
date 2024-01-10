import openai

class GPT:
    def __init__(self, apiKey):
        openai.api_key = apiKey

    def getJobSuggestion(self, prompt):
        #to be switched out for completion model but currently does not function well using text-davinci-003
        chatCompletion = openai.ChatCompletion.create(
            model= "gpt-3.5-turbo",
            messages= [{"role": "user", "content": prompt}]
        )
        return chatCompletion.choices[0].message.content #type: ignore

    def talkToBot(self, message):
        chatCompletion = openai.ChatCompletion.create(
            model= "gpt-3.5-turbo",
            messages= [{"role": "user", "content": message}]
        )

        return chatCompletion.choices[0].message.content #type: ignore