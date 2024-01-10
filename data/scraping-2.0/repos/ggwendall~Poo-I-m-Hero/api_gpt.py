import openai

class API_GPT:

    __API_Key = "Votre clé OpenAi"

    @classmethod
    def initialiser(cls, api_key):
        cls.__API_Key = api_key

    @classmethod
    def demande_GPT(cls, query):
        openai.api_key = cls.__API_Key
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "system", "content": query}], max_tokens=1000
        )

        return completion['choices'][0]['message']['content']
