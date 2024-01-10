import openai


class FAQAI():
    def __init__(self, api_key):
        openai.api_key = api_key

    def prompt(self, question):
        # CURRENT MODEL: text-davinci-003
        # CURRENT MODEL: babbage
        # FINE TUNED MODEL: davinci:ft-personal-2023-06-20-18-54-27
        # ALT: babbage:ft-personal-2023-06-20-19-31-10
        prompt = f"{question} and this question is about Hello Chef in the UAE"

        print(prompt)
        response = openai.Completion.create(
            model="babbage:ft-personal-2023-06-20-19-31-10", prompt=prompt, max_tokens=128, temperature=0)
        return response['choices'][0]['text']
