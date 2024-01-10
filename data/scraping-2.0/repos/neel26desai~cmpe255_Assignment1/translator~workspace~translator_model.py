import openai

class TranslatorModel:
    def __init__(self):
        #openai.api_key = 'your-api-key'
        pass
    #modify the translate_text method to accept a language parameter
    def translate_text(self, text, language='French'):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f'Translate the following text from English to {language}:\n\n"{text}"\n\{language}]: ',
            
        )
        return response['choices'][0].text.strip().replace('"', '')
