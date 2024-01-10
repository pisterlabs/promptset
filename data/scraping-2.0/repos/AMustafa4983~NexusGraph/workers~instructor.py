from openai import OpenAI
import os

class Instructor:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print('Instructor Initialized')
    
    def predict(self, section_text):
        print('Instructor Generating Instructions...')
        prediction = self.client.chat.completions.create(
        model='gpt-3.5-turbo',
        temperature=0.2,
        messages=[
            {'role':'system','content':'''you're an ner system used to extract and form instructions and methodolgies to help researcher do an experiment from a given text and form it in a JSON file.
                                            the output should be in this structure:
                                            {
                                            'instructions' : [
                                            "Obtain...",
                                            "Provide the mice with free...",

                                            ],
                                            'methodolgies' : [
                                            "Conduct a late-on...",
                                            "...",

                                            ]
                                            }'''},
            {'role':'user', 'content':section_text[:4050]},
            {'role':'user', 'content':section_text[4050:]},

        ]
    )    
        return prediction
