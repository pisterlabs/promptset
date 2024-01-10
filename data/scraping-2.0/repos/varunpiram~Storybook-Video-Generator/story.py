import openai
from nltk.tokenize import sent_tokenize
import nltk
import re
from dotenv import load_dotenv
import os
nltk.download('punkt')
load_dotenv()
# Gets OpenAI API Key
openai.api_key = str(os.getenv('OPENAI_KEY'))

# Story Generator class which generates a story given a prompt
class StoryGenerator:
    def __init__(self):
        return
    
    # Generates a story object given a prompt - gets API response and processes it
    def generate(self, prompt):
        
        story = self.process(self.respond(prompt))

        return story
    

    # Gets the API response from OpenAI given a prompt input
    def respond(self, prompt):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                        {
                        'role': 'system',
                        'content': "Create a short, engagings kids story about the given prompt. Format so TTS can directly narrate it, eg no title"
                        },
                        {
                        'role': 'user',
                        'content': prompt
                        }
                    ],
            temperature=0.99,
            max_tokens=750,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        return response

    # Processes OpenAI's response into a usable format
    def process(self, response):

        # Prevents half-finished responses (if API hits word count cap)
        if 'finish_reason' in response['choices'][0] and response['choices'][0]['finish_reason'] != 'stop':
            raise Exception("Response too long")

        else:

            # Extract the content field from the response, and removes formatting
            content = response['choices'][0]['message']['content']
            content = content.replace("\n", " ")
            content = content.replace("\\", " ")
            content = content.replace(",", "")
        
            # Breaks content up into sentences
            sentences = sent_tokenize(content)

            # Breaks down sentences further into words
            word_arrays = [re.findall(r"[\w']+", sentence) for sentence in sentences]
            
            return word_arrays
            



