import os
import openai
import pandas as pd

# intents:
df = pd.DataFrame({'OS System': ['My system is mac os',
                                 'mac os', 'windows', 'linux',
                                 'I want to change my system'],
                   'Programming Language': ['I use python',
                                            'Java', 'python', 'C', 'C++', 'Php',
                                            'I want to change the language'],
                   'Function': ['Guiding QA', 'Error message solving'],
                   'Ask Question': ['What is flask?',
                                    'How to use numpy?',
                                    'How to fix CORS problems?',
                                    "Why can't my api response to my website?"],
                   'Welcome': ['Hi', 'start', 'Hello']})

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("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")

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": "Detect the intent of the user."}]
)