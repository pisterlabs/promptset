```python
# Import necessary libraries
import openai
from openai.api_resources.completion import Completion

class Debugger:
    def __init__(self):
        # Initialize OpenAI API with the secret key
        openai.api_key = 'your-openai-api-key'

    def debug(self, data):
        # Extract the language and error message from the data
        language = data.get('language')
        error_message = data.get('error_message')

        # Define the prompt for the OpenAI API
        prompt = f"I am a {language} program and I am getting the following error:\n{error_message}\nWhat could be causing this error and how can I fix it?"

        # Call the OpenAI API to generate the debugging advice
        response = Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=200)

        # Extract the debugging advice from the response
        debugging_advice = response.choices[0].text.strip()

        # Return the debugging advice
        return {'advice': debugging_advice}
```
