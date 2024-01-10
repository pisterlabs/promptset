```python
# Import necessary libraries
import openai
from openai.api_resources.completion import Completion

class CodeGenerator:
    def __init__(self):
        # Initialize OpenAI API with the secret key
        openai.api_key = 'your-openai-api-key'

    def generate(self, data):
        # Extract the language and description from the data
        language = data.get('language')
        description = data.get('description')

        # Define the prompt for the OpenAI API
        prompt = f"Translate the following English text to {language} code:\n{description}"

        # Call the OpenAI API to generate the code
        response = Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100)

        # Extract the generated code from the response
        generated_code = response.choices[0].text.strip()

        # Return the generated code
        return {'code': generated_code}
```
