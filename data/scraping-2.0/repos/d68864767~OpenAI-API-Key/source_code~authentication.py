```python
import os
from openai import OpenAI

def authenticate():
    # Get the OpenAI API key from environment variables
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

    # Create an OpenAI object with the API key
    openai = OpenAI(api_key)

    return openai

if __name__ == "__main__":
    openai = authenticate()
    print("Successfully authenticated with OpenAI.")
```
