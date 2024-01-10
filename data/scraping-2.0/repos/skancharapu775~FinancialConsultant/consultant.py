import openai
import keys

'''
Financial Consultant 
- Take inputs of financial advice
- Create outlets using specific prompts
'''

# Avoid large responses
MAX_TOKENS = 400
TEMPERATURE = 1

# Cheaper model and most powerful
MODEL = "gpt-3.5-turbo"

openai.api_key = keys.API_KEY

def generate_fresponse(prompt):
    
    # Give context for response
    prompt = prompt + "-> Answer as if you are a financial consultant."

    # Create model request with custom parameters
    response = openai.ChatCompletion.create(
            model=MODEL,
            temperature=1,
            max_tokens=MAX_TOKENS,
            messages = [{"role":"user", "content": prompt}]
        )

    # Get text from json response and return
    f_response = response["choices"][0]["message"]["content"]
    return f_response

