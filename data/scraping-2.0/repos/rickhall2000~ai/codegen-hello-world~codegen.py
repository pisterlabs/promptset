import openai
import os

api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key


def generate_code(prompt, max_tokens=150):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates Python code."},
        {"role": "user", "content": prompt}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Assuming this is the identifier for GPT-4
        messages=messages,
        max_tokens=max_tokens
    )
    
    # Extracting the assistant's message from the response
    return response.choices[0].message['content'].strip()

if __name__ == "__main__":
    code_prompt = input("Enter a code description: ")
    generated_code = generate_code(code_prompt)
    print("Generated code:\n", generated_code)
    