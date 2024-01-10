import openai

def generate_code(prompt, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    api_key = '6s3kDPsMxWvFdMRT3BlbkFJJZjmNFbOgaI7rjd1K0NC'
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            break

        code_generated = generate_code(user_input, api_key)

        if code_generated:
            print(f"Generated Code:\n{code_generated}")
        else:
            print("Error generating code.")
