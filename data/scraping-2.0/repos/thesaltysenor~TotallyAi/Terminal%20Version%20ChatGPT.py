import os
import openai

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY_MAX')

def get_openai_response(prompt_text):
    try:
        response = openai.Completion.create(
          engine="text-davinci-004",
          prompt=prompt_text,
          temperature=0.7,
          max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

def main():
    while True:
        user_input = input("Enter a prompt: ")
        if user_input.lower() == 'exit':
            break
        response = get_openai_response(user_input)
        print(response)

if __name__ == "__main__":
    main()
