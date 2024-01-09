import openai
import sys

def neo_eliza():
    messages = [{"role": "system", "content": "You are a cool assistant named Neo-ELIZA, based on the 1966 machine learning experiment, with GPT-4"}]   
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            sys.exit(0)

        messages.append({"role": "user", "content": user_input})
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=2048 
            )
            response_msg = response.choices[0].message.content.strip()
            print("Neo-Eliza: " + response_msg)
            messages.append({"role": "assistant", "content": response_msg})
        except openai.error.OpenAIError as e:
            if 'authentication' in str(e).lower():
                print("Error: Invalid API key.")
                return
            else:
                print(f"An error occurred: {e}")
                return