import openai
import os

def check_env_var(key_var):
    api_key = os.getenv(key_var)
#    if api_key is not None:
#        get_api_key(api_key)
#        update_key()
#        update_api_status("API Status: Ready")
#    else:
#        get_api_key("null")
#        update_api_status("API Status: No Key")
        
check_env_var("OPENAI_API_KEY")

def chat_with_openai(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        n=1,
        stop=None,
        timeout=15
    )
    
    return response.choices[0].text.strip()

# Start the conversation
print("Welcome! Let's chat. Type 'quit' to exit.")

while True:
    user_input = input("User: ")
    
    if user_input.lower() == 'quit':
        break
    
    # Append user input as the prompt
    prompt = f'User: {user_input}\nAI(davinci):'
    
    # Get response from OpenAI
    ai_response = chat_with_openai(prompt)
    
    print("AI(davinci):", ai_response)