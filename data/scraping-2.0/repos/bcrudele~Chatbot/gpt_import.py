import config       # Has API & File Path
import openai       # ChatGPT API
import json
# import personalities  # Personalities file, integrate later
# import google_tts    # for testing

# Loads the conversation history from a file
def load_conversation_history(file_path):
    try:
        with open(file_path, 'r') as file:
            conversation_history = json.load(file)
        return conversation_history
    except FileNotFoundError:
        return []

# Saves the conversation history to a file
def save_conversation_history(file_path, conversation_history):
    with open(file_path, 'w') as file:
        json.dump(conversation_history, file, indent=1)
        
# Local active memory list
conversation_history = load_conversation_history("conversation_history.json")
 
def communicate_with_openai(prompt):
    openai.api_key = config.OPENAI_API_KEY 
    
    conversation_history.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages= conversation_history,
        max_tokens = 100
    )

    ai_response = response['choices'][0]['message']['content']

    # Add to memory
    conversation_history.append({"role": "assistant", "content": ai_response})

    # Save json
    save_conversation_history("conversation_history.json", conversation_history)
    
    return ai_response

'''
if __name__ == "__main__":
    iter = 0
    while True:
        prompt = input("Enter a prompt: ")
        result = communicate_with_openai(prompt)
        google_tts.tts(result, iter)
        iter += 1
        print(result)
'''