# conda create -n chatbot python=3.11
# conda activate chatbot
from openai import OpenAI

client = OpenAI()

def chatbot(prompt):
    response = client.chat.completions.create(model = "gpt-3.5-turbo", 
    messages = [{'role': 'user', 'content': prompt}])
    print(response)
    return response.choices[0].message.content
    
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Bot: ", chatbot(user_input))