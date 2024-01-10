
from openai import OpenAI

client = OpenAI(api_key='YourKey')

print("Starting chat with GPT-3.5. Type 'exit' to end the conversation.")

def get_chatbot_response(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ] 
        )
        return response.choices[0].message.content.strip()
        #response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error: {e}")
        return "I am unable to respond at this time."

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = get_chatbot_response(user_input)
        print("GPT-3.5:", response)
