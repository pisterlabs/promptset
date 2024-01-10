import openai

def chatbot(prompt, api_key):
    try:
        completions = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
            api_key=api_key
        )
        message = completions.choices[0].text
        return message
    except Exception as e:
        print("Error in chatbot:", e)
        return None

print("Hi there! I'm Z4GR3US, your virtual chatbot. How can I help you today?")

while True:
    user_input = input("\nUser: ")
    if user_input == "exit":
        break
    
    if "who are you" in user_input.lower():
        response = "I am Z4GR3US, your virtual chatbot."
    else:
        response = chatbot(user_input, "YOUR-API")
        
    if response is not None:
        print("\nChatbot:", response)
    else:
        print("Error in chatbot")
