import openai

# In content message we state the bot to how it should work or simply we can say that what the nature of the bot should be.

messages = [
    {"role": "system", "content": "You are a kind helpful assistant."},
]

#A simple loop to make the chat bot the chat bot to keep the countinuity
 
while True:
    message = input("User : ")
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    messages.append({"role": "assistant", "content": reply})
    
    