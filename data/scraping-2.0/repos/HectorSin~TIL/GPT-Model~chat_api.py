import openai

openai.api_key = "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"

messages = []
with open('sp500.txt','w',encoding='UTF-8') as f:
    {"role": "system", "content": "You are a helpful assistant."},
    
    while True:
        user_content = input("user : ")

        if user_content == "quit":
            break
        
        messages.append({"role": "user", "content": f"{user_content}"})
        f.write(f"User: {user_content}"+'\n')

        completion = openai.ChatCompletion.create(model = "gpt-3.5-turbo", messages = messages)

        assistant_content = completion.choices[0].message["content"].strip()

        messages.append({"role": "assistant", "content": f"{assistant_content}"})
        f.write(f"GPT: {assistant_content}"+'\n')
        f.write("")
        f.write("--------------------")
        f.write("")

        print(f"GPT: {assistant_content}")