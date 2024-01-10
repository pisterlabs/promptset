import openai


openai.api_key = key

model = "gpt-3.5-turbo"


temp = 0.5
messages = []
while True: 
    mes = input("\nEnter input: ")

    if "hist" in mes and len(mes) <= 10:
        print(messages)
    elif "temp" in mes and len(mes) <= 10:
        new = float(input("Enter int for temp: "))
        temp = new
    else: 
        messages.append({"role" : "user", "content" : mes})
        response = openai.ChatCompletion.create(
                model = model,
                messages = messages,
                temperature = temp
        )

        textOut = response.choices[0].message.content
        messages.append({"role" : "assistant", "content" : textOut})
        print()
        print(textOut)
