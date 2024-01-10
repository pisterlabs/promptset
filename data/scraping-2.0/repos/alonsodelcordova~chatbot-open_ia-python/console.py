import openai

openai.api_key = "{your_key}"

messages = []
system_msg = input("¿Qué tipo de chat bot te gustaría crear?\n")
messages.append({"role": "system", "content": system_msg})

print("¡Tu nuevo asistente virtual está listo! \nIngrese su mensaje o escriba quit() para salir.")
while input != "quit()":
    message = input()
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")
    print("Ingrese su mensaje o escriba quit() para salir.")