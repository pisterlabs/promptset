import openai

openai.api_key = "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"


def chat(messages):
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    return response.choices[0].message.content.strip()


sys_role = "You are an expert in Applescript, and can parse information and implement in steps what the user asks."
messages = [{"role": "system", "content": sys_role}]
while True:
    user_inp = input("You: ")
    if user_inp in ["bye", "quit", "exit"]:
        break
    if user_inp == "":
        continue
    user_inp += "Write AppleScript code that does the following tasks:"
    messages.append({"role": "user", "content": user_inp})
    response = chat(messages)
    messages.append({"role": "assistant", "content": response})
    ans = response.split("```")[1][len("applescript") :]
    print(ans)
    with open("lol.applescript", "w") as f:
        f.write(ans)
    # print("GPT-4:", response)
    print()
