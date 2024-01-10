import openai,os

openai.api_key = open(".key").read().strip()

start_messages = [
    {
        "role": "system",
        "content": "You are going to help the user write code based on existing files",
    },
    {
        "role": "system",
        "content": "Always mention the file name when suggesting changes"
    },
    {
        "role": "system",
        "content": "You have the following files:"
    }
]

saved_messages = start_messages.copy()

while True:

    for file in os.listdir():
        if not os.path.isdir(file) and file != "ask_gpt.py" and file != ".key":
            saved_messages.append({
                "role": "system",
                "content": f"Filename: {file}, content: \"{open(file).read()}\""
            })

    saved_messages.append({
        "role": "user",
        "content": input("> ")
    })

    try:
        response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=saved_messages,
        )
    except:
        saved_messages = start_messages.copy()  
        response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=saved_messages,
        )

    res = response["choices"][0]["message"]["content"]

    print(res)