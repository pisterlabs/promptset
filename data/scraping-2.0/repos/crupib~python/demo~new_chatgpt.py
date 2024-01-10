import openai
openai.api_key = ""
question = None

while(question != "end"):

    print("\n")
    question = input("Ask: ")

    if question != "end":

        res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{question}"}]
        )

        response = res.choices[0].message["content"]

        print(f"{response}\n")
