import os, json, time
import openai
# Load your API key from an environment variable or secret management service
openai.api_key = "sk-**********************************************l2"

messages = [
        "Authentication passed",
        "you have successfully logged in",
        "you could log in",
        "login success",
        "Credentials matched",
        "login successful"
]

# models = openai.Model.list()
# print(models)
for i in range(len(messages)):
    for j in range(i+1, len(messages)):
        s = time.perf_counter()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f'"{messages[i]}" and "{messages[j]}" what % similarity in meaning the sentences have'}
            ],
            temperature=0,
            stream=True
        )
        e = time.perf_counter()
        a = list(response)
        result = " ".join([i["choices"][0]["delta"]["content"] for i in a[1:-1]])
        print(json.dumps([messages[i],messages[j],result,f"Response time: {round(e-s,2)} sec"],indent=2))
        # print(json.dumps([messages[i],messages[j],response["choices"][0]["message"]["content"],f"Response time: {round(e-s,2)} sec"],indent=2))
print()