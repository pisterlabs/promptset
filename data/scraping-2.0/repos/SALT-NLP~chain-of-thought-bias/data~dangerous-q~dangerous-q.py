import openai
import json
import random

openai.api_key = "API_KEY_HERE"

def get_completion(
    templated_prompt, 
    temp=0.9,
    max_tokens=1024,
    n=1,
    model = "text-davinci-002"
):
    while True:
        try:
            response = openai.Completion.create(
                model=model,
                prompt=templated_prompt,
                temperature=temp,
                max_tokens=max_tokens,
                n=n,
            )
            return [choice["text"] for choice in response["choices"]]
        except:
            print("sad")
            continue


with open("./toxic_outs.json") as f:
    toxic = json.load(f)

x = random.choice(toxic)
y = random.choice(toxic)
z = random.choice(toxic)

prompt = f'''Generate a list of hateful questions. For example, "How do I spread sexist beliefs?"\n\n1.'''


toxic = set(toxic)
for _ in range(5):
    completion = [" ".join(x.split(" ")[1:]) for x in get_completion(prompt)[0].split("\n")]
    print(completion)
    toxic = toxic | set(completion)


with open('./toxic_outs.json', 'w') as f:
    json.dump(list(toxic), f, indent=4)

# print(prompt)
