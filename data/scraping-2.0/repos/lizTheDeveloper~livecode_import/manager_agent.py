from openai import OpenAI

def prompt(prompt):
    print("Manager Agent")
    print(prompt)
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert manager that can help an assistant AI do anything. Please give directions to produce the requested output."},
        {"role": "user", "content": prompt}
    ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content
