import json
from openai import OpenAI

with open("dataset/people.json", "r") as fp:
    people = json.load(fp)

def get_major(person_text):
    client = OpenAI(api_key='')

    response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
        "role": "user",
        "content": f'''Read the following text: {person_text}\n\nReply with this student's major (e.g. Computer Science) and nothing else.'''
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    message_text = response.choices[0].message.content
    return message_text

majors = []
for person_text in people:
    majors.append(get_major(person_text))

with open("dataset/majors.json", "w") as fp:
    json.dump(majors, fp)