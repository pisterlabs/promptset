import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

def generate_topics(base):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="List 3 topics relating to {base}".format(base=base),
        temperature=0.5,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.52,
        presence_penalty=0.5,
        stop=["4."]
    )
    topics_as_string = response.choices[0].text.strip()
    topics_as_list = topics_as_string.split("\n")
    for i in range(len(topics_as_list)):
        topics_as_list[i] = topics_as_list[i][2:].strip()
    return topics_as_list
    
def generate_body(topic):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Write an educational article on {topic} that is at least 500 words long".format(topic=topic),
        temperature=0.7,
        max_tokens=1024,
        top_p=1.0,
        stop="",
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

