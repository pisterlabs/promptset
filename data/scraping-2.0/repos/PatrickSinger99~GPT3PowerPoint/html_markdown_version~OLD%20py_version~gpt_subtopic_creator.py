from gpt import GPT
import openai
from gpt import Example

# Openai key
with open("openai_key.txt") as file:
    key = file.read()
    openai.api_key = key

# GPT Model to create subtopics from a topic
gpt_subtopic_creation = GPT(engine="davinci", temperature=.5, max_tokens=120)

gpt_subtopic_creation.add_example(Example("car",
                                       "motor, vehicle, transportation, road, wheels"
                                       ))

gpt_subtopic_creation.add_example(Example("tiger",
                                       "cat, panthera, predator, wildlife, endangered"
                                       ))

gpt_subtopic_creation.add_example(Example("blockchain",
                                       "crypto, cryptography, transaction, bitcoin, cryptocurrency"
                                       ))

gpt_subtopic_creation.add_example(Example("germany",
                                       "country, europe, berlin, european union, demogracy"
                                       ))


# Create Text (Bulletpoints) from a topic
def create_subtopics_from_topic(prompt):
    output = gpt_subtopic_creation.submit_request(prompt)
    text_output = output.choices[0].text[8:]
    return text_output
