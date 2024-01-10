from gpt import GPT
import openai
from gpt import Example

# Openai key
with open("openai_key.txt") as file:
    key = file.read()
    openai.api_key = key

# GPT Model to create bulletpoints from a topic
gpt_point_creation = GPT(engine="davinci", temperature=.5, max_tokens=120)

gpt_point_creation.add_example(Example("Napoleon III",
                                       "Napoleon III was the first President of France. He founded the Second Empire, reigning until the defeat. He made the French merchant navy the second largest in the world."
                                       ))

gpt_point_creation.add_example(Example("mitochondrion",
                                       "A mitochondrion is a double-membrane-bound organelle. Mitochondria generate most of the cell's supply of adenosine triphosphate. The mitochondrion is often called the powerhouse of the cell."
                                       ))

gpt_point_creation.add_example(Example("blockchain",
                                       "A blockchain is a list of blocks, that are linked together. Blockchains are resistant to modification of their data. The data in any given block cannot be altered once recorded."
                                       ))

gpt_point_creation.add_example(Example("germany",
                                       "Germany is a country in Central Europe. A region named Germania was documented before AD 100 In the 10th century. It covers an area of 357,022 square kilometres. Germany has a population of over 83 million within its 16 constituent states."
                                       ))


# Create Text (Bulletpoints) from a topic
def create_text_from_topic(prompt):
    output = gpt_point_creation.submit_request(prompt)
    text_output = output.choices[0].text[8:]
    text_output = text_output.strip()
    print("GPT-3 generated Text:\n" + text_output)
    return text_output
