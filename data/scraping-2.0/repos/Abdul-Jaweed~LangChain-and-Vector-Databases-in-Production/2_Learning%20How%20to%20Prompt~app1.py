## Few Shot Prompting

# Few Shot Prompting In the next example, the LLM is asked to provide the emotion associated with a given color based on a few examples of color-emotion pairs:

from langchain import PromptTemplate,FewShotPromptTemplate, LLMChain
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    llm=llm,
    model="text-davinci-003",
    temperature=0
)

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"}.
]

example_formatter_template = """
Color: {color}
Emotion: {emotion}\n
"""

example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of colors and the emotions associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    input_variables=["input"],
    example_separator="\n",
)

formatted_prompt = few_shot_prompt.format(input="purple")

chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=formatted_prompt,
                          input_variables=[])
)

response = chain.run([])

print("Color: purple")
print("Emotion: ", response)