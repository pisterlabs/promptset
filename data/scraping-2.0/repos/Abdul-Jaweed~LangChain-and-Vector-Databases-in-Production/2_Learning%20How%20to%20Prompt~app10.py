## Example selectors:

# Example selectors can be used to provide a few-shot learning experience. The primary goal of few-shot learning is to learn a similarity function that maps the similarities between classes in the support and query sets. In this context, an example selector can be designed to choose a set of relevant examples that are representative of the desired output.

# The ExampleSelector is used to select a subset of examples that will be most informative for the language model. This helps in generating a prompt that is more likely to generate a good response. Also, the LengthBasedExampleSelector is useful when you're concerned about the length of the context window. It selects fewer examples for longer queries and more examples for shorter queries.




from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]

example_template = """
Word: {word}
Antonym: {antonym}
"""


example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template
)


example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,
)


dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
    example_separator="\n\n",
)


print(dynamic_prompt.format(input="big"))


