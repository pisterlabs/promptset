from __future__ import annotations

from langchain.llms import OpenAI
from langchain.output_parsers import MarkdownListOutputParser
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

examples = [
    {
        "text": "See, I was right! You gotta watch out for these things. They're everywhere @ mindblowing on the road \
        in NJ #sandy",
        "image_caption": "A shark fin is seen rising above the water next to a car.",
        "reason": "The text mentions an event in New Jersey during Hurricane Sandy. The image shows a shark swimming \
        on a road, which is suspicious and possibly doctored. It may not have happened at the mentioned location.",
        "query": "shark photographed swimming on road, New Jersey, Hurricane Sandy",
    },
]

example_template = """
Input:
    - text: {text}
    - image_caption: {image_caption}
Ouput:
    - reason: {reason}
    - query: {query}
"""

example_prompt = PromptTemplate(
    input_variables=["text", "image_caption", "reason", "query"],
    template=example_template,
)

prefix = """Imagine you're an AI tasked with verifying the authenticity of a social media tweet. Your input is the \
tweet's text and image caption. Your job is to extract crucial information from the tweet (like time, location, people,\
and key entities) and use this information to generate a search engine query. This query will help verify the tweet's \
authenticity.
Identify which pieces of information are most likely to contradict known facts. Separate different pieces of key \
information with commas.
Example:"""

suffix = """---

Input:
    - text: {text_input}
    - image_caption: {image_caption}
Output:"""

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["text_input", "image_caption"],
    example_separator="\n\n",
)


def get_qga_chain():
    chain = prompt | OpenAI(temperature=0.7) | MarkdownListOutputParser()
    return chain


if __name__ == "__main__":
    from pyrootutils import setup_root

    root = setup_root(".", dotenv=True)

    print(
        prompt.invoke(
            {
                "text_input": "Hello world!",
                "image_caption": "Hello world!",
            }
        ).text,
    )
    chain = get_qga_chain()
    print(
        chain.invoke(
            {
                "text_input": "NASA has just discovered a new planet in the Andromeda galaxy",
                "image_caption": "Not provided.",
            }
        ),
    )
