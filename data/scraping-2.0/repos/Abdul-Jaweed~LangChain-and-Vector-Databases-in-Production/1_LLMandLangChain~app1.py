# Few-shot learning

# Few-shot learning is a remarkable ability that allows LLMs to learn and generalize from limited examples. Prompts serve as the input to these models and play a crucial role in achieving this feature. With LangChain, examples can be hard-coded, but dynamically selecting them often proves more powerful, enabling LLMs to adapt and tackle tasks with minimal training data swiftly.

# This approach involves using the FewShotPromptTemplate class, which takes in a PromptTemplate and a list of a few shot examples. The class formats the prompt template with a few shot examples, which helps the language model generate a better response. We can streamline this process by utilizing LangChain's FewShotPromptTemplate to structure the approach:

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate

# create our example 

examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, 
    {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
] 

# Create an example template

example_template = """sumary_line
User: {query}
AI: {answer}
"""

# Create a prompt example from above template 

example_prompt = PromptTemplate(
    input_variables=['query', 'answer'],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions

prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""

# and the suffix our user input and output indicator

suffix = """
User: {query}
AI: """

# now create the few-shot prompt template

few_short_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_selector="\n\n"
)

# After creating a template, we pass the example and user query, and we get the results

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

import os
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

# load the model

chat = ChatOpenAI(
    openai_api_key=apikey,
    model="gpt-4",
    temperature=0.0
)

chain = LLMChain(
    llm=chat,
    prompt=few_short_prompt_template
)

chain.run("What's the meaning of life?")

