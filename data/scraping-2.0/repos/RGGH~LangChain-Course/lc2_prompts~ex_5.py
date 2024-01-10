import openai
from langchain import FewShotPromptTemplate
from langchain import OpenAI
from langchain import PromptTemplate

llm = OpenAI()

# # create our examples
examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }
]

# # create a example template
example_template = """
User: {query}
AI: {answer}
"""

# # create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# # now break our previous prompt into a prefix and suffix
# # the prefix is our instructions
prefix = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 
"""
# and then suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# # now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)


query = "What is the meaning of life?"

print(llm(
    few_shot_prompt_template.format(
        query=query
    )
))

# '''
# To live life to the fullest, laugh often, 
# and never take yourself too seriously.
# '''

# # https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples


