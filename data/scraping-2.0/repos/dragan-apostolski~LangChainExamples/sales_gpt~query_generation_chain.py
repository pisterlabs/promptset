from langchain import LLMChain, PromptTemplate, FewShotPromptTemplate
from langchain.llms import BaseLLM

KEYWORD_GENERATION_PROMPT_TEMPLATE = """
You are an assistant that generates a query for a vector database, based on the conversation history.
As a context, you are given a conversation between a salesperson and a customer, 
where the customer is interested in buying a camera lens. 
The salesperson is trying to help the customer find the right lens for their needs.
In the vector database we have a list of camera lenses, and their descriptions. 
Based on the query that you generate, we will fetch the relevant lenses from the vector database,
and return them to a salesperson, who will then recommend the most relevant lens to the customer.

Your job is to generate a query for the vector database, by understanding the essence of the conversation history,
and what is the customer looking for.
After that, generate a query with the keywords that will maximise the probability of the vector database returning the 
most relevant lenses.
Please output only the keywords, don't generate anything else in the answer. 
You should output up to 5 keywords, separated by commas. 
"""

EXAMPLE_PROMPT_TEMPLATE = """
Conversation: {conversation}
Keywords: {keywords}
"""

EXAMPLES = [
    {
        "conversation": "Customer: I am looking for a lens for my camera. "
                        "Salesperson: What type of photography do you do? "
                        "Customer: I do landscape photography, and I also do night sky photography."
                        "Salesperson: Do you shoot wide scenes, or close ups?"
                        "Customer: I shoot wide scenes.",
        "keywords": "landscape, night sky, wide scenes",
    },
    {
        "conversation": "Customer: I am looking to buy a lens for my camera."
                        "Salesperson: Wat type of photography do you do?"
                        "Customer: I have a Canon EOS 5D Mark IV. I do portrait photography. "
                        "Salesperson: What type of portraits do you do, and where do you shoot?"
                        "Customer: I do environmental portraits, and I shoot in the city.",
        "keywords": "environmental portraits, city",
    }
]


class KeywordGenerationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:

        example_prompt = PromptTemplate(
            input_variables=["conversation", "keywords"],
            template=EXAMPLE_PROMPT_TEMPLATE,
        )

        prompt_template = FewShotPromptTemplate(
            examples=EXAMPLES,
            example_prompt=example_prompt,
            prefix=KEYWORD_GENERATION_PROMPT_TEMPLATE,
            suffix="Conversation: {conversation}\nKeywords: \n",
            input_variables=["conversation"],
        )
        return cls(prompt=prompt_template, llm=llm, verbose=verbose)
