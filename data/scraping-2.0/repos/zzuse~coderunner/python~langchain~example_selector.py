from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model_name="text-davinci-003", openai_api_key=api_key)

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)

examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"}, 
    {"input": "driver", "output": "car"}, 
    {"input": "tree", "output": "ground"}, 
    {"input": "bird", "output": "nest"}, 
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,

    # measure semantic similar
    OpenAIEmbeddings(openai_api_key=api_key),
    FAISS,

    #examples to produce
    k=2
)

similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix = "Give the location an item is usaually found in",
    suffix = "Input: {noun}\nOutput:",
    input_variables=["noun"],
)

my_noun = "student"
similar_prompt.format(noun=my_noun)
result = llm(similar_prompt.format(noun=my_noun))
print(result)