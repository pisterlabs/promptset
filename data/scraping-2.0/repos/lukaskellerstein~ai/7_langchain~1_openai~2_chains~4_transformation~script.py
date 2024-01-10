from dotenv import load_dotenv, find_dotenv
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

_ = load_dotenv(find_dotenv())  # read local .env file

with open("./interstellar.txt") as f:
    state_of_the_union = f.read()


def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    shortened_text = "\n\n".join(text.split("\n\n")[:3])
    return {"output_text": shortened_text}


# ---------------------------
# Transform Chain
# ---------------------------

transform_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=transform_func
)

# ---------------------------
# LLM Chain
# ---------------------------
template = """Summarize this text:

{output_text}

Summary:"""
prompt = PromptTemplate(input_variables=["output_text"], template=template)
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

# ---------------------------
# Simple sequence
# ---------------------------
sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

result = sequential_chain.run(state_of_the_union)

print(result)
