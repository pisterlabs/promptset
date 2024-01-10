from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# Full Sequence Chain
#
# MULTIPLE INPUTS/OUTPUTS !!!
#
# A more general form of sequential chains,
# allowing for **multiple inputs/outputs**.
#
# OPEN AI API - POST https://api.openai.com/v1/completions
# ---------------------------


# ---------------------------
# Chain 1
# ---------------------------

# This is an LLMChain to write a synopsis given a title of a play and the era it is set in.
llm = OpenAI(temperature=0.7)
template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")


# ---------------------------
# Chain 2
# ---------------------------

# This is an LLMChain to write a review of a play given a synopsis.
llm = OpenAI(temperature=0.7)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")


# ---------------------------
# Sequential Chain
# ---------------------------

# This is the overall chain where we run these two chains in sequence.
overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True,
)


overall_chain({"title": "Tragedy at sunset on the beach", "era": "Victorian England"})
