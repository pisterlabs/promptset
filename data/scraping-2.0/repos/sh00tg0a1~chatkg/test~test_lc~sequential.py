import openai
import json
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ENDPOINT  = os.environ.get("OPENAI_ENDPOINT")
OPENAI_API_VERSION  = os.environ.get("OPENAI_API_VERSION")
OPENAI_API_TYPE  = os.environ.get("OPENAI_API_TYPE")

openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_ENDPOINT
openai.api_version = OPENAI_API_VERSION
openai.api_key = OPENAI_API_KEY

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain, SequentialChain

llm = OpenAI(
    # engine = "asada001",
    # engine="aschatgpt35",
    model_kwargs= {
        "engine": "asdavinci003",
    },
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_ENDPOINT,
    max_tokens=256,
)

# This is an LLMChain to write a synopsis given a title of a play.

template = """You are a playwriter. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

# This is an LLMChain to write a review of a play given a synopsis.
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")


# overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)
# review = overall_chain.run("Tragedy at sunset on the beach")

overall_chain = SequentialChain(chains=[synopsis_chain, review_chain], input_variables=["title"], output_variables=["synopsis", "review"], verbose=True)
review = overall_chain({"title": "Tragedy at sunset on the beach"})
print(json.dumps(review))
