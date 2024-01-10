# imports
import os
from dotenv import load_dotenv

from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.prompts.prompt import PromptTemplate


# load the .env file variables
load_dotenv()


huggingfacehub_api_token = os.getenv("HUGGING_FACE_API_TOKEN")
repo_id = "tiiuae/falcon-7b-instruct"
from langchain import PromptTemplate, LLMChain

llm = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id=repo_id,
    model_kwargs={"temperature": 0.6, "max_new_tokens": 500},
)


## CHAT:OUT: Hello my name tabib i am here to assist you today please tell what are yor symptomes
def first_answer(query):
    template1 = """
    "you are a ai chat model  talking to a patient  and he   provided you  with his medical  symptomes ,  ask him about his  reported symptoms, and inquire about any additional symptomes. when asking only ask 3 dircet question
    {query}

    Answer:
    """
    prompt1 = PromptTemplate(template=template1, input_variables=["query"])
    llm_chain1 = LLMChain(prompt=prompt1, llm=llm)
    answer1 = llm_chain1.run(query)
    return answer1


## CHAT:OUT: [answer1]


def second_answer(query):
    template2 = """
    "you are a ai model chatting with a preson he provided you  with medical symptoms . ask about those symptoms and how long have he been experinicng them ask only 3 qeustion adn dont provide anymore text
     medical symptoms the person have are 
    {query}

    Answer: 
    """
    prompt2 = PromptTemplate(template=template2, input_variables=["query"])
    llm_chain2 = LLMChain(prompt=prompt2, llm=llm)

    answer2 = llm_chain2.run(query)
    return answer2


## CHAT:OUT: [question3]
question3 = "how have you been felling stressed or  maybe had  late night work putting to much effort"
## CHAT:OUT: [question4]
question4 = "and would you be able to rpovide me with your age and if you have certain allergies of long term disease and"


def fifth_answer(query, answer1, answer2, answer4):
    allquery = f"{answer1} also {answer2} and {answer4}"

    template3 = """
    "Given a set of medical symptoms provided by a patient, your role is to assess whether the patient may require a CT scan,
    X-ray, or bloodwork. Once a medical test is deemed necessary due to the symptoms indicating a potential medical disease, recommend the appropriate test.
    below is the providede set
    {allquery}

    Answer: 
    """
    prompt3 = PromptTemplate(template=template3, input_variables=["allquery"])
    llm_chain3 = LLMChain(prompt=prompt3, llm=llm)

    answer5 = llm_chain3.run(allquery)
    return answer5


## CHAT:OUT: [question5]
