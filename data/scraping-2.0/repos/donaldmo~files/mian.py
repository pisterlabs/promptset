from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain 
from langchain.prompts import PromptTemplate


load_dotenv()

llm = HuggingFaceHub(
    repo_id='mrm8488/t5-base-finetuned-wikiSQL', 
    # model_kwargs={"temperature": 0.5, "max_length": 64}
)

template = 'Translate English to SQL: {question}'

prompt = PromptTemplate(
    template=template,
    input_variables=["question"]
)

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

question = "What is the average of the respondents using a mobile device?"
print(llm_chain.run(question))