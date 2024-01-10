import pytest  
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from promptwatch import register_prompt_template, PromptWatch
from config import OPENAI_API_KEY, PROMPTWATCH_API_KEY

@pytest.fixture
def llmChain():
    TEST_TEMPLATE_NAME = "You are a joke teller. You write very short jokes about {topic} topic. You should write the joke as a reply to {sentence}, so it should be relevant:"

    # we need to ensure that our template is registered !
    registered_prompt_template = register_prompt_template("joke_teller", PromptTemplate.from_template(TEST_TEMPLATE_NAME))
    api_base = "http://127.0.0.1:8000/api"  # 设置你的代理地址  
    llm = OpenAI(temperature=0.0, openai_api_base=api_base, openai_api_key=OPENAI_API_KEY)   
    llmChain = LLMChain(llm=llm, prompt=registered_prompt_template)

    yield llmChain

  
def test_openai_rebase(llmChain):  
    # we need to run our chains in PromptWatch context to capture the logs
    with PromptWatch(api_key=PROMPTWATCH_API_KEY):
        #run few examples    
        print(llmChain(inputs={"topic":"dogs", "sentence":"I'm a dog lover"}))
        print(llmChain(inputs={"topic":"dogs", "sentence":"My dog eats only vegetables"}))
        