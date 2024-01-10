from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

def start(local_name): 
    # location 链
    llm = OpenAI(temperature=1)
    template = """Your job is to come up with a classic dish from the area than the users suggests.
    % USER LOCATION
    {user_location}
    
    YOUR RESPONSE:
    """
    
    prompt_template = PromptTemplate(input_variables=['user_location'], template=template)
    location_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # meal 链
    template = """Given a meal, give a short and simple recipe on how to make that dish at home.
    % MEAL
    {user_meal}
    
    YOUR RESPONSE:
    """
    
    prompt_template = PromptTemplate(input_variables=['user_meal'], template=template)
    meal_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # 通过 SimpleSequentialChain 将两个链串联起来，第一个答案会被替换第二个钟的 user_meal，然后再进行询问
    overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
    review = overall_chain.run(local_name)
    
    
    