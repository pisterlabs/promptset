from langchain.chat_models import ChatOpenAI as OpenAI
from langchain import PromptTemplate, LLMChain

def generate_story(scenario, openai_api_key):
    """
    Generate a story based on the given scenario using OpenAI's API.
    
    Parameters:
    scenario (str): The context or scenario for the story.
    openai_api_key (str): The OpenAI API key for authorization.
    
    Returns:
    str: The generated story.
    """
    template = """You are a funny story teller;
    you can generate a short funny story of less than 100 words based on a simple narrative;
    CONTEXT : {scenario}
    STORY:"""  
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1, openai_api_key=openai_api_key), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario, verify=False)
    return story
