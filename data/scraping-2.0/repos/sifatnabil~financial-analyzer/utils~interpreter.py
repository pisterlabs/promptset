from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import dotenv_values

async def interpret(current_summary, previous_summary, prompt):
    """
    Interpret the summary using the prompt

    Args:
        current_summary (str): The current summary
        previous_summary (str): The previous summary
        prompt (str): The prompt to interpret the summary

    Returns:
        str: The interpreted summary
    """

    # Get the api key from the .env file
    config = dotenv_values(".env")
    openai_api_key = config["OPENAI_API_KEY"]

    # Set up the LLM
    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                     model_name="gpt-3.5-turbo",
                     temperature=0)
    
    # Set up the prompt
    prompt_template = prompt

    # Create the template using the prompt and the input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["current_summary", "comparison"])

    # Create the Chain
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    # Get the answer
    answer = await llm_chain.apredict(current_summary=current_summary, comparison=previous_summary)

    # Return the answer
    return answer