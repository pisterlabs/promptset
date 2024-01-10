from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE ="""{prompt}\n
    CONTEXT: {context}\n
    TEXT: {text}
"""

def inference(api_key: str = "",
        model_name: str = "gpt-3.5-turbo", 
        temperature: int = 0,
        prompt: str = "",
        prompt_variables: dict = {},
        max_tokens: int = 2048):
    """Inference function for langchain

    Args:
        api_key (str, optional): OpenAI API Key. Defaults to "".
        model_name (str, optional): Name of OpenAI Model . Defaults to "gpt-3.5-turbo".
        temperature (int, optional): Controls randomness. Defaults to 0.
        prompt (str, optional): Prompt to be used for inference. Defaults to "".
        prompt_variables (dict, optional): Variable to be inserted into template. Defaults to {}.
        max_tokens (int, optional): _description_. Defaults to 2048.
    """
    
    prompt_template = PromptTemplate(
        input_variables=["prompt","context","text"],
        template=PROMPT_TEMPLATE
    )
    llm = OpenAI(
            temperature = temperature,
            model_name=model_name,
            openai_api_key = api_key
        )

    message = llm(prompt_template.format(
        prompt=prompt,
        context=prompt_variables.get("context",""),
        # instruction=prompt_variables.get("instruction",""),
        text=prompt_variables.get("text","")
    ), max_tokens=max_tokens)
    
    if(message):
        return message
    