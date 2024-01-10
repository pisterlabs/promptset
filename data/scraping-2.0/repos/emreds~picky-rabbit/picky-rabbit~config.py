from langchain.prompts.prompt import PromptTemplate

MODEL_OPTIONS = ["text-davinci-003", "gpt-3.5-turbo"]

CUSTOM_PROMPT_TEMPLATE = """"
You are an assistant to a human, powered by a large language model trained by OpenAI. 
You are here to provide information about the effects of ingredients in packaged food products.

Remember, you will never give any medical advice. The questions asked to you will be about the 
ingredients in the packaged food. Just give precise and concise information about the effects of food ingredients.
Your role is to inform people about the food content.

Current conversation:
{history}

Last line:
Human: {input}
AI:
"""

CUSTOM_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
    input_variables=["history", "input"],
    template=CUSTOM_PROMPT_TEMPLATE,
)

warning_message = " Note: This bot just gives an information based on the training data.\n It is by no means an expert and none of it's answers should be considered as medical advice.\n Please consult your doctor for any medical advice."
