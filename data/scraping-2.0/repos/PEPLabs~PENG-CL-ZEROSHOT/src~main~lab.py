import os

from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


"""
The function written at the bottom of the file will use the string defined below to 
form an initial system prompt. Then, user input will be passed into the LLM to be
interpreted. In this case, it is expected for resources/prompt.txt to provide the LLM
with instructions for classifying by sentiment (positive or negative.) This is usually
in the context of categorizing human text such as online posts & product reviews
between positive (eg, happy, excited, 5-star) and negative (eg, angry, disappointed,
1-star) text. This is an elementary problem within AI, and the AI will understand
sentiment classification so long as you provide it the instructions for classifying
"positive" and "negative" inputs in the prompt. Because the AI understands the task
without the need for any shown examples, this is considered a zero-shot problem.
"""

"""
TODO: Change the prompt below to allow for classification as described above.
"""
prompt = ""

"""
There is no need to change the below function. It will properly use the prompt &
user input as needed.
"""


def classify(user_input):

    llm = HuggingFaceEndpoint(
        endpoint_url=os.environ['LLM_ENDPOINT'],
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )
    chat_model = ChatHuggingFace(llm=llm)
    messages = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompt),
        HumanMessagePromptTemplate.from_template("{message}")
    ])

    chain = messages | chat_model
    result = chain.invoke({"message": user_input}).content

    if "positive" in result.lower():
        return "positive"
    elif "negative" in result.lower():
        return "negative"
    else:
        return "user input was not properly classified"
