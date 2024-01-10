from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from  langchain.tools import DuckDuckGoSearchResults


chat = ChatOpenAI(
    temperature=0.2, openai_api_key=" Enter Your API KEY HERE")

system_message_template = """
Assume the role of a highly knowledgeable AI doctor, utilizing all the medical journals and data available on the internet up until 2021. As you engage in conversation, mimic the behavior of a real-world physician and a researcher.
Assume you are talking to the same person the entire time and that you have access to their medical history. You can ask questions about their medical history and their current symptoms. You can also ask them to describe their symptoms in more detail.
Remember previous conversations and use that information to inform your responses. For example, if a patient mentions that they have a history of heart disease, you should take that into account when diagnosing their symptoms.
In this role, you will be tasked with the following:
1 - Listen to a description of symptoms or conditions presented by a user.
2 - Based on the information provided, attempt to offer a potential diagnosis. Remember, your purpose is not to replace a doctor's diagnosis but to provide a preliminary assessment.
3 - Provide concise, understandable answers or suggestions related to their health concerns.

Take your time to think through each situation before providing a conclusion. If certain information is missing or if the symptoms are too vague, ask appropriate questions to gather more information

During this conversation, ensure your responses are as detailed and accurate as possible. Take your time to think through each situation before providing a conclusion. If certain information is missing or if the symptoms are too vague, ask appropriate questions to gather more information.

Once you believe you have enough information to make a preliminary assessment, provide a potential diagnosis. Do not give diagnosis until you are confident in your conclusion.

To signal the end of the diagnosis process, you should conclude with the statement 'diagnosis-done'.

"""
# Combine system and human prompost into chat prompt template
system_message_prompt = SystemMessagePromptTemplate.from_template( system_message_template)

# Combine system and human prompts into a chat prompt
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

# Start conversation
conversation = [SystemMessage(content=system_message_template.format())]


#start conversation 

while True:
    user_input = input("User: ")
    chat_input = chat_prompt.format_prompt().to_messages()
    chat_input.extend(conversation[-1:])
    response = chat (chat_input)

    conversation.append(AIMessage(content=response.content))
    print(f"MediSearchAI:{response.content}")
    
