import os
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

class AIrephraser:

    prompt_more_formal = """Make the following text more formal:
{message}"""
    prompt_shorter = """Make the following text a lot shorter:
{message}"""
    prompt_longer = """Make the following text longer:
{message}"""
    prompt_friendlier = """Make the following text more friendly. Add emojis:
{message}"""


    def __init__(self, openAI_APIKEY):
        self.openAI_APIKEY = openAI_APIKEY
        os.environ['OPENAI_API_KEY'] = openAI_APIKEY

    def rephraseMessage(self, message, prompt):
        
        system_prompt = SystemMessage(content="You will be rephrasing a message provided by user. Try not to loose any important information from the original text when rephrasing.")
        
        if prompt == "more_formal":
            human_message_template = HumanMessagePromptTemplate.from_template(self.prompt_more_formal)
        elif prompt == "longer":
            human_message_template = HumanMessagePromptTemplate.from_template(self.prompt_longer)
        elif prompt == "shorter":
            human_message_template = HumanMessagePromptTemplate.from_template(self.prompt_shorter)
        elif prompt == "friendlier":
            human_message_template = HumanMessagePromptTemplate.from_template(self.prompt_friendlier)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_message_template])

            
        chain = LLMChain(
        llm=ChatOpenAI(temperature="1.0", model_name='gpt-3.5-turbo-16k'),
        #llm=ChatOpenAI(temperature="0", model_name='gpt-4'),
        prompt=chat_prompt,
        verbose=True
        )
        rephrased_message = chain.run({"message": message})

        return rephrased_message
    
    def change_message(self, message, prompt):
        system_prompt = SystemMessage(content=""""Lets think step by step.

You are helping a support agent answer support questions about our platform.
You are providing him with answers that can be copy and pasted to customers.
support agent will tell you how you should change your previous reply.
Come up with an answer he can copy and paste based on how he told you to change your previous reply.

Answer should be formal and short.""")
        
        prompt_change_message = "Your previous reply: " + message + """

    Change your previous answer based on this user prompt: """ + prompt
                    
        prompt_change_message += """
Reply to the support agent's last messages best as you can based on chat history. Only provide a reply to support agent's last message. Provide a message support agent can copy and paste - no explaination or chat history and unneccessary content. Reply should be in the same language as user's message.

Give only the reply support agent can copy and paste. No other text."""
        
        human_message_template = HumanMessagePromptTemplate.from_template(prompt_change_message)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_message_template])

            
        chain = LLMChain(
        llm=ChatOpenAI(temperature="1.0", model_name='gpt-3.5-turbo-16k'),
        #llm=ChatOpenAI(temperature="0", model_name='gpt-4'),
        prompt=chat_prompt,
        verbose=True
        )
        changed_message = chain.run({"message": message})

        return changed_message

#reph = AIrephraser(keys.openAI_APIKEY)
#print(reph.rephraseMessage("We apologize for any inconvenience caused. Losing starred emails after a manual update is not expected behavior. Our team will investigate the issue and try to find a solution. Thank you for bringing this to our attention.", "more_formal"))