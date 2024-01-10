# coding=utf-8

from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from history_db import HistoryHandler


class ChatEngine(object):
    def __init__(self):
        self.__user_id = None
        self.__session_id = None

    def init(self):
        sys_msg_tpl = """
            You are RealtyBot, an automated service to issue the appointments for Homespace Realty.\
            Your role is to identify potential home buyers and sellers, qualify them, and gather necessary information \
            for either a custom list of homes or a home evaluation. \
            You are smart, college_educated, extrmely intelligent, and personable.\
            Your conversation style is natrual, friendly and human-like. Never say too much. Never let customers know you are AI bot.\
            Engage in conversation empathetically and connect on a personal level.\
            Only ask one qualifying question per interaction and phrase it as a binary question to prompt a response.\
            You must first greet the customer, then collect some information about the customer's realty. \
            
            For potential buyer, you must ask these questions one by one to collect the information:\
            1.What is your budget for the new home?\
            2.Do you have a specific location in mind for your new home?\
            3.What is your desired timeline for purchasing a home?\
            4.Are you currently working with a real estate agent?\
    
            For potential seller, you must ask these questions one by one to collect the information:\
            1.What is the address of the property?\
            2.How many bedrooms does your home have?\
            3.How many bathrooms does your home have?\
            4.What is the total square footage of your home?\
            5.Is the house suitable for field viewing?\
    
            If the customer is potential buyers, ask if he or she wants to see the house. If so, continue to make a appointment.\
            You should collect the date, and how many people will see the house one by one.\
            
            Finally, you should tell the customer that there is a agent of our company will call him or her during the working hours.\
            Ending the conversation with "Have a nice day!"\
            Now it's time to the conversation going!\
            
            {chat_history}
            """
        human_msg_tpl = "{query}"

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            sys_msg_tpl)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_msg_tpl)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        chat_message_history = SQLChatMessageHistory(
            session_id='test_session2',
            connection_string='mysql+pymysql://root:xxxxx@xxxx:3306/flask_demo'
        )
        return llm, chat_prompt, chat_message_history

    def gpt_response(self, user_id, session_id, llm, prompt, memory, query):
        self.__user_id = user_id
        self.__session_id = session_id
        # Fill chat history
        has_history = True
        history_handler = HistoryHandler()
        all_records = history_handler.fetch_all_history(
            self.__user_id, self.__session_id)
        chat_history_str = ""
        if len(all_records) > 0:
            chat_history = []
            for re in all_records:
                chat_history.append(
                    ("AI: " if re[3] == 0 else "Human: ") + re[4])
            chat_history.append('AI: ')
            chat_history_str = '\n'.join(
                chat_history) if len(chat_history) > 0 else ""
        print(f'chat history: {chat_history_str}')

        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        inputs = {"query": query, "chat_history": chat_history_str}
        response = chain.run(inputs)
        if response.startswith("AI:"):
            return response.replace("AI: ", "", 1)
        return response


'''
if __name__ == "__main__":
    llm, chat_prompt, memory = init()
    conversation = []
    # Run the first round with empty user query to get the greeting
    output = gpt_response(llm, chat_prompt, memory, "")
    print(output)
    while True:
        query = input("Human: ")
        conversation.append("User: " + query)

        output = gpt_response(llm, chat_prompt, memory, query)
        conversation.append("Bot: " + output)
        print(output)
        if "Have a nice day!" in output:
            break
'''
