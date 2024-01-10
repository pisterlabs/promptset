from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,    
)
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from PlaySongFromYoutubeAction import PlaySongFromYoutubeAction

from TimerAction import TimerAction

class QueryProcessor:

    def __init__(self, openai_api_key):
        model_name = "gpt-3.5-turbo"
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7, openai_api_key = openai_api_key, verbose=False)
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human."
                ),                
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("Keep your answers short if possible only one to two sentences. Here is the user question: {question}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=memory
        )
        timeraction = TimerAction(self.llm)        
        playyoutube_action = PlaySongFromYoutubeAction(self.llm)
        self.action_list=[timeraction, playyoutube_action]

    def query(self, prompt):
        messages = [
            SystemMessage(content = "You are a helpful assistant and with every request from the human you have to decide if it is an action that you need to execute or just answer the question from your knowledge."),
            HumanMessage(content = prompt)
        ]
        response = self.llm(messages)
        return response.content


    def process(self, query):
        prompt = query+"\nTask: Is the question from the human with respect to one of the actions: "
        options="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        optionsprompt=""
        for i in range(len(self.action_list)):
            prompt+=options[i]+") "+self.action_list[i].selection_prompt+"\n"
            optionsprompt+=options[i]+") "
        general_question_option = options[len(self.action_list)]
        prompt+=general_question_option+") Something else\n"
        optionsprompt+=general_question_option+") "
        prompt+="Answer only with one of the options "+optionsprompt+"\nOption:"
        response = self.query(prompt)
        print("Action selection response: "+str(response))
        if response[0] in optionsprompt:            
            if response[0] == general_question_option:
                return self.get_plain_answer(query)
            else:
                print("Action selected: "+str(response[0])+"  Query is: "+query)
                action = self.action_list[options.index(response[0])]
                response = action.execute(query)
                return response               


    def get_plain_answer(self, query):
        response = self.conversation({"question": query})        
        return response['text']
