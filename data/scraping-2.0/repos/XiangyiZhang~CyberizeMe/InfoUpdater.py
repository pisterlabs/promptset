from langchain.chains import LLMChain, SequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from prompt_templates import thoughts_template, importance_template

class InfoUpdater():
    '''
    The information updater to update thoughts, status, and relationship.
    It is a wrapper function that sequentially calls the specific updater.
    '''
    def __init__(self,llm, agent_name, sender) -> None:
        self.llm = llm
        self.agent_name = agent_name
        self.sender = sender,
        # self.reply = reply

    async def update_thoughts(self, conversation) -> str:
        thought_prompt = PromptTemplate(
            input_variables=["agent_name", "sender", "conversation"],
            template=thoughts_template,
        )
        thought_chain = LLMChain(llm=self.llm, prompt=thought_prompt, output_key="thoughts")

        importance_prompt = PromptTemplate(
            input_variables=["thoughts"],
            template=importance_template
        )
        importance_chain = LLMChain(llm=self.llm, prompt=importance_prompt, output_key="importance")

        overall_chain = SequentialChain(
            chains=[thought_chain, importance_chain],
            input_variables=["agent_name", "sender", "conversation"],
            # Here we return multiple variables
            output_variables=["thoughts", "importance"],
            verbose=True)
        
        updated=overall_chain({"agent_name":self.agent_name, 
                               "sender": self.sender, 
                               "conversation": conversation})
        thoughts = updated['importance']
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 200,
            chunk_overlap  = 20,
            length_function = len,
        )
        docs = text_splitter.create_documents([thoughts])
        with open('config/thoughts.txt', 'a') as file:
            for doc in docs:
                file.write(doc.page_content + '\n') 
        db = Chroma(persist_directory="./chromadb/", embedding_function=OpenAIEmbeddings())
        #print(db._collection.count())
        ids = db.add_documents(docs)
        #print(db._collection.count())
        #db.persist()
        # return f"Updated: thoughts updated for {self.sender}."