################################################################################
### Step 1
################################################################################


import os
import openai
from openai import OpenAI


from ast import literal_eval

def init_api():
     with open(".env") as env:
         for line in env:
             key, value = line.strip().split("=")
             os.environ[key] = value

     
     # TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=os.environ.get("ORG_ID"))'
     # openai.organization = os.environ.get("ORG_ID")
init_api()
client = OpenAI(api_key=os.environ.get("API_KEY"))


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb
from moderation import Moderation



llm_name = "gpt-4-1106-preview"
def load_db(file, chain_type, k):
   
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=10000, 
           chunk_overlap=1500)
    docs1 = text_splitter.split_documents(documents)
    # define embedding
    
    embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("API_KEY"))
    dir = "./docs/"
    with open("./web_text.txt","r") as file:
        text = file.read()
        text = text.replace("\n"," ")
    text = "".join( [x for x in text if x.isalnum() or x == " "] )
    docs2 = text_splitter.split_text(text)
    s = chromadb.config.Settings()
    s.tenant_id = "default"
    texts = []
    metas = []
    for doc in docs1:
        texts.append(doc.page_content)
        metas.append(doc.metadata)
    
    for text in docs2:
        texts.append(text)
        metas.append({"title": "all_texts"})

    vectordb = Chroma(persist_directory=dir, embedding_function=embeddings, client_settings=s)
    vectordb.add_texts(texts = texts, embedding_function=embeddings,metadatas = metas)
    
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

        
    # define retriever
    retriever = vectordb.as_retriever()
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key = os.environ.get("API_KEY")), 
        chain_type=chain_type, 
        retriever=retriever,
        # memory=memory,
    )
    return qa 
class ChatGpt:
    def __init__(self):
        self.state = "AVAILABLE"
        self.qa = load_db("2023Catalog.pdf", "stuff", 4)
        self.moderation = Moderation()
        
    
    def answer_question(self, question,chat_history = []):
        if self.moderation.detect_prompt_injection(question):
            return "Your request has been flagged cannot be processed."
        
        moderation_output = self.moderation.moderation_check(question)
        if moderation_output != '':
            return moderation_output
        prompt = "You had following conversation with the student:\n\n"
        for texts in chat_history:
            if len(texts) == 0:
                continue
            if len(texts) == 1:
                query = texts[0]
                answer = ""
            else:
                query = texts[0]
                answer = texts[1]
            prompt += f"Student: {query}\n"
            prompt += f"AI: {answer}\n\n"
        prompt += f"{question} \n\n Rewrite the query above from context of the chat, Also do not mention the university at all. Abbreiviate Masters in computer science as MSCS. Abbreviate Bachelor of Computer Science as BSCS, Abbreviate masters in business administration to MBA. Don't expand any abbreviations."

        print(prompt)
        res = client.chat.completions.create(temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop="\n",
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ])
        # print(res)
        question = res.choices[0].message.content.strip()
        
        data = {"question": question, "chat_history": []}
        
        print(question)
        answer = self.qa(data)


        # print(answer['answer'])
        return answer['answer']
        
    
    


if __name__ == "__main__":
    chat = ChatGpt()

    print(chat.answer_question(question="What day is it?"))

    print(chat.answer_question(question="What is fastest time i can finish MSCS?"))

    while True:
        print(chat.answer_question(question=input("Question: ")))