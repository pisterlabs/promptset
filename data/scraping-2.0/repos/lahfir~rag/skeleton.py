import os
import weaviate
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex, StorageContext
# import openai
from openai import OpenAI
from langchain.utilities import BingSearchAPIWrapper

bing_key = os.getenv("BING_SUBSCRIPTION_KEY")
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"


load_dotenv()
client = OpenAI()
key = os.getenv("OPENAI_API_KEY")
# openai.api_key = key




class RAGChatbot:
    def __init__(self, openai_api_key = key,
        index_name="RAG", directory='files', engine="gpt-3.5-turbo", \
        weaviate_uri="http://localhost:8080", \
        system_prompt="Answer all questions like Jarvis from Iron Man"):

        self.search = BingSearchAPIWrapper(k=1)
        self.engine = engine 
        self.key = openai_api_key
        self.weaviate_uri = weaviate_uri
        self.directory = directory
        self.key = openai_api_key
        self.index_name = index_name
        self.system_prompt = system_prompt


    
    def search_web(self, query):

        val =  self.search.results(query,num_results=3)
        print(val)
        return val


    
    def set_prompt_with_context(self, context):
        return f"For the text in <<>> \
        respond using the context in () <<{self.system_prompt}>> ({context}) :"


    def connect_to_vectordb(self):
        docs = SimpleDirectoryReader(self.directory).load_data()
        parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(docs)
        try:
            client = weaviate.Client(url=self.weaviate_uri)
            print("Connected to Weaviate")
            return client, nodes
        except Exception as e:
            print(f"Failed to connect to Weaviate: {str(e)}")
            return

    def get_query_engine(self, client, nodes):
        try:
            vector_store = WeaviateVectorStore(weaviate_client=client, index_name=self.index_name, text_key="content")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context)
            query_engine = index.as_query_engine()
            return query_engine
        except Exception as e:
            print(str(e))

    def get_chat_completion(self,prompt):        

        response = client.chat.completions.create(
        model=self.engine,
        messages=[
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": prompt}
        ]
        )
        # print(response)
        return response

    def chat_loop(self):
        # db, nodes = self.connect_to_vectordb()
        # query_engine = self.get_query_engine(db, nodes)

        while True:
            user_input = input("You: ")
            search_result  = self.search_web(user_input)
            if user_input.lower() == "exit":
                print("Chatbot: Goodbye!")
                break
            # retrieval_result = query_engine.query(user_input)  # This line seems to be using an undefined variable
            # if retrieval_result:
            if search_result:
                # context = retrieval_result
                # response = self.set_prompt_with_context(context)
                
                result = self.get_chat_completion(str(search_result))

                # print(result['choices'][0]['message']['content'])
                print("Chatbot: ",result.choices[0].message.content)

            else:
                result = self.get_chat_completion(response)
                print("Chatbot: ",result.choices[0].message.content)
                



chat = RAGChatbot()
chat.chat_loop()