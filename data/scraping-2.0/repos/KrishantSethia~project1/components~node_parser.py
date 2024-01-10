from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.node_parser import SimpleNodeParser
import os
os.getenv("OPENAI_API_KEY")


class NodeParsers():

    def node_parser_1(documents):
        llm = OpenAI(model='gpt-4')
        service_context = ServiceContext.from_defaults(chunk_size=256, llm=llm)
        nodes = service_context.node_parser.get_nodes_from_documents(documents)
        return nodes

    def node_parser_2(documents):
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        return nodes


"""
#Langchain Method
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
splits = text_splitter.split_documents(data)
"""
