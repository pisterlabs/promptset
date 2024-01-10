from abc import ABC, abstractmethod

from pa.constants import retriever_tempate, resolver_template
from pa.llm.base import LLM
from pa.constants import PERSIST_DIRECTORY

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma


class Agent(ABC):
    def __init__(self, llm: LLM, name: str='agent') -> None:
        self.llm = llm
        self.name = name
        
    @abstractmethod    
    def generate(self) -> str:
        pass

    
class RetrieverAgent(Agent):
    def __init__(self, llm: LLM, name: str='agent') -> None:
        super().__init__(llm, name)
        
        print(f'Initializing the retriever agent...')
        print(f'Loading the embedding model...')
        self.embedding_instruction = "Represent the question for retrieving supporting documents "
        self.instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                              model_kwargs={"device": "cuda"},
                                                              query_instruction=self.embedding_instruction)
        print(f'Done')

        print(f'Loading the vector database retriever client...')
        self.vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=self.instructor_embeddings)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})
        print(f'Done\n')
        
        self.context_docs = []
        self.context_proc = ""
        self.context_included = []
        
    def generate(self, question: str, config: dict={}) -> str:
        # Query vectorDB and get relevant documents
        self.context_docs = [doc for doc in self.retriever.get_relevant_documents(question)]
        self.context_proc = ""
        self.context_included = []
        # Process the documents to create a context, remove duplicate documents
        for i, doc in enumerate(self.context_docs):
            if self.context_proc.find(doc.page_content) == -1:
                self.context_included.append(True)
                self.context_proc += f'Source: {i}' + doc.metadata['source'] + "\n"
                self.context_proc += 'Information: ' + doc.page_content + '\n' 
            else:
                self.context_included.append(False)
        # Create a query from the template    
        query = retriever_tempate.format(context_proc=self.context_proc, question=question)  
        # Generate response
        print('-'*20)
        print('Retrival agent response: \n\n')
        response = self.llm.generate_response(query, config=config)
        print('-'*20)
        self.print_sources()
    
        return response, self.context_proc
    
    def print_sources(self):
        # Formatted print of the used source documents
        print(f'Sources: \n\n')
        for i, doc in enumerate(self.context_docs):
            if self.context_included[i]:
                print(f'Source: {i}' + doc.metadata['source'])
                print('Context: ' + doc.page_content)
                print('-'*10 + '\n')
        print('End of chain.')
        print('-'*20)


class ReviewerAgent(Agent):
    def __init__(self, llm: LLM, name: str = 'agent') -> None:
        super().__init__(llm, name)
        
    def generate(self, target_response: str, question: str, context: str, 
                 config: dict = {}) -> str:
        
        query = resolver_template.format(response=target_response, 
                                         question=question,
                                         context=context)
        
        # Generate response
        print('-'*20)
        print('Reviewer agent response: \n\n')
        response = self.llm.generate_response(query, config=config)
        print('-'*20)
        
        return response
