from decouple import config
from .models import UserProfile
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.docstore.document import Document 
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import logging
from langchain.prompts import PromptTemplate
import tiktoken
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
    ConversationSummaryMemory,
)
from langchain.chains import LLMChain
from langchain.llms import OpenAI





OPENAI_API_KEY = config("OPENAI_API_KEY")

model_name = 'gpt-4-0613'

PINECONE_API_KEY = config("Pinecone_API")
PINECONE_ENVIRONMENT = config("Pinecone_env")

llm = ChatOpenAI(temperature=0, model=model_name)
llm_title = ChatOpenAI(temperature=0,model='gpt-3.5-turbo')


pinecone.init(
    api_key=PINECONE_API_KEY ,  # find at app.pinecone.io
    environment=PINECONE_ENVIRONMENT  # next to api key in console
)

logger = logging.getLogger(__name__)

def query_pinecone(query):
    # generate embeddings for the query
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    embed_query = embeddings.embed_query(query)
    # search pinecone index for context passage with the answer
    
    return embed_query



def generate_title(query):

    
    prompt = PromptTemplate(
    input_variables=["question"],
    template="Generate title (less than 25 characters) for the conversation for the given {question}",
)
    

    chain = LLMChain(llm=llm_title, prompt=prompt)

    title = chain.run(question=query)



    return title








def get_response(query, pinecone_index ,namespace):

    conv_memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    summary_memory = ConversationSummaryMemory(llm=ChatOpenAI(), input_key="question")



    
    memory = CombinedMemory(memories=[conv_memory, summary_memory])
    """
    Returns a response to the given query using the given Pinecone index.
    If the Pinecone index is not initialized, it raises a ValueError.
    """

    
    prompt_template = """
       In the primary conversation, the user will pose a question, for which you will receive a context in the form of a list of documents obtained via a vector similarity search based on the user's query. 

        Context (Vector Similarity Search Documents): {context}

        Additionally, you will have access to the chat history. This will allow you to remember the entities involved and be intelligent with your responses. 

        Chat History: {chat_history}

        At the start of each new session, you will also have the summary of the conversation to assist you in being consistent and knowledgeable in your responses.

        Conversation Summary: {history}

        Now, the Question from the user is: {question}

        Using the above inputs, your aim is to answer the user's question as best as possible. If the question exceeds the scope of the given context or chat history, kindly request the user to provide more context to enhance your understanding and improve your response's accuracy. 

        Also, if the User asks for help with any code, provide them with an example in Python (unless another language is specified), and continue the dialogue to find a solution.

        The memory provided to you is a combination of ConversationBufferMemory and ConversationSummaryMemory. This will assist you in maintaining context and coherence throughout your responses.

        No additional suffix should be added to your answers.

        Let's proceed:

"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question","chat_history","history"]
    )

    try:
        index = pinecone.Index(index_name=pinecone_index)
        xq = query_pinecone(query)

        xc = index.query(xq,top_k=5, include_metadata=True,namespace=namespace)


        # Use OpenAI API to generate a response
        test2 = []
        for i in xc['matches']:
            metadata = i['metadata']
            text_value = metadata.pop('text', None)
            if text_value is not None:
                test2.append(Document(page_content=text_value,metadata=metadata,lookup_index=0))

        chain = load_qa_chain(llm, chain_type="stuff",prompt=PROMPT,memory=memory)
        response = chain({"input_documents": test2, "question": query}, return_only_outputs=True)

        
        print("Response",response['output_text'])

        
        return response['output_text']
    
    except AttributeError as e:
        logger.exception(f"Error accessing Pinecone index: {e}")
        return None
    except Exception as e:
        logger.exception(f"Error getting response: {e}")
        return None
