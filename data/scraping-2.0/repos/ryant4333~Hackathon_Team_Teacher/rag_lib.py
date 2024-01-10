import os
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms.bedrock import Bedrock

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory



def get_llm():
    
    model_kwargs = { #AI21
        "maxTokens": 1024, 
        "temperature": 0,
        "topP": 0.5, 
        "stopSequences": [], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }
    
    llm = Bedrock(
        # credentials_profile_name=
        # os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",
        # os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        # endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id="ai21.j2-ultra-v1", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm


def get_index(): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = BedrockEmbeddings(
        # credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1" #sets the region name (if not the default)
        # endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
    ) #create a Titan Embeddings client
    
    pdf_path = "Text_Book_for_Year_6_Science_Knowledge.pdf" #assumes local PDF file with this name

    loader = PyPDFLoader(file_path=pdf_path) #load the pdf file
    documents = loader.load()


    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # load it into Chroma
    db = Chroma.from_documents(docs, embeddings)


    
    # text_splitter = RecursiveCharacterTextSplitter( #create a text splitter
    #     separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
    #     chunk_size=1000, #divide into 1000-character chunks using the separators above
    #     chunk_overlap=100 #number of characters that can overlap with previous chunk
    # )
    
    # index_creator = VectorstoreIndexCreator( #create a vector store factory
    #     vectorstore_cls=FAISS, #use an in-memory vector store for demo purposes
    #     embedding=embeddings, #use Titan embeddings
    #     text_splitter=text_splitter, #use the recursive text splitter
    # )
    
    # index_from_loader = index_creator.from_loaders([loader]) #create an vector store index from the loaded PDF

    # print(index_from_loader)
    # print(type(index_from_loader))
    # # print(index_from_loader.similaritySearch("Black holes", 2))

    return db #return the index to be cached by the client app

def get_rag_response2(question): #rag client function
    llm = get_llm()

    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )

    predicted_text = conversation.predict(input="Hi there!")
    
    # response_text = index.query(question=question, llm=llm) #search against the in-memory index, stuff results into a prompt and send to the llm
    
    print(predicted_text)

    return predicted_text

def get_rag_response(index, question): #rag client function
    llm = get_llm()

    docs = index.similarity_search(question)

    # print results
    print(docs[0].page_content)


    # # result = index.query_index(question=question, llm=llm) #search against the in-memory index, stuff results into a prompt and send to the llm
    # # print(result)
    # response_text = index.query(question=question, llm=llm) #search against the in-memory index, stuff results into a prompt and send to the llm
    # # print(response_text)
    # response_text2 = index.query_with_sources(question=question, llm=llm) #search against the in-memory index, stuff results into a prompt and send to the llm
    # print(response_text2)
    return str(docs[0].page_content)

def get_custom_response(question, content, student_info):
    llm = get_llm()

    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )

    new_question2 = f"""
Student Profile:
{student_info}\n\n
Content:
{content}\n\n

Instructions: Answer the question: {question}
Ensure you Help the Student to understand the basic concept of the content above in a way that relates to their world and interests.
Always create a link between the topic and the student's interests and skills. Make the answer personal to the student by utilising relevant analogies. Weave it into a coherent narrative utilising a natural writing style.
"""

    # new_question = f"Please modify the answer: {temp_answer} so it is personalised to {student_info}.\n Make a relevant analogy that the student would understand. Begin your answer with 'Think about it like: '"

    personal_answer = conversation.predict(input=new_question2)

    return personal_answer



# Prompt them to think more rather than just giving them the answer