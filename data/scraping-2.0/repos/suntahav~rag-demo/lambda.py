import json
import os
import pinecone
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def lambda_handler(event, context):
    """
    This function is the lambda handler for the lambda function  which is triggered by the API Gateway
    :param event: event is the input to the lambda function (contains the prompt in request body)
    :param context: context object
    :return: return the output of llm as a json object or an error message
    """
    try:
        # Extracting the prompt from the event
        query = json.loads(event['body'])['userprompt']

        # Setting the cohere API key
        os.environ["COHERE_API_KEY"] = "YOUR COHERE API KEY"

        # Setting up pinecone environment
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'YOUR PINECONE API KEY')
        PINECONE_API_ENV = os.getenv('PINECONE_API_ENV', 'YOUR PINECONE API ENVIRONMENT')
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_API_ENV,
        )
        index_name = "rag-langchain-test"

        embeddings = CohereEmbeddings(model="embed-english-v3.0")

        # switch back to normal index for langchain
        index = pinecone.Index(index_name)

        text_field = "text"
        vector_store = Pinecone(index, embeddings, text_field)

        # Init the LLM
        llm = ChatCohere()

        # Create the prompt template
        template = """Context information is below.
            ---------------------
            {context}
            ---------------------
            Using the context information mentioned above,
            answer the question: {question}
            Answer:
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        # Run the QA chain
        output = qa_chain({"query": query})

        return {
            'statusCode': 200,
            'body': json.dumps(output['result'])
        }
    except Exception as e:
        # Handling the exceptions
        return {
            'statusCode': 500,
            'body': "Something went wrong : " + str(e)
        }