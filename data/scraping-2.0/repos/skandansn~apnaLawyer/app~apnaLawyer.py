import whisper
import requests
import pinecone
from app.config import settings
from fastapi import status, HTTPException
from langchain.llms import OpenAI
from langchain import PromptTemplate
from .schemas import QueryInput
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


async def document_input_feeder(email: str):
    loader = DirectoryLoader('./storage/files/' + email)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model_name="ada", openai_api_key=settings.OPENAI_API_KEY)

    pinecone.init(
        api_key=settings.PINECONE_API_KEY,
        environment=settings.PINECONE_ENV
    )
    index_name = settings.PINECONE_INDEX
    Pinecone.from_documents(texts, embeddings, index_name=index_name, namespace=email)


async def get_relevant_docs(query: str, namespace: str):
    embeddings = OpenAIEmbeddings(model_name="ada", openai_api_key=settings.OPENAI_API_KEY)
    pinecone.init(
        api_key=settings.PINECONE_API_KEY,
        environment=settings.PINECONE_ENV
    )

    index_name = settings.PINECONE_INDEX
    docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)
    docs = docsearch.similarity_search(query)

    return docs


async def audio_transcribe(user):
    model = whisper.load_model("base")
    result = model.transcribe("./storage/files/johndoe/tamil.mp3", task="translate")['text']
    input_query = QueryInput(query=result)
    return await langchain_query_processor(input_query, user)


async def langchain_query_processor(input_query: QueryInput, user):
    query_template = """
    You are an expert lawyer named "Apna Lawyer" well versed in Indian law.
    Answer the query of {query} in a detailed and complete way. 
    Reject if the query is not involving a law or constitution in any way.
    """
    query_prompt = PromptTemplate(
        input_variables=["query"],
        template=query_template,
    )

    llm = OpenAI(model_name=input_query.model, openai_api_key=settings.OPENAI_API_KEY)

    result = {}

    if input_query.kanoon:
        if user.role == "paid":
            headers = {'Authorization': f'Token ' + settings.KANOON_API_TOKEN}
            response = requests.post(settings.KANOON_API_URL + input_query.query, headers=headers, json={})
            docsResponse = response.json()['docs']
            docAndUrlList = []
            for i in docsResponse:
                docAndUrlList.append([i['title'], i['url']])
            result["docsList"] = docAndUrlList
            return result
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail='This feature is only available for paid tier customers.')
    if input_query.query_docs:
        if user.role == "paid":
            docs = await get_relevant_docs(input_query.query, user.email)
            prompt_template = """If the question is not related to any law or the 
            constitution, do not answer the question. If it is indeed related to a law or constitution, use the following pieces of context to answer the question at the end. If you don't 
            know the answer, try using your existing Open AI Chatgpt's general knowledge model apart form this input document to answer the question, but make sure 
            to notify that this is not in the given input context. 

            {context}

            Question: {question}
            Answer:"""
            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain({"input_documents": docs, "question": input_query.query}, return_only_outputs=True)['output_text']
            result["answer"] = response
            return result
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail='This feature is only available for paid tier customers.')

    query_output = llm(query_prompt.format(query=input_query.query))

    result['answer'] = query_output

    negation_output = None

    if input_query.negation:
        negation_template = "Turn the {answer} and explain to me what will happen if i go against this law. Reject if query is not related to law or constitution in any way."

        negation_prompt = PromptTemplate(
            input_variables=["answer"],
            template=negation_template,
        )

        negation_output = llm(negation_prompt.format(answer=query_output))

        result['negation'] = negation_output

    return result
