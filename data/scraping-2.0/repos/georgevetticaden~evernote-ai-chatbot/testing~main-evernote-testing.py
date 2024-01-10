import json
import os
import uuid

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Pinecone
import pinecone


from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain



from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT


def split_pdf_text(fullpath, chunk_size, chunk_overlap):
    global doc, single_json_doc
    loader = PyPDFLoader(fullpath)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(doc)

    # Create a list of documents in json format
    chunk_docs_json_list = [doc.json() for doc in chunked_docs]

    print("Number of Json docs in list is: " + str(len(chunk_docs_json_list)))
    json_doc = chunk_docs_json_list[0]
    print("The first elemnet in the Json doc list looks like: " + json_doc)

    # Convert the list into a single json string to return from Chunk Processor
    single_json_doc = json.dumps(chunk_docs_json_list)
    # print(single_json_doc)

def get_openAI_vector_embedding(docs, open_api_key, model, chunk_size):
    texts = []
    metadatas = []
    for doc in docs:
        doc_dict = json.loads(doc)
        texts.append(doc_dict['page_content'])
        metadatas.append(doc_dict['metadata'])

    embeddings = OpenAIEmbeddings(openai_api_key=open_api_key, model=model)
    vector_embeddings = embeddings.embed_documents(texts=texts, chunk_size=chunk_size)

    # print("Number of elements in texts is: " + str(len(texts)))
    # print("Number of elements in metadatas is: " + str(len(metadatas)))
    # print("Number of elements in vector_embeddings is: " + str(len(vector_embeddings)))

    # Now that we have the embeddings, lets create list of json elements with text, metadata and vector embedding
    json_list_with_text_embeddings = []
    for text, vector_embedding, metadata in zip(texts, vector_embeddings, metadatas):
        text_embedding_json = {"text": text, "embedding": vector_embedding, "metadata": metadata}
        json_list_with_text_embeddings.append(text_embedding_json)

    print("***Number of elements in json_list_with_text_embeddings is: " + str(len(json_list_with_text_embeddings)))
    print("***First element in json_list_with_text_embeddings is: " + json.dumps(json_list_with_text_embeddings[0]))

    # Convert the list of json strings into a single json string
    json_embedding_string = json.dumps(json_list_with_text_embeddings)
    return json_embedding_string;

def put_pinecone_vector_embedding(api_key, environment, index_name, vector_embeddings_doc_list, namespace, batch_size):
    print(".........Inside upsertDocsUsingPineCone")

    texts = []
    embeddings = []
    metadatas = []
    for doc_dict in vector_embeddings_doc_list:
        texts.append(doc_dict["text"])
        embeddings.append(doc_dict['embedding'])
        metadatas.append(doc_dict['metadata'])
    pinecone.init(
        api_key=api_key,
        environment=environment
    )

    #assumes the index already exists
    index = pinecone.Index(index_name=index_name)

    # Create the List of tuples to insert into pinecone where tuple is string Id, tuple 2 is list of floats, and tuple 3 is metadata dictionary
    for i in range(0, len(texts), batch_size):
        # set end position of batch
        i_end = min(i + batch_size, len(texts))
        # get batch of texts and ids
        lines_batch = texts[i:i_end]
        # create ids
        ids_batch = [str(uuid.uuid4()) for n in range(i, i_end)]
        # get batch of embeddings
        embeddings_batch = embeddings[i:i_end]
        # prep metadata and upsert batch
        metadata_batch = metadatas[i:i_end]

        for j, line in enumerate(lines_batch):
            metadata_batch[j]["text"] = line
        to_upsert = zip(ids_batch, embeddings_batch, metadata_batch)

        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert), namespace=namespace)


def get_pinecone_vector_semantic_search (open_api_key, opean_ai_model, pinecone_api_key, environment, index_name, namespace, query, number_of_docs_to_return):

    pinecone.init(
        api_key=pinecone_api_key,
        environment=environment
    )

    embeddings = OpenAIEmbeddings(openai_api_key=open_api_key, model=opean_ai_model)

    # Do semantic search
    index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace)
    doc_results_from_search = index.similarity_search(query=query, k=number_of_docs_to_return, namespace=namespace)

    print("Number of results returned by pinecone is: " + str(len(doc_results_from_search)))

    # Convert list of Documents into list of json strings
    search_results_json_list = [result.json() for result in doc_results_from_search]

    # Convert list of json strings into single json string to mimic what we return in NiFi flowfile result
    return json.dumps(search_results_json_list)

def get_answer_from_open_Ai_LLM(open_api_key, llm_model, temperature, query, doc_results):

    doc_search_results_list = []

    for doc in doc_results:
        doc_dict = json.loads(doc)
        doc_search_results_list.append(Document(page_content=doc_dict['page_content'], metadata=doc_dict['metadata']))

    print("The following is the search results from Pine as LLM Documents")
    print(doc_search_results_list)

    llm = OpenAI(temperature=temperature, openai_api_key=open_api_key, model_name=llm_model)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=doc_search_results_list, question=query)
    return answer


def get_answer_from_open_Ai_Chat_3_with_params(open_api_key, opean_ai_model, embeddings_model,  temperature, chat_history, pinecone_api_key, environment, index_name, namespace, query):


    pinecone.init(
        api_key=pinecone_api_key,
        environment=environment
    )

    # Get Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=open_api_key, model=embeddings_model_name)

    # Get Index
    vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace)


    _template = """Given the following extracted parts of a long document and a question,
    create a final answer with references ("SOURCES") unless identified below.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    But if you are asked something similar to what your purpose is as an AI Assistant, then answer with the following:
    I'm a helpful assistant for {username} answering his questions based on the notes within his Evernote.
    Also, ALWAYS return a "SOURCES" part in your answer.

    QUESTION: Which state/country's law governs the interpretation of the contract?
    =========
    Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
    Source: 28-pl
    Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
    Source: 30-pl
    Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
    Source: 4-pl
    =========
    FINAL ANSWER: This Agreement is governed by English law.
    SOURCES: 28-pl

    QUESTION: What did the president say about Michael Jackson?
    =========
    Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
    Source: 0-pl
    Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
    Source: 24-pl
    Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
    Source: 5-pl
    Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
    Source: 34-pl
    =========
    FINAL ANSWER: The president did not mention Michael Jackson.
    SOURCES:

    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""



    # _template = """You are {username}'s Evernote Chatbot assistant answering his questions based on the notes within his Evernote.
    #   Given the following extracted parts of a long document and a question,
    #   create a final answer with references ("SOURCES") unless identified below.
    #   If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    #   But if you are asked something similar to what your purpose is as an AI Assistant, then answer with the following:
    #   I'm a helpful assistant for {username} answering his questions based on the notes within his Evernote.
    #   Also, ALWAYS return a "SOURCES" part in your answer.
    #
    #   QUESTION: Which state/country's law governs the interpretation of the contract?
    #   =========
    #   Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
    #   Source: 28-pl
    #   Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
    #   Source: 30-pl
    #   Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
    #   Source: 4-pl
    #   =========
    #   FINAL ANSWER: This Agreement is governed by English law.
    #   SOURCES: 28-pl
    #
    #   QUESTION: What did the president say about Michael Jackson?
    #   =========
    #   Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
    #   Source: 0-pl
    #   Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
    #   Source: 24-pl
    #   Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
    #   Source: 5-pl
    #   Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
    #   Source: 34-pl
    #   =========
    #   FINAL ANSWER: The president did not mention Michael Jackson.
    #   SOURCES:
    #
    #   QUESTION: {question}
    #   =========
    #   {summaries}
    #   =========
    #   FINAL ANSWER:"""

    # Create the QA prompt from the template above
    QA_PROMPT = PromptTemplate.from_template(_template)

    # Create the different chains that will be wired up
    chat_llm = ChatOpenAI(openai_api_key=open_api_key,temperature=temperature, model=opean_ai_model)
    question_generator = LLMChain(llm=chat_llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
    doc_chain = load_qa_with_sources_chain(chat_llm, chain_type="stuff", verbose=False, prompt=QA_PROMPT)

    # Use the Conversation chain to wire evertything together
    qa_chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        verbose = False
    )

    # Ask the question the LLM and get the response
    result = qa_chain({"username":'George', "question": query, "chat_history": chat_history})
    return  result["answer"]

def parse_response(response):
    answer, source = response.split('-SOURCES', 1)
    return {"answer": answer, "source": source}


# absolulate_path = '/Users/aju/Dev-Testing/evernotes/'
# filename = 'Aju_evernote_master-301.pdf'
#
# fullpath = absolulate_path + filename
#
# # Chunk the docs which returns a single json list string
# chunked_docs_as_single_json_doc = split_pdf_text(fullpath, chunk_size=1000, chunk_overlap=100)
#
# # Deserialize the json string list into a list of json documents
# chunk_docs_json_list_deserialized = json.loads(single_json_doc)
# print("The number of docs in the deserialized list is: " + str(len(chunk_docs_json_list_deserialized)))
# print("The first element in the chunk docs json list deserialized: " + chunk_docs_json_list_deserialized[0])
#
# # Create embeddings for each of the docs
# json_embedding_string = get_openAI_vector_embedding(docs=chunk_docs_json_list_deserialized,
#                                                     open_api_key=openai_api_key,
#                                                     chunk_size=1000, model=embeddings_model_name)
#
# # Deserialize the json embedding string into a list of json embeddings
# json_embedding_string_deserialized = json.loads(json_embedding_string)
# print("The number of docs in the deserialized list of json embeddings is: " + str(len(json_embedding_string_deserialized)))
# print("The first element in the json embeddings list deserialized: " + json.dumps(json_embedding_string_deserialized[0]))
#
#
# # Now insert vectors into PineCone using its python API and not Langchain
# put_pinecone_vector_embedding(api_key=pinecone_api_key, environment=pinecone_env,
#                               index_name=pinecone_index, vector_embeddings_doc_list=json_embedding_string_deserialized,
#                               namespace="evernote_content", batch_size=100)


#Get Similar docs for the query
# query = "When did my elbow pain occur? What was the cause and what did the doctor suggest???"
# vector_search_results = get_pinecone_vector_semantic_search(open_api_key=openai_api_key, opean_ai_model=embeddings_model_name,
#                                     pinecone_api_key=pinecone_api_key, environment=pinecone_env,
#                                     index_name=pinecone_index, namespace=pinecone_namespace, query=query, number_of_docs_to_return=4)
# print("Vector Search results as single json string")
# print(vector_search_results)
#
# # Deserialize the json embedding string into a list of json embeddings
# vector_search_results_deserialized = json.loads(vector_search_results)
#
# # Ask LLM the question with contextual information
# llm_answer = get_answer_from_open_Ai_LLM(open_api_key=openai_api_key, llm_model="text-davinci-003", temperature=0, query=query, doc_results=vector_search_results_deserialized)
# print("For the query["+ query +"] the answer from LLM is: " + llm_answer)

# Work with LLM Chat Model
#Get Similar docs for the query
# query = "When did my elbow pain occur? What was the cause and what did the doctor suggest???"
# query = "When did my elbow pain occur? What was the cause and what did the doctor suggest?"
# vector_search_results = get_pinecone_vector_semantic_search(open_api_key=openai_api_key, opean_ai_model=embeddings_model_name,
#                                     pinecone_api_key=pinecone_api_key, environment=pinecone_env,
#                                     index_name=pinecone_index, namespace=pinecone_namespace, query=query, number_of_docs_to_return=4)
# print("Vector Search results as single json string")
# print(vector_search_results)

# # Deserialize the json embedding string into a list of json embeddings
# vector_search_results_deserialized = json.loads(vector_search_results)
#
# # Ask Chat model the question with contextual information
# get_answer_from_open_Ai_Chat(open_api_key=openai_api_key, llm_model="text-davinci-003", temperature=0, query=query, doc_results=vector_search_results_deserialized)




query0 = "What can you help with?"
query1 = "When did my right elbow pain occur?"
query2= "What doctor did i see for it?"
query3 = "What was the cause for it and what did the doctor suggest?"
query4 = "Where did i go for physical therapy for the pain?"
query5= "What medications do i take for the elbow pain?"
query6= "What was the dosage for it?"

openai_api_key = os.environ['OPENAI_API_KEY']
pinecone_api_key = os.environ['PINECONE_API_KEY']
pinecone_env = os.environ['PINECONE_API_ENV']

llm_model_name = "gpt-3.5-turbo"
embeddings_model_name = "text-embedding-ada-002"

pinecone_index = 'book-demo'
pinecone_namespace = 'evernote_export_via_api_9'

chat_history = []
answer = get_answer_from_open_Ai_Chat_3_with_params(
    open_api_key=openai_api_key,
    opean_ai_model= llm_model_name,
    embeddings_model=embeddings_model_name,
    temperature = 0,
    chat_history= chat_history,
    pinecone_api_key=pinecone_api_key,
    environment=pinecone_env,
    index_name=pinecone_index,
    namespace=pinecone_namespace,
    query = query0
)

chat_history.append((query0, answer))
print(chat_history[0])

answer = get_answer_from_open_Ai_Chat_3_with_params(
    open_api_key=openai_api_key,
    opean_ai_model= llm_model_name,
    embeddings_model=embeddings_model_name,
    temperature = 0,
    chat_history= chat_history,
    pinecone_api_key=pinecone_api_key,
    environment=pinecone_env,
    index_name=pinecone_index,
    namespace=pinecone_namespace,
    query = query1
)

chat_history.append((query1, answer))
print(chat_history[1])

answer = get_answer_from_open_Ai_Chat_3_with_params(
    open_api_key=openai_api_key,
    opean_ai_model= llm_model_name,
    embeddings_model=embeddings_model_name,
    temperature = 0,
    chat_history= chat_history,
    pinecone_api_key=pinecone_api_key,
    environment=pinecone_env,
    index_name=pinecone_index,
    namespace=pinecone_namespace,
    query = query2
)

chat_history.append((query2, answer))
print(chat_history[2])
#
# answer = get_answer_from_open_Ai_Chat_3_with_params(
#     open_api_key=openai_api_key,
#     opean_ai_model= llm_model_name,
#     embeddings_model=embeddings_model_name,
#     temperature = 0,
#     chat_history= chat_history,
#     pinecone_api_key=pinecone_api_key,
#     environment=pinecone_env,
#     index_name=pinecone_index,
#     namespace=pinecone_namespace,
#     query = query3
# )
#
# chat_history.append((query3, answer))
# print(chat_history[3])
#
# answer = get_answer_from_open_Ai_Chat_3_with_params(
#     open_api_key=openai_api_key,
#     opean_ai_model= llm_model_name,
#     embeddings_model=embeddings_model_name,
#     temperature = 0,
#     chat_history= chat_history,
#     pinecone_api_key=pinecone_api_key,
#     environment=pinecone_env,
#     index_name=pinecone_index,
#     namespace=pinecone_namespace,
#     query = query4
# )
#
# chat_history.append((query4, answer))
# print(chat_history[4])
#
# answer = get_answer_from_open_Ai_Chat_3_with_params(
#     open_api_key=openai_api_key,
#     opean_ai_model= llm_model_name,
#     embeddings_model=embeddings_model_name,
#     temperature = 0,
#     chat_history= chat_history,
#     pinecone_api_key=pinecone_api_key,
#     environment=pinecone_env,
#     index_name=pinecone_index,
#     namespace=pinecone_namespace,
#     query = query5
# )
#
# chat_history.append((query5, answer))
# print(chat_history[5])
#
# answer = get_answer_from_open_Ai_Chat_3_with_params(
#     open_api_key=openai_api_key,
#     opean_ai_model= llm_model_name,
#     embeddings_model=embeddings_model_name,
#     temperature = 0,
#     chat_history= chat_history,
#     pinecone_api_key=pinecone_api_key,
#     environment=pinecone_env,
#     index_name=pinecone_index,
#     namespace=pinecone_namespace,
#     query = query6
# )
#
# chat_history.append((query6, answer))
# print(chat_history[6])