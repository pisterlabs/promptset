from dotenv import load_dotenv, find_dotenv
from uuid import uuid4
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationChain
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select

from models import FileEmbedding, create_user_embedding_table
load_dotenv(find_dotenv())

embedding_model = OpenAIEmbeddings()


def file_to_embedding(filepath, session, email, file_id):
    reader = PdfReader(filepath)

    totaltext = ""
    for page in reader.pages:
        totaltext = totaltext + " " + page.extract_text()

    # start generating embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )

    embedding_documents = text_splitter.split_text(totaltext)
    document_embeddings = embedding_model.embed_documents(embedding_documents)
    for i in range(len(document_embeddings)):
        insert_stmt = insert(FileEmbedding).values(embedding=document_embeddings[i], embedding_owner=email,
                                                   source_file=file_id, embedding_text=embedding_documents[i],
                                                   embedding_id=uuid4())
        session.execute(insert_stmt)


def get_answer_from_chunks(question, session, email, file_ids, conversation_id):
    user_embedding_table = create_user_embedding_table(email)

    question_embedding = embedding_model.embed_query(question)

    select_stmt = select(user_embedding_table.embedding_text).order_by(
        user_embedding_table.embedding.cosine_distance(question_embedding)).limit(6)

    relevant_chunks = session.execute(select_stmt).all()

    print(relevant_chunks)
    return "successful answer"
    prompt_template = "This is some information on a topic: {context}. Please answer this question using this information: " + question

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context"]
    )

    chain = load_qa_chain(OpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", prompt=prompt)
    answer = chain.run({"input_documents": similar_chunks})
    answer = answer.replace("\n", "")
    return answer