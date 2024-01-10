"""
# !pip install langchain
# !pip install huggingface_hub
# !pip install faiss-gpu
# !pip install sentence_transformers
# !pip install google.generativeai
# pip install dill
# """

from langchain.llms import GooglePalm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
import os
import traceback
import dill

from backend.services.db.conn import get_cursor_conn

from dotenv import load_dotenv
load_dotenv()

TRAINING_FALLBACK_ENABLED = os.getenv('TRAINING_FALLBACK_ENABLED', 'false').lower().strip() == 'true'
DEFAULT_TRAINING_PATH = os.getenv('TRAINING_CONTEXT')
DEFAULT_DB_DILL_PATH = os.getenv('DB_DILL_PATH')
DEFAULT_CHAIN_DILL_PATH = os.getenv('CHAIN_DILL_PATH')
DEFAULT_PALM_THRESHOLD = float(os.getenv('PALM_THRESHOLD', 0.5))
NO_RESPONSE_MSG = "No response found."
REDIRECTION_MSG = "Please visit https://www.pfw.edu/etcs/computer-science for more information.\n You may also reach out to Seula Daily- Director of CS program- at dailys@pfw.edu"

def train(path_to_data=DEFAULT_TRAINING_PATH):
    global DB, CHAIN
    loader = TextLoader(path_to_data)
    documents = loader.load()

    # Convert the content into raw text.
    raw_text = ''
    for i, doc in enumerate(documents):
        text = doc.page_content
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = GooglePalmEmbeddings()
    DB = FAISS.from_texts(texts,embeddings)
    CHAIN = load_qa_chain(GooglePalm(), chain_type="stuff")

    with open(DEFAULT_DB_DILL_PATH, "wb") as f:
        dill.dump(DB, f)
    with open(DEFAULT_CHAIN_DILL_PATH, "wb") as f:
        dill.dump(CHAIN, f)

    return DB, CHAIN


def test(query, threshold=DEFAULT_PALM_THRESHOLD, **kwargs):
    """
    :return: a tuple of values final response (meant for user), original response (None in case of error), confidence
    """
    docs = DB.similarity_search(query)
    try:
        result = CHAIN.run(input_documents=docs, question=query).strip()
        updated_query = f"How accurate/relevant is the {query} to the {result}. Just give score between 0 and 1"
        accuracy = CHAIN.run(input_documents=docs, question=updated_query).strip()

        cursor, conn = get_cursor_conn()

        data = (kwargs.get('user_id'), query, result, accuracy)
        cursor.execute("""
                    INSERT INTO chats(user_id, query, response, accuracy) VALUES(?, ?, ?, ?)
                """, data)
        conn.commit()

        if float(accuracy) < threshold:
            return REDIRECTION_MSG, result, accuracy
        else:
            return result, result, accuracy
    except (IndexError, AttributeError) as e:
        print(traceback.format_exc())
        print()
        return NO_RESPONSE_MSG, None, 0


DB = CHAIN = None
try:
    with open(DEFAULT_DB_DILL_PATH, "rb") as f:
        DB = dill.load(f)
    with open(DEFAULT_CHAIN_DILL_PATH, "rb") as f:
        CHAIN = dill.load(f)
except FileNotFoundError as e:
    if TRAINING_FALLBACK_ENABLED:
        train()
    else:
        raise e
