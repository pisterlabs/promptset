import pinecone, tiktoken, os
from tqdm.auto import tqdm
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from uuid import uuid4
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
import pickle

load_dotenv()
# find API key in console at app.pinecone.io
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') 
# find ENV (cloud region) next to API key in console
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

data = []
with open('data.pkl', 'rb') as f:
    data, _ = pickle.load(f) # saved by pubparser.py

# for filename in os.listdir("publications"):
#     with open(os.path.join("publications", filename), "r") as f:
#         fulltext = f.read()
#         data.append({'text': fulltext, 'name': filename})

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["<Page ", "\n", " ", ""]
)

# chunks = text_splitter.split_text(data[0]['text])

embed = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=OPENAI_API_KEY
)

# res = embed.embed_documents(chunks[2:3])

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

index_name = 'taxman'
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )

index = pinecone.Index(index_name)
# index.delete(delete_all=True)
# assert False
batch_limit = 1
texts = []
metadatas = []

if index.describe_index_stats()['total_vector_count'] == 0:

    for i, record in enumerate(tqdm(data)):
        if i > 251:
            # first get metadata fields for this record
            metadata = {
                'source': record['source'],
                'title': record['title'],
                # 'date': record['date'],
                # 'updated': record['updated']
            }
            # now we create chunks from the record text
            record_texts = text_splitter.split_text(record['text'])
            # create individual metadata dicts for each chunk
            record_metadatas = [{
                "chunk": j, "text": text, **metadata
            } for j, text in enumerate(record_texts)]
            # append these to current batches
            texts.extend(record_texts)
            metadatas.extend(record_metadatas)
            # if we have reached the batch_limit we can add texts
            if len(texts) >= batch_limit:
                ids = [str(uuid4()) for _ in range(len(texts))]
                embeds = embed.embed_documents(texts)
                index.upsert(vectors=zip(ids, embeds, metadatas))
                texts = []
                metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))


print(index.describe_index_stats())

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# query = "My wife and I own a business. How should we declare our income?"
"""
If you and your wife jointly own and operate an unincorporated business and share in the profits and losses, you are considered partners in a partnership. In this case, you should file Form 1065, U.S. Return of Partnership Income, instead of using Schedule C. This applies whether or not you have a formal partnership agreement.
However, there are exceptions. If you and your wife wholly own the business as community property under the community property laws of a state, foreign country, or U.S. possession, you can treat the business either as a sole proprietorship or a partnership. You can also make a joint election to be treated as a qualified joint venture instead of a partnership if you and your wife each materially participate as the only members of the business and file a joint tax return.
It's important to note that if your business is owned and operated through an LLC, it does not qualify for the election of a qualified joint venture.
To determine the best way to declare your income, you may want to consult the instructions for Form 1065, Pub. 541 (Partnerships), and Pub. 555 (Community Property).
"""
# query = "I own a business with a handful of employees, but I plan on expanding. What steps can I take to pay less tax?"
query = "I own a small business and sometimes use my car for business. Can I get a tax write off on my car payments?"
"""Yes, you may be able to deduct the costs of operating and maintaining your car for business purposes. This includes car payments, as well as other expenses such as gas, oil, repairs, insurance, and parking fees. However, you must divide your expenses between business and personal use based on the miles driven for each purpose. You can only deduct the portion of expenses that is attributable to your business use of the car. For more information, you can refer to Publication 463."""

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)


# # Llama2 7B params with 4 bit quantization on MacBook Pro M1 is just okay..., will have to see how a bigger model performs
# # Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# # Verbose is required to pass to the callback manager
# n_gpu_layers = 1  # Metal set to 1 is enough.
# n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="../../llama.cpp/models/7B/ggml-model-q4_0.bin",
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     temperature=0.5,
#     n_ctx=3000,
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     callback_manager=callback_manager,
#     verbose=True,
# )

# Llama2 13B params on cluster
# Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# # Verbose is required to pass to the callback manager
# n_gpu_layers = 1  # Metal set to 1 is enough.
# n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="../../llama.cpp/models/7B/ggml-model-q4_0.bin",
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     temperature=0.5,
#     n_ctx=3000,
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     callback_manager=callback_manager,
#     verbose=True,
# )

qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print(qa(query))

"""
If you and your wife jointly own and operate an unincorporated business and share in the profits and losses, you are considered partners in a partnership. In this case, you should file Form 1065, U.S. Return of Partnership Income, instead of using Schedule C. This applies whether or not you have a formal partnership agreement.
However, there are exceptions. If you and your wife wholly own the business as community property under the community property laws of a state, foreign country, or U.S. possession, you can treat the business either as a sole proprietorship or a partnership. You can also make a joint election to be treated as a qualified joint venture instead of a partnership if you and your wife each materially participate as the only members of the business and file a joint tax return.
It's important to note that if your business is owned and operated through an LLC, it does not qualify for the election of a qualified joint venture.
To determine the best way to declare your income, you may want to consult the instructions for Form 1065, Pub. 541 (Partnerships), and Pub. 555 (Community Property).
"""

# Can add 'source' url to use RetrievalQAWithSourcesChain