import torch
from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from lanchainlit import load_documents
import streamlit as st

# https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
model_id = 'openchat/openchat_v3.2'  # 60.85
# 'TheBloke/Llama-2-13B-fp16'  # 58.63
model_id = 'meta-llama/Llama-2-7b-chat-hf'  # 56.34
# model_id = 'Open-Orca/OpenOrca-Platypus2-13B'  # 64.6
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
with open('d:/demo/huggingface-api-key.txt', 'r', encoding='utf-8') as f:
    hf_token = f.readline()
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_token
)

# "In FP32 the model requires more than 60GB of RAM, you can load it in FP16 or BF16 in ~30GB, or in 8bit under 20GB of RAM..."
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_token,
    load_in_8bit=True,
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_token
)
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False
stopping_criteria = StoppingCriteriaList([StopOnTokens()])


generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
# check generate_text working fine
# res = generate_text("Explain me the difference between Data Lakehouse and Data Warehouse.")
# print(res[0]["generated_text"])


llm = HuggingFacePipeline(pipeline=generate_text)
# checking again that everything is working fine
# llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse.")


web_links = ["https://www.databricks.com/",
             "https://help.databricks.com",
             "https://databricks.com/try-databricks",
             "https://help.databricks.com/s/",
             "https://docs.databricks.com",
             "https://kb.databricks.com/",
             "http://docs.databricks.com/getting-started/index.html",
             "http://docs.databricks.com/introduction/index.html",
             "http://docs.databricks.com/getting-started/tutorials/index.html",
             "http://docs.databricks.com/release-notes/index.html",
             "http://docs.databricks.com/ingestion/index.html",
             "http://docs.databricks.com/exploratory-data-analysis/index.html",
             "http://docs.databricks.com/data-preparation/index.html",
             "http://docs.databricks.com/data-sharing/index.html",
             "http://docs.databricks.com/marketplace/index.html",
             "http://docs.databricks.com/workspace-index.html",
             "http://docs.databricks.com/machine-learning/index.html",
             "http://docs.databricks.com/sql/index.html",
             "http://docs.databricks.com/dev-tools/index.html",
             "http://docs.databricks.com/integrations/index.html",
             "http://docs.databricks.com/data-governance/index.html",
             "http://docs.databricks.com/lakehouse-architecture/index.html",
             "http://docs.databricks.com/reference/api.html",
             "http://docs.databricks.com/resources/index.html",
             "http://docs.databricks.com/archive/index.html",
             "http://docs.databricks.com/lakehouse/index.html",
             "http://docs.databricks.com/getting-started/quick-start.html",
             "http://docs.databricks.com/getting-started/etl-quick-start.html",
             "http://docs.databricks.com/getting-started/lakehouse-e2e.html",
             "http://docs.databricks.com/getting-started/free-training.html",
             "http://docs.databricks.com/sql/language-manual/index.html",
             "http://docs.databricks.com/error-messages/index.html",
             ]
loader = WebBaseLoader(web_links)
web_documents = loader.load()
data_documents = load_documents('data')
documents = web_documents + data_documents


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
# storing embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

chain = ConversationalRetrievalChain.from_llm(llm, 
                                              vectorstore.as_retriever(), 
                                              return_source_documents=True)

MODE = 'chat'
chat_history = []
# query = "How to add new b2b2c domain?"
# result = chain({"question": query, "chat_history": chat_history})
# print(result['answer'])
# print(result['source_documents'])
if MODE == 'chat':
    while True:
        query = input("query: ")
        if query == 'quit' or query == 'q':
            exit(0)
        result = chain({"question": query, "chat_history": chat_history})
        answer = result['answer'].strip()
        print(answer)
        print("")
    exit(0)

# streamlit run streamlit_app.py
st.title("⚡🔗 Flash's Q&A")
with st.form('my_form'):
    query = st.text_area('Enter text:', 'How to new b2b2d domain?')
    if query == 'quit' or query == 'q':
        exit(0)
    submitted = st.form_submit_button('Submit')
    # st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted:
        with st.spinner('Wait for it...'):
            result = chain({"question": query, "chat_history": chat_history})
            st.info(result['answer'])