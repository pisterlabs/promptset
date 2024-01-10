from flask import Flask, request, jsonify
from torch import cuda, bfloat16
import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from datetime import datetime
app = Flask(__name__)
from loguru import logger
from langchain.callbacks import FileCallbackHandler
logfile = "output_streamlit.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)
# Model setup
model_id = 'meta-llama/Llama-2-13b-chat-hf'
#tiiuae/falcon-7b-instruct
#meta-llama/Llama-2-13b-chat-hf
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [torch.LongTensor(tokenizer(x)['input_ids']).to(device) for x in stop_list]
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
    return_full_text=True,
    task='text-generation',
    stopping_criteria=stopping_criteria,
    temperature=0.2,
    max_new_tokens=500,
    repetition_penalty=1.1
)
llm = HuggingFacePipeline(pipeline=generate_text)
from langchain.document_loaders import TextLoader
loader = TextLoader("context_handbook.txt")
documents = loader.load()
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)
# Embeddings setup
#hkunlp/instructor-base
#sentence-transformers/all-MiniLM-L6-v2
model_name = "hkunlp/instructor-base"
model_kwargs = {"device": "cuda:0"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectorstore = FAISS.from_documents(all_splits, embeddings)
# Conversational Chain setup
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(search_kwargs={"k": 8}),callbacks=[handler],verbose=True)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    chat_history = []
    if 'question' in data:
        query = data['question']
        chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(search_kwargs={"k": 8}),callbacks=[handler],verbose=True)
        if "correct yourself" in query.lower():
            result = {}
            date = datetime.today().strftime('%Y-%m-%d')
            query = query.replace('correct yourself', '')
            vectorstore.add_texts(["Updated info as of "+str(date)+" :"+query])
            chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(search_kwargs={"k": 8}),callbacks=[handler],verbose=True)
            result['answer'] = "The information is updated.Thank you"
            with open("context_handbook.txt", "a") as context_file:
                context_file.write("Updated info as of " + str(date) + ": " + query + "\n")
            #print(result['answer'])
            return jsonify({"answer": result['answer']})
        else:
            result = chain({"question":query, "chat_history": chat_history})
            logger.info(result)
            #print("Source Documents",result['source_documents'])
            return jsonify({"answer": result['answer']})
    else:
        return jsonify({"error": "Missing 'question' field in request."})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5110)