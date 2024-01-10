import re
import os
import datetime
from typing import TypeVar, Dict, List, Tuple
import time
from itertools import compress
import pandas as pd
import numpy as np

# Model packages
import torch.cuda
from threading import Thread
from transformers import pipeline, TextIteratorStreamer

# Alternative model sources
#from dataclasses import asdict, dataclass

# Langchain functions
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.retrievers import SVMRetriever 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# For keyword extraction (not currently used)
#import nltk
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT

# For Name Entity Recognition model
#from span_marker import SpanMarkerModel # Not currently used

# For BM25 retrieval
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity

import gradio as gr

torch.cuda.empty_cache()

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

embeddings = None  # global variable setup
vectorstore = None # global variable setup
model_type = None # global variable setup

max_memory_length = 0 # How long should the memory of the conversation last?

full_text = "" # Define dummy source text (full text) just to enable highlight function to load

model = [] # Define empty list for model functions to run
tokenizer = [] # Define empty list for model functions to run

## Highlight text constants
hlt_chunk_size = 12
hlt_strat = [" ", ". ", "! ", "? ", ": ", "\n\n", "\n", ", "]
hlt_overlap = 4

## Initialise NER model ##
ner_model = []#SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd") # Not currently used

## Initialise keyword model ##
# Used to pull out keywords from chat history to add to user queries behind the scenes
kw_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Currently set gpu_layers to 0 even with cuda due to persistent bugs in implementation with cuda
if torch.cuda.is_available():
    torch_device = "cuda"
    gpu_layers = 0
else: 
    torch_device =  "cpu"
    gpu_layers = 0

print("Running on device:", torch_device)
threads = 8 #torch.get_num_threads()
print("CPU threads:", threads)

# Flan Alpaca (small, fast) Model parameters
temperature: float = 0.1
top_k: int = 3
top_p: float = 1
repetition_penalty: float = 1.3
flan_alpaca_repetition_penalty: float = 1.3
last_n_tokens: int = 64
max_new_tokens: int = 256
seed: int = 42
reset: bool = False
stream: bool = True
threads: int = threads
batch_size:int = 256
context_length:int = 2048
sample = True


class CtransInitConfig_gpu:
    def __init__(self, temperature=temperature,
                 top_k=top_k,
                 top_p=top_p,
                 repetition_penalty=repetition_penalty,
                 last_n_tokens=last_n_tokens,
                 max_new_tokens=max_new_tokens,
                 seed=seed,
                 reset=reset,
                 stream=stream,
                 threads=threads,
                 batch_size=batch_size,
                 context_length=context_length,
                 gpu_layers=gpu_layers):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty# repetition_penalty
        self.last_n_tokens = last_n_tokens
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.reset = reset
        self.stream = stream
        self.threads = threads
        self.batch_size = batch_size
        self.context_length = context_length
        self.gpu_layers = gpu_layers
        # self.stop: list[str] = field(default_factory=lambda: [stop_string])

    def update_gpu(self, new_value):
        self.gpu_layers = new_value

class CtransInitConfig_cpu(CtransInitConfig_gpu):
    def __init__(self):
        super().__init__()
        self.gpu_layers = 0

gpu_config = CtransInitConfig_gpu()
cpu_config = CtransInitConfig_cpu()


class CtransGenGenerationConfig:
    def __init__(self, temperature=temperature,
                 top_k=top_k,
                 top_p=top_p,
                 repetition_penalty=repetition_penalty,
                 last_n_tokens=last_n_tokens,
                 seed=seed,
                 threads=threads,
                 batch_size=batch_size,
                 reset=True
                 ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty# repetition_penalty
        self.last_n_tokens = last_n_tokens
        self.seed = seed
        self.threads = threads
        self.batch_size = batch_size
        self.reset = reset

    def update_temp(self, new_value):
        self.temperature = new_value

# Vectorstore funcs

def docs_to_faiss_save(docs_out:PandasDataFrame, embeddings=embeddings):

    print(f"> Total split documents: {len(docs_out)}")

    vectorstore_func = FAISS.from_documents(documents=docs_out, embedding=embeddings)
        
    '''  
    #with open("vectorstore.pkl", "wb") as f:
        #pickle.dump(vectorstore, f) 
    ''' 

    #if Path(save_to).exists():
    #    vectorstore_func.save_local(folder_path=save_to)
    #else:
    #    os.mkdir(save_to)
    #    vectorstore_func.save_local(folder_path=save_to)

    global vectorstore

    vectorstore = vectorstore_func

    out_message = "Document processing complete"

    #print(out_message)
    #print(f"> Saved to: {save_to}")

    return out_message

# Prompt functions

def base_prompt_templates(model_type = "Flan Alpaca (small, fast)"):    
  
    #EXAMPLE_PROMPT = PromptTemplate(
    #    template="\nCONTENT:\n\n{page_content}\n\nSOURCE: {source}\n\n",
    #    input_variables=["page_content", "source"],
    #)

    CONTENT_PROMPT = PromptTemplate(
        template="{page_content}\n\n",#\n\nSOURCE: {source}\n\n",
        input_variables=["page_content"]
    )

# The main prompt:

    instruction_prompt_template_alpaca_quote = """### Instruction:
Quote directly from the SOURCE below that best answers the QUESTION. Only quote full sentences in the correct order. If you cannot find an answer, start your response with "My best guess is: ".
    
CONTENT: {summaries}   
QUESTION: {question}

Response:"""

    instruction_prompt_template_alpaca = """### Instruction:
### User:
Answer the QUESTION using information from the following CONTENT.
CONTENT: {summaries}
QUESTION: {question}

Response:"""


    instruction_prompt_template_wizard_orca = """### HUMAN:
Answer the QUESTION below based on the CONTENT. Only refer to CONTENT that directly answers the question.
CONTENT - {summaries}
QUESTION - {question}
### RESPONSE:
"""


    instruction_prompt_template_orca = """
### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.
### User:
Answer the QUESTION with a short response using information from the following CONTENT.
QUESTION: {question}
CONTENT: {summaries}

### Response:"""

    instruction_prompt_template_orca_quote = """
### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.
### User:
Quote text from the CONTENT to answer the QUESTION below.
QUESTION: {question}
CONTENT: {summaries}  
### Response:
"""


    instruction_prompt_mistral_orca = """<|im_start|>system\n
You are an AI assistant that follows instruction extremely well. Help as much as you can.
<|im_start|>user\n
Answer the QUESTION using information from the following CONTENT. Respond with short answers that directly answer the question.
CONTENT: {summaries}
QUESTION: {question}\n
Answer:<|im_end|>"""

    if model_type == "Flan Alpaca (small, fast)":
        INSTRUCTION_PROMPT=PromptTemplate(template=instruction_prompt_template_alpaca, input_variables=['question', 'summaries'])
    elif model_type == "Mistral Open Orca (larger, slow)":
        INSTRUCTION_PROMPT=PromptTemplate(template=instruction_prompt_mistral_orca, input_variables=['question', 'summaries'])

    return INSTRUCTION_PROMPT, CONTENT_PROMPT

def write_out_metadata_as_string(metadata_in):
    metadata_string = [f"{'  '.join(f'{k}: {v}' for k, v in d.items() if k != 'page_section')}" for d in metadata_in] # ['metadata']
    return metadata_string

def generate_expanded_prompt(inputs: Dict[str, str], instruction_prompt, content_prompt, extracted_memory, vectorstore, embeddings, out_passages = 2): # , 
        
        question =  inputs["question"]
        chat_history = inputs["chat_history"]
        

        new_question_kworded = adapt_q_from_chat_history(question, chat_history, extracted_memory) # new_question_keywords, 
        
       
        docs_keep_as_doc, doc_df, docs_keep_out = hybrid_retrieval(new_question_kworded, vectorstore, embeddings, k_val = 25, out_passages = out_passages,
                                                                          vec_score_cut_off = 0.85, vec_weight = 1, bm25_weight = 1, svm_weight = 1)#,
                                                                          #vectorstore=globals()["vectorstore"], embeddings=globals()["embeddings"])
        
        #print(docs_keep_as_doc)
        #print(doc_df)
        if (not docs_keep_as_doc) | (doc_df.empty):
            sorry_prompt = """Say 'Sorry, there is no relevant information to answer this question.'.
RESPONSE:"""
            return sorry_prompt, "No relevant sources found.", new_question_kworded
        
        # Expand the found passages to the neighbouring context
        file_type = determine_file_type(doc_df['meta_url'][0])

        # Only expand passages if not tabular data
        if (file_type != ".csv") & (file_type != ".xlsx"):
            docs_keep_as_doc, doc_df = get_expanded_passages(vectorstore, docs_keep_out, width=3)

        
 
        # Build up sources content to add to user display
        doc_df['meta_clean'] = write_out_metadata_as_string(doc_df["metadata"]) # [f"<b>{'  '.join(f'{k}: {v}' for k, v in d.items() if k != 'page_section')}</b>" for d in doc_df['metadata']]
        
        # Remove meta text from the page content if it already exists there
        doc_df['page_content_no_meta'] = doc_df.apply(lambda row: row['page_content'].replace(row['meta_clean'] + ". ", ""), axis=1)
        doc_df['content_meta'] = doc_df['meta_clean'].astype(str) + ".<br><br>" + doc_df['page_content_no_meta'].astype(str)

        #modified_page_content = [f" Document {i+1} - {word}" for i, word in enumerate(doc_df['page_content'])]
        modified_page_content = [f" Document {i+1} - {word}" for i, word in enumerate(doc_df['content_meta'])]
        docs_content_string = '<br><br>'.join(modified_page_content)

        sources_docs_content_string = '<br><br>'.join(doc_df['content_meta'])#.replace("  "," ")#.strip()
     
        instruction_prompt_out = instruction_prompt.format(question=new_question_kworded, summaries=docs_content_string)
        
        print('Final prompt is: ')
        print(instruction_prompt_out)
                
        return instruction_prompt_out, sources_docs_content_string, new_question_kworded

def create_full_prompt(user_input, history, extracted_memory, vectorstore, embeddings, model_type, out_passages):
    
    if not user_input.strip():
        return history, "", "Respond with 'Please enter a question.' RESPONSE:"

    #if chain_agent is None:
    #    history.append((user_input, "Please click the button to submit the Huggingface API key before using the chatbot (top right)"))
    #    return history, history, "", ""
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("User input: " + user_input)
    
    history = history or []
    
    # Create instruction prompt
    instruction_prompt, content_prompt = base_prompt_templates(model_type=model_type)
    instruction_prompt_out, docs_content_string, new_question_kworded =\
                generate_expanded_prompt({"question": user_input, "chat_history": history}, #vectorstore,
                                    instruction_prompt, content_prompt, extracted_memory, vectorstore, embeddings, out_passages)
    
  
    history.append(user_input)
    
    print("Output history is:")
    print(history)

    print("Final prompt to model is:")
    print(instruction_prompt_out)
        
    return history, docs_content_string, instruction_prompt_out

# Chat functions
def produce_streaming_answer_chatbot(history, full_prompt, model_type,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            sample=sample,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k
):
    #print("Model type is: ", model_type)

    #if not full_prompt.strip():
    #    if history is None:
    #        history = []

    #    return history

    if model_type == "Flan Alpaca (small, fast)": 
        # Get the model and tokenizer, and tokenize the user text.
        model_inputs = tokenizer(text=full_prompt, return_tensors="pt", return_attention_mask=False).to(torch_device) # return_attention_mask=False was added

        # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
        # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
        streamer = TextIteratorStreamer(tokenizer, timeout=120., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            temperature=temperature,
            top_k=top_k
        )

        print(generate_kwargs)

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        # Pull the generated text from the streamer, and update the model output.
        start = time.time()
        NUM_TOKENS=0
        print('-'*4+'Start Generation'+'-'*4)

        history[-1][1] = ""
        for new_text in streamer:
            if new_text == None: new_text = ""
            history[-1][1] += new_text
            NUM_TOKENS+=1
            yield history
            
        time_generate = time.time() - start
        print('\n')
        print('-'*4+'End Generation'+'-'*4)
        print(f'Num of generated tokens: {NUM_TOKENS}')
        print(f'Time for complete generation: {time_generate}s')
        print(f'Tokens per secound: {NUM_TOKENS/time_generate}')
        print(f'Time per token: {(time_generate/NUM_TOKENS)*1000}ms')

    elif model_type == "Mistral Open Orca (larger, slow)":
        tokens = model.tokenize(full_prompt)

        gen_config = CtransGenGenerationConfig()
        gen_config.update_temp(temperature)

        print(vars(gen_config))

        # Pull the generated text from the streamer, and update the model output.
        start = time.time()
        NUM_TOKENS=0
        print('-'*4+'Start Generation'+'-'*4)

        history[-1][1] = ""
        for new_text in model.generate(tokens, **vars(gen_config)): #CtransGen_generate(prompt=full_prompt)#, config=CtransGenGenerationConfig()): # #top_k=top_k, temperature=temperature, repetition_penalty=repetition_penalty,
            if new_text == None: new_text =  ""
            history[-1][1] += model.detokenize(new_text) #new_text
            NUM_TOKENS+=1
            yield history
        
        time_generate = time.time() - start
        print('\n')
        print('-'*4+'End Generation'+'-'*4)
        print(f'Num of generated tokens: {NUM_TOKENS}')
        print(f'Time for complete generation: {time_generate}s')
        print(f'Tokens per secound: {NUM_TOKENS/time_generate}')
        print(f'Time per token: {(time_generate/NUM_TOKENS)*1000}ms')

# Chat helper functions

def adapt_q_from_chat_history(question, chat_history, extracted_memory, keyword_model=""):#keyword_model): # new_question_keywords, 
 
        chat_history_str, chat_history_first_q, chat_history_first_ans, max_memory_length = _get_chat_history(chat_history)

        if chat_history_str:
            # Keyword extraction is now done in the add_inputs_to_history function
            #remove_q_stopwords(str(chat_history_first_q) + " " + str(chat_history_first_ans))
            
           
            new_question_kworded = str(extracted_memory) + ". " + question #+ " " + new_question_keywords
            #extracted_memory + " " + question
            
        else:
            new_question_kworded = question #new_question_keywords

        #print("Question output is: " + new_question_kworded)
            
        return new_question_kworded

def determine_file_type(file_path):
        """
        Determine the file type based on its extension.
    
        Parameters:
            file_path (str): Path to the file.
    
        Returns:
            str: File extension (e.g., '.pdf', '.docx', '.txt', '.html').
        """
        return os.path.splitext(file_path)[1].lower()


def create_doc_df(docs_keep_out):
    # Extract content and metadata from 'winning' passages.
            content=[]
            meta=[]
            meta_url=[]
            page_section=[]
            score=[]

            doc_df = pd.DataFrame()

            

            for item in docs_keep_out:
                content.append(item[0].page_content)
                meta.append(item[0].metadata)
                meta_url.append(item[0].metadata['source'])

                file_extension = determine_file_type(item[0].metadata['source'])
                if (file_extension != ".csv") & (file_extension != ".xlsx"):
                    page_section.append(item[0].metadata['page_section'])
                else: page_section.append("")
                score.append(item[1])       

            # Create df from 'winning' passages

            doc_df = pd.DataFrame(list(zip(content, meta, page_section, meta_url, score)),
               columns =['page_content', 'metadata', 'page_section', 'meta_url', 'score'])

            docs_content = doc_df['page_content'].astype(str)
            doc_df['full_url'] = "https://" + doc_df['meta_url'] 

            return doc_df

def hybrid_retrieval(new_question_kworded, vectorstore, embeddings, k_val, out_passages,
                           vec_score_cut_off, vec_weight, bm25_weight, svm_weight): # ,vectorstore, embeddings

            #vectorstore=globals()["vectorstore"]
            #embeddings=globals()["embeddings"]
            doc_df = pd.DataFrame()


            docs = vectorstore.similarity_search_with_score(new_question_kworded, k=k_val)

            print("Docs from similarity search:")
            print(docs)

            # Keep only documents with a certain score
            docs_len = [len(x[0].page_content) for x in docs]
            docs_scores = [x[1] for x in docs]

            # Only keep sources that are sufficiently relevant (i.e. similarity search score below threshold below)
            score_more_limit = pd.Series(docs_scores) < vec_score_cut_off
            docs_keep = list(compress(docs, score_more_limit))

            if not docs_keep:
                return [], pd.DataFrame(), []

            # Only keep sources that are at least 100 characters long
            length_more_limit = pd.Series(docs_len) >= 100
            docs_keep = list(compress(docs_keep, length_more_limit))

            if not docs_keep:
                return [], pd.DataFrame(), []

            docs_keep_as_doc = [x[0] for x in docs_keep]
            docs_keep_length = len(docs_keep_as_doc)


                
            if docs_keep_length == 1:

                content=[]
                meta_url=[]
                score=[]
                
                for item in docs_keep:
                    content.append(item[0].page_content)
                    meta_url.append(item[0].metadata['source'])
                    score.append(item[1])       

                # Create df from 'winning' passages

                doc_df = pd.DataFrame(list(zip(content, meta_url, score)),
                columns =['page_content', 'meta_url', 'score'])

                docs_content = doc_df['page_content'].astype(str)
                docs_url = doc_df['meta_url']

                return docs_keep_as_doc, docs_content, docs_url
            
            # Check for if more docs are removed than the desired output
            if out_passages > docs_keep_length: 
                out_passages = docs_keep_length
                k_val = docs_keep_length
                     
            vec_rank = [*range(1, docs_keep_length+1)]
            vec_score = [(docs_keep_length/x)*vec_weight for x in vec_rank]

            # 2nd level check on retrieved docs with BM25

            content_keep=[]
            for item in docs_keep:
                content_keep.append(item[0].page_content)

            corpus = corpus = [doc.lower().split() for doc in content_keep]
            dictionary = Dictionary(corpus)
            bm25_model = OkapiBM25Model(dictionary=dictionary)
            bm25_corpus = bm25_model[list(map(dictionary.doc2bow, corpus))]
            bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                   normalize_queries=False, normalize_documents=False)
            query = new_question_kworded.lower().split()
            tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')  # Enforce binary weighting of queries
            tfidf_query = tfidf_model[dictionary.doc2bow(query)]
            similarities = np.array(bm25_index[tfidf_query])
            #print(similarities)
            temp = similarities.argsort()
            ranks = np.arange(len(similarities))[temp.argsort()][::-1]

            # Pair each index with its corresponding value
            pairs = list(zip(ranks, docs_keep_as_doc))
            # Sort the pairs by the indices
            pairs.sort()
            # Extract the values in the new order
            bm25_result = [value for ranks, value in pairs]
            
            bm25_rank=[]
            bm25_score = []

            for vec_item in docs_keep:
                x = 0
                for bm25_item in bm25_result:
                    x = x + 1
                    if bm25_item.page_content == vec_item[0].page_content:
                        bm25_rank.append(x)
                        bm25_score.append((docs_keep_length/x)*bm25_weight)

            # 3rd level check on retrieved docs with SVM retriever
            svm_retriever = SVMRetriever.from_texts(content_keep, embeddings, k = k_val)
            svm_result = svm_retriever.get_relevant_documents(new_question_kworded)

         
            svm_rank=[]
            svm_score = []

            for vec_item in docs_keep:
                x = 0
                for svm_item in svm_result:
                    x = x + 1
                    if svm_item.page_content == vec_item[0].page_content:
                        svm_rank.append(x)
                        svm_score.append((docs_keep_length/x)*svm_weight)

        
            ## Calculate final score based on three ranking methods
            final_score = [a  + b + c for a, b, c in zip(vec_score, bm25_score, svm_score)]
            final_rank = [sorted(final_score, reverse=True).index(x)+1 for x in final_score]
            # Force final_rank to increment by 1 each time
            final_rank = list(pd.Series(final_rank).rank(method='first'))

            #print("final rank: " + str(final_rank))
            #print("out_passages: " + str(out_passages))

            best_rank_index_pos = []

            for x in range(1,out_passages+1):
                try:
                    best_rank_index_pos.append(final_rank.index(x))
                except IndexError: # catch the error
                    pass

            # Adjust best_rank_index_pos to 

            best_rank_pos_series = pd.Series(best_rank_index_pos)


            docs_keep_out = [docs_keep[i] for i in best_rank_index_pos]
        
            # Keep only 'best' options
            docs_keep_as_doc = [x[0] for x in docs_keep_out]
                               
            # Make df of best options
            doc_df = create_doc_df(docs_keep_out)

            return docs_keep_as_doc, doc_df, docs_keep_out

def get_expanded_passages(vectorstore, docs, width):

    """
    Extracts expanded passages based on given documents and a width for context.
    
    Parameters:
    - vectorstore: The primary data source.
    - docs: List of documents to be expanded.
    - width: Number of documents to expand around a given document for context.
    
    Returns:
    - expanded_docs: List of expanded Document objects.
    - doc_df: DataFrame representation of expanded_docs.
    """

    from collections import defaultdict
    
    def get_docs_from_vstore(vectorstore):
        vector = vectorstore.docstore._dict
        return list(vector.items())

    def extract_details(docs_list):
        docs_list_out = [tup[1] for tup in docs_list]
        content = [doc.page_content for doc in docs_list_out]
        meta = [doc.metadata for doc in docs_list_out]
        return ''.join(content), meta[0], meta[-1]
    
    def get_parent_content_and_meta(vstore_docs, width, target):
        #target_range = range(max(0, target - width), min(len(vstore_docs), target + width + 1))
        target_range = range(max(0, target), min(len(vstore_docs), target + width + 1)) # Now only selects extra passages AFTER the found passage
        parent_vstore_out = [vstore_docs[i] for i in target_range]
        
        content_str_out, meta_first_out, meta_last_out = [], [], []
        for _ in parent_vstore_out:
            content_str, meta_first, meta_last = extract_details(parent_vstore_out)
            content_str_out.append(content_str)
            meta_first_out.append(meta_first)
            meta_last_out.append(meta_last)
        return content_str_out, meta_first_out, meta_last_out

    def merge_dicts_except_source(d1, d2):
            merged = {}
            for key in d1:
                if key != "source":
                    merged[key] = str(d1[key]) + " to " + str(d2[key])
                else:
                    merged[key] = d1[key]  # or d2[key], based on preference
            return merged

    def merge_two_lists_of_dicts(list1, list2):
        return [merge_dicts_except_source(d1, d2) for d1, d2 in zip(list1, list2)]

    # Step 1: Filter vstore_docs
    vstore_docs = get_docs_from_vstore(vectorstore)
    doc_sources = {doc.metadata['source'] for doc, _ in docs}
    vstore_docs = [(k, v) for k, v in vstore_docs if v.metadata.get('source') in doc_sources]

    # Step 2: Group by source and proceed
    vstore_by_source = defaultdict(list)
    for k, v in vstore_docs:
        vstore_by_source[v.metadata['source']].append((k, v))
        
    expanded_docs = []
    for doc, score in docs:
        search_source = doc.metadata['source']
        

        #if file_type == ".csv" | file_type == ".xlsx":
        #     content_str, meta_first, meta_last = get_parent_content_and_meta(vstore_by_source[search_source], 0, search_index)

        #else:
        search_section = doc.metadata['page_section']
        parent_vstore_meta_section = [doc.metadata['page_section'] for _, doc in vstore_by_source[search_source]]
        search_index = parent_vstore_meta_section.index(search_section) if search_section in parent_vstore_meta_section else -1

        content_str, meta_first, meta_last = get_parent_content_and_meta(vstore_by_source[search_source], width, search_index)
        meta_full = merge_two_lists_of_dicts(meta_first, meta_last)

        expanded_doc = (Document(page_content=content_str[0], metadata=meta_full[0]), score)
        expanded_docs.append(expanded_doc)

    doc_df = pd.DataFrame()

    doc_df = create_doc_df(expanded_docs)  # Assuming you've defined the 'create_doc_df' function elsewhere

    return expanded_docs, doc_df

def highlight_found_text(search_text: str, full_text: str, hlt_chunk_size:int=hlt_chunk_size, hlt_strat:List=hlt_strat, hlt_overlap:int=hlt_overlap) -> str:
    """
    Highlights occurrences of search_text within full_text.
    
    Parameters:
    - search_text (str): The text to be searched for within full_text.
    - full_text (str): The text within which search_text occurrences will be highlighted.
    
    Returns:
    - str: A string with occurrences of search_text highlighted.
    
    Example:
    >>> highlight_found_text("world", "Hello, world! This is a test. Another world awaits.")
    'Hello, <mark style="color:black;">world</mark>! This is a test. Another <mark style="color:black;">world</mark> awaits.'
    """

    def extract_text_from_input(text, i=0):
        if isinstance(text, str):
            return text.replace("  ", " ").strip()
        elif isinstance(text, list):
            return text[i][0].replace("  ", " ").strip()
        else:
            return ""

    def extract_search_text_from_input(text):
        if isinstance(text, str):
            return text.replace("  ", " ").strip()
        elif isinstance(text, list):
            return text[-1][1].replace("  ", " ").strip()
        else:
            return ""

    full_text = extract_text_from_input(full_text)
    search_text = extract_search_text_from_input(search_text)



    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=hlt_chunk_size,
        separators=hlt_strat,
        chunk_overlap=hlt_overlap,
    )
    sections = text_splitter.split_text(search_text)

    found_positions = {}
    for x in sections:
        text_start_pos = 0
        while text_start_pos != -1:
            text_start_pos = full_text.find(x, text_start_pos)
            if text_start_pos != -1:
                found_positions[text_start_pos] = text_start_pos + len(x)
                text_start_pos += 1

    # Combine overlapping or adjacent positions
    sorted_starts = sorted(found_positions.keys())
    combined_positions = []
    if sorted_starts:
        current_start, current_end = sorted_starts[0], found_positions[sorted_starts[0]]
        for start in sorted_starts[1:]:
            if start <= (current_end + 10):
                current_end = max(current_end, found_positions[start])
            else:
                combined_positions.append((current_start, current_end))
                current_start, current_end = start, found_positions[start]
        combined_positions.append((current_start, current_end))

    # Construct pos_tokens
    pos_tokens = []
    prev_end = 0
    for start, end in combined_positions:
        if end-start > 15: # Only combine if there is a significant amount of matched text. Avoids picking up single words like 'and' etc.
            pos_tokens.append(full_text[prev_end:start])
            pos_tokens.append('<mark style="color:black;">' + full_text[start:end] + '</mark>')
            prev_end = end
    pos_tokens.append(full_text[prev_end:])

    return "".join(pos_tokens)


# # Chat history functions

def clear_chat(chat_history_state, sources, chat_message, current_topic):
    chat_history_state = []
    sources = ''
    chat_message = ''
    current_topic = ''

    return chat_history_state, sources, chat_message, current_topic

def _get_chat_history(chat_history: List[Tuple[str, str]], max_memory_length:int = max_memory_length): # Limit to last x interactions only

    if (not chat_history) | (max_memory_length == 0):
        chat_history = []

    if len(chat_history) > max_memory_length:
        chat_history = chat_history[-max_memory_length:]
        
    #print(chat_history)

    first_q = ""
    first_ans = ""
    for human_s, ai_s in chat_history:
        first_q = human_s
        first_ans = ai_s

        #print("Text to keyword extract: " + first_q + " " + first_ans)
        break

    conversation = ""
    for human_s, ai_s in chat_history:
        human = f"Human: " + human_s
        ai = f"Assistant: " + ai_s
        conversation += "\n" + "\n".join([human, ai])

    return conversation, first_q, first_ans, max_memory_length

def add_inputs_answer_to_history(user_message, history, current_topic):
    
    if history is None:
        history = [("","")]

    #history.append((user_message, [-1]))

    chat_history_str, chat_history_first_q, chat_history_first_ans, max_memory_length = _get_chat_history(history)


    # Only get the keywords for the first question and response, or do it every time if over 'max_memory_length' responses in the conversation
    if (len(history) == 1) | (len(history) > max_memory_length):
        
        #print("History after appending is:")
        #print(history)

        first_q_and_first_ans = str(chat_history_first_q) + " " + str(chat_history_first_ans)
        #ner_memory = remove_q_ner_extractor(first_q_and_first_ans)
        keywords = keybert_keywords(first_q_and_first_ans, n = 8, kw_model=kw_model)
        #keywords.append(ner_memory)

        # Remove duplicate words while preserving order
        ordered_tokens = set()
        result = []
        for word in keywords:
                if word not in ordered_tokens:
                        ordered_tokens.add(word)
                        result.append(word)

        extracted_memory = ' '.join(result)

    else: extracted_memory=current_topic
    
    print("Extracted memory is:")
    print(extracted_memory)
    
    
    return history, extracted_memory

# Keyword functions

def remove_q_stopwords(question): # Remove stopwords from question. Not used at the moment 
    # Prepare keywords from question by removing stopwords
    text = question.lower()

    # Remove numbers
    text = re.sub('[0-9]', '', text)

    tokenizer = RegexpTokenizer(r'\w+')
    text_tokens = tokenizer.tokenize(text)
    #text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]

    # Remove duplicate words while preserving order
    ordered_tokens = set()
    result = []
    for word in tokens_without_sw:
        if word not in ordered_tokens:
            ordered_tokens.add(word)
            result.append(word)
     


    new_question_keywords = ' '.join(result)
    return new_question_keywords

def remove_q_ner_extractor(question):
    
    predict_out = ner_model.predict(question)



    predict_tokens = [' '.join(v for k, v in d.items() if k == 'span') for d in predict_out]

    # Remove duplicate words while preserving order
    ordered_tokens = set()
    result = []
    for word in predict_tokens:
        if word not in ordered_tokens:
            ordered_tokens.add(word)
            result.append(word)
     


    new_question_keywords = ' '.join(result).lower()
    return new_question_keywords

def apply_lemmatize(text, wnl=WordNetLemmatizer()):

    def prep_for_lemma(text):

        # Remove numbers
        text = re.sub('[0-9]', '', text)
        print(text)

        tokenizer = RegexpTokenizer(r'\w+')
        text_tokens = tokenizer.tokenize(text)
        #text_tokens = word_tokenize(text)

        return text_tokens

    tokens = prep_for_lemma(text)

    def lem_word(word):
    
        if len(word) > 3: out_word = wnl.lemmatize(word)
        else: out_word = word

        return out_word

    return [lem_word(token) for token in tokens]

def keybert_keywords(text, n, kw_model):
    tokens_lemma = apply_lemmatize(text)
    lemmatised_text = ' '.join(tokens_lemma)

    keywords_text = KeyBERT(model=kw_model).extract_keywords(lemmatised_text, stop_words='english', top_n=n, 
                                                   keyphrase_ngram_range=(1, 1))
    keywords_list = [item[0] for item in keywords_text]

    return keywords_list
    
# Gradio functions
def turn_off_interactivity(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

def restore_interactivity():
        return gr.update(interactive=True)

def update_message(dropdown_value):
        return gr.Textbox.update(value=dropdown_value)

def hide_block():
        return gr.Radio.update(visible=False)
    
# Vote function

def vote(data: gr.LikeData, chat_history, instruction_prompt_out, model_type):
    import os
    import pandas as pd

    chat_history_last = str(str(chat_history[-1][0]) + " - " + str(chat_history[-1][1]))

    response_df = pd.DataFrame(data={"thumbs_up":data.liked,
                                        "chosen_response":data.value,
                                          "input_prompt":instruction_prompt_out,
                                          "chat_history":chat_history_last,
                                          "model_type": model_type,
                                          "date_time": pd.Timestamp.now()}, index=[0])

    if data.liked:
        print("You upvoted this response: " + data.value)
        
        if os.path.isfile("thumbs_up_data.csv"):
             existing_thumbs_up_df = pd.read_csv("thumbs_up_data.csv")
             thumbs_up_df_concat = pd.concat([existing_thumbs_up_df, response_df], ignore_index=True).drop("Unnamed: 0",axis=1, errors="ignore")
             thumbs_up_df_concat.to_csv("thumbs_up_data.csv")
        else:
            response_df.to_csv("thumbs_up_data.csv")

    else:
        print("You downvoted this response: " + data.value)

        if os.path.isfile("thumbs_down_data.csv"):
             existing_thumbs_down_df = pd.read_csv("thumbs_down_data.csv")
             thumbs_down_df_concat = pd.concat([existing_thumbs_down_df, response_df], ignore_index=True).drop("Unnamed: 0",axis=1, errors="ignore")
             thumbs_down_df_concat.to_csv("thumbs_down_data.csv")
        else:
            response_df.to_csv("thumbs_down_data.csv")            
