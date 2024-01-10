import torch, transformers, os, json, textwrap, pickle, faiss, textwrap
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def store_embeddings(docs, embeddings, store_name, path):
    vectorStore = FAISS.from_documents(docs, embeddings)
    with open(f"{path}/faiss_{store_name}.pkl", "wb") as f:
        pickle.dump(vectorStore, f)

def load_embeddings(store_name, path):
    with open(f"{path}/faiss_{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore

def get_prompt(human_prompt):
    prompt_template=f"### Human: {human_prompt} \n### Assistant:"
    return prompt_template

def remove_human_text(text):
    return text.split('### Human:', 1)[0]

def parse_text(data):
    for item in data:
        text = item['generated_text']
        assistant_text_index = text.find('### Assistant:')
        if assistant_text_index != -1:
            assistant_text = text[assistant_text_index+len('### Assistant:'):].strip()
            assistant_text = remove_human_text(assistant_text)
            wrapped_text = textwrap.fill(assistant_text, width=100)
            print(wrapped_text)

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


def main():
    parser = argparse.ArgumentParser(description='Final Project: Approach')
    parser.add_argument('--config_path', type=str, default='config_scenario2.json', help='path to the config.json file')

    # variable parser explanation
    args = parser.parse_args()

    config_path = args.config_path
    config = load_config(config_path)

    # Extract parameters from the config dictionary
    pdf_param = config['pdf_param']
    folder_param = config['folder_param']
    db_param = config['db_param']
    save_param = config['save_param']
    checkpoint_loader_param = config['checkpoint_loader_param']
    model_name_param = config['model_name_param']
    chain_type_param = config['chain_type_param']
    device_param = config['device_param']
    huggingface_token_param = config['huggingface_token_param']
    openapi_token_param = config['openapi_token_param']
    max_length_param = config['max_length_param']
    pad_token_id_param = config['pad_token_id_param']
    temperature_param = config['temperature_param']
    top_p_param = config['top_p_param']
    repetation_penalty_param = config['repetation_penalty_param']
    k_param = config['k_param']
    chunk_size_param = config['chunk_size_param']
    chunk_overlap_param = config['chunk_overlap_param']
    queries = config['queries']

    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_token_param
    os.environ["OPENAI_API_KEY"] = openapi_token_param

    # Load and process the text files
    if load_one_file_active == True:
       loader = TextLoader(pdf_param)
    if load_one_folder_active == True:
       loader = DirectoryLoader(folder_param, glob="./*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()
    print("length of Documents: {}".format(len(documents)))

    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size_param, chunk_overlap= chunk_overlap_param)
    texts = text_splitter.split_documents(documents)
    print("length texts: {}".format(len(texts)))

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name= model_name_param,
                                                          model_kwargs={"device": device_param})

    if local_llm_active == True:
      tokenizer = AutoTokenizer.from_pretrained(checkpoint)
      base_model = AutoModelForCausalLM.from_pretrained(checkpoint_loader_param,
                                                        device_map='auto',
                                                        torch_dtype=torch.float16,
                                                        load_in_8bit=True)

      pipe = pipeline('text-generation',
                      model = base_model,
                      tokenizer = tokenizer,
                      max_length= max_length_param,
                      do_sample=True,
                      pad_token_id= pad_token_id_param,
                      temperature= temperature_param,
                      top_p= top_p_param,
                      repetition_penalty= repetation_penalty_param
                      )

      local_llm = HuggingFacePipeline(pipeline=pipe)
      print(local_llm(queries[0][1]))

    if openapi_llm_active == True:
      local_llm = OpenAI(temperature= temperature_param,)
      print(local_llm(queries[0][1]))


    if method_param == 'faiss_instructor':
        ##method 1: using faiss embedding store
        Embedding_store_path = folder_param
        store_embeddings(texts,
                    instructor_embeddings,
                    store_name='instructEmbeddings',
                    path=Embedding_store_path)

        db_instructEmbedd = load_embeddings(store_name='instructEmbeddings',
                                        path=Embedding_store_path)

        retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": k_param})

        # create the chain to answer questions
        qa_chain_instruction = RetrievalQA.from_chain_type(llm=OpenAI(temperature= temperature_param,),
                                                          chain_type= chain_type_param,
                                                          retriever=retriever,
                                                          return_source_documents=True)


    elif method_param == 'chroma_instructor':
       ## Embed and store the texts
       persist_directory = db_param ## Supplying a persist_directory will store the embeddings on disk
       ##method 2 {secodn scenario}:
       vectordb = Chroma.from_documents(documents=texts,
                                        embedding=instructor_embeddings,
                                        persist_directory=persist_directory)

       retriever = vectordb.as_retriever(search_kwargs={"k": k_param})

       # create the chain to answer questions
       qa_chain_instruction = RetrievalQA.from_chain_type(llm=OpenAI(temperature= temperature_param,),
                                                        chain_type= chain_type_param,
                                                        retriever=retriever,
                                                        return_source_documents=True)

    #example
    #docs = retriever.get_relevant_documents(queries[0][1])
    #print(docs[0])

    elif method_param == 'openai':
      #method 3 {third scenario}:
      store_embeddings(texts,
                       embeddings= OpenAIEmbeddings(),
                       store_name= 'openAIEmbeddings',
                       path=Embedding_store_path)

      db_openAIEmbedd = load_embeddings(store_name='openAIEmbeddings',
                                         path=Embedding_store_path)

      retriever_openai = db_openAIEmbedd.as_retriever(search_kwargs={"k": k_param})

      # create the chain to answer questions
      qa_chain_openai = RetrievalQA.from_chain_type(llm=OpenAI(temperature= temperature_param, ),
                                                    chain_type= chain_type_param,
                                                    retriever= retriever_openai,
                                                    return_source_documents=True)

    # print(get_prompt(queries[0][1]))

    # data = [{'generated_text': '### Human: What is the capital of England? \n### Assistant: The capital city of England is London.'}]
    # parse_text(data)

    # qa_chain.retriever.search_type , qa_chain.retriever.vectorstore
    # print(qa_chain.combine_documents_chain.llm_chain.prompt.template)

    # qa_chain.combine_documents_chain.llm_chain.prompt.template ='''### Human: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # {context}
    # Question:  {question}
    # \n### Assistant:'''

    responses = []

    for query in queries:
        print('-------------------Instructor Embeddings------------------\n')
        if method2 == instruction_chain:
          llm_response = qa_chain_instruction(query)
        if method2 == openai_chain:
          llm_response = qa_chain_openai(query)

        response = process_llm_response(llm_response)
        responses.append(response)
        print(response)

    # Sample string
    data = {
      "author": response[0],
      "title": response[1],
      "Theoretical/ Conceptual Framework": response[2],
      "Research Question(s)/ Hypotheses": response[3],
      "methodology": response[4],
      "Analysis & Results study": response[5],
      "conclusion": response[6],
      "Implications for Future research": response[7],
      "Implication for practice": response[8],
    }

    # Save the JSON object to a file
    with open(save_param, 'w') as f:
        json.dump(data, f)



if __name__ == "__main__":
   main()
