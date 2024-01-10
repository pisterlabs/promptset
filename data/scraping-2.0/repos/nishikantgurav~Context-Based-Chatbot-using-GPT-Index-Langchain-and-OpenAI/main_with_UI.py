from flask import Flask, render_template, request,jsonify
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
import os

app = Flask(__name__)


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    # index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir='NEW_ChatBot_with_GPT_INDEX_AND _LANGCHAIN')

    # ChatBot_with_GPT_INDEX_AND _LANGCHAIN

    # index.save_to_disk('index.json')

    return index

'''
def ask_ai(query):
    # ... same code as you provided, but replace the `input()` function with the `query` parameter ...
    response = query_engine.query(query)
    return response.response
'''


def ask_ai(query):
    try:
        # index = GPTSimpleVectorIndex.load_from_disk('index.json')
        storage_context = StorageContext.from_defaults(persist_dir='NEW_ChatBot_with_GPT_INDEX_AND _LANGCHAIN')
        index = load_index_from_storage(storage_context)

        while True:
            #query = input("What do you want to ask? ")
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            return response.response

            # response = index.query(query)
            #display(Markdown(f"Response: <b>{response.response}</b>"))
    except Exception as e:
        print(f"An error occurred: {e}")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        response = ask_ai(query)
        return jsonify({'response': response})
    else:
        return render_template('index.html')
'''    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        response = ask_ai(query)
        return render_template('index.html', response=response)
    else:
        return render_template('index.html')
'''

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = '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'
    construct_index("context_data/data")
    app.run(debug=True)
