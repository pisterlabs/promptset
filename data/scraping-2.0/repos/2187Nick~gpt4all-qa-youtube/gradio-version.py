import gradio as gr
from pyllamacpp.model import Model
from langchain.llms import GPT4All
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# enter the location of your model
select_model = './models/gpt4all-converted.bin'
# set the length of the response you want
results_length = 30
# increase this number to get more random results
model_temp = 0.3

model = Model(ggml_model=select_model)

def youtube_video_to_text(Youtube_Link, Question):
    youtube_link = Youtube_Link
    loader = YoutubeLoader.from_youtube_channel(Youtube_Link, add_video_info=True)

    documents = loader.load()
    title = documents[0].metadata['title']
    print(documents[0].metadata['title'])

    # chromadb can give error with publish date. So deleting it
    del documents[0].metadata['publish_date']

    # remove special characters
    k = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";
    title = title.replace(" ", "_") 
    getVals = list(filter(lambda x: x in k, title))
    title = "".join(getVals)

    # set the directory to store the vectorized database
    persist_directory = title

    # select an embeddings transformer  # list of models and their speeds : https://www.sbert.net/docs/pretrained_models.html
    #model_name = "sentence-transformers/all-mpnet-base-v2"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)


    # if the video has already been processed then don't process it again
    if not os.path.exists(persist_directory):

        # Split the documents into chunks of 100 characters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )

        documents = text_splitter.split_documents(documents)
        db_vector = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
        
    else:
        db_vector = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    llm = GPT4All(model=select_model, n_predict=30, temp=0.3) # n_ctx=512, n_threads=8, n_predict=100, temp=.8

    retriever = db_vector.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    parse_answer = lambda x: x["result"].split("Answer:")[1].strip()

    # Query the model directly
    general_answer = model.generate(Question, n_predict=results_length, temp=model_temp)
    print("General Answer: ", general_answer)

    # Query the model from the youtube video data
    result_raw = qa(Question, return_only_outputs=True)
    result = parse_answer(result_raw)
    print("Youtube Sourced Answer: ", result)

    return result, general_answer

def query_both(Youtube_Link, Question):
    result, general_answer = youtube_video_to_text(Youtube_Link, Question)
    
    return result, general_answer


demo = gr.Interface(
    fn=query_both,
    inputs=["text", "text"],
    outputs=["text", "text"],
) 
 
demo.launch()
