#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time
import os

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Create Empty files
    createEmptyFiles()
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    
    query = "Find my matching experience for " + input("\nEnter the job name: ") + "from the text files"
    # Get the answer from the chain
    start = time.time()
    res = qa(query)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']
    end = time.time()

    # Print the result
    print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)

    """
    # write to a text file
    files = [doc.metadata["source"] for doc in docs]
    files = set(files)
    data = ""
    for fi in files:
        with open(fi) as fp:
            data += fp.read() + "\n"
    with open("result.txt", "w") as f:
        f.write(data)
    """
    
    # write to a latex file
    files = [doc.metadata["source"] for doc in docs]
    files = set(files)
    e = 1
    p = 1
    for fi in files:
        print(fi)
        if 'E' in fi:
            txtToLatex(fi, "resumeFolder/src/Experience" + str(e) + ".tex")
            e += 1
        if 'P' in fi:
            txtToLatex(fi, "resumeFolder/src/Project" + str(p) + ".tex")
            p += 1
    
    
    # Print the relevant sources used for the answer
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

def createEmptyFiles():
    with open(os.path.join("resumeFolder/src/", "Experience1.tex"), 'w') as fp:
        fp.write("% Empty file")
    with open(os.path.join("resumeFolder/src/", "Experience2.tex"), 'w') as fp:
        fp.write("% Empty file")
    with open(os.path.join("resumeFolder/src/", "Project1.tex"), 'w') as fp:
        fp.write("% Empty file")
    with open(os.path.join("resumeFolder/src/", "Project2.tex"), 'w') as fp:
        fp.write("% Empty file")

def txtToLatex(filename, outputName):
    with open(filename) as f:
        lines = [line.rstrip('\n') for line in f]
    with open(outputName, "w") as f:
        f.write("\\resumeSubheading\n")
        f.write("   {" + lines[0] + "}" + "{" + lines[1] + "}\n")
        f.write("   {" + lines[2] + "}" + "{" + lines[3] + "}\n")
        f.write("   \\resumeItemListStart\n")
        for i in range(4, len(lines)):
            f.write("       \\resumeItem{" + lines[i] + "}\n")
        f.write("   \\resumeItemListEnd\n")


if __name__ == "__main__":
    main()
