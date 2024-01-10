########################### IMPORTS ###########################
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFium2Loader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from peft import PeftModel, PeftConfig, LoraConfig
import matplotlib.pyplot as plt
import pandas as pd

########################### INPUT ###########################
def input_parser():
    print("Parsing input arguments...")
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-t",
                    "--task",
                    help="The task to run. Should be the name of a folder in the data/benchmark_data/ folder. Default is 'single_paper'.",
                    type = str, default="single_paper")
    ap.add_argument("-id",
                    "--save_id",
                    help="The ID to use when saving the results. Default is 'saiphipaca'.",
                    type = str, default="saiphipaca")
    ap.add_argument("-phipaca",
                    "--include_phipaca",
                    help="Include the phipaca adapter. Either 'yes' or 'no'. Default is 'yes'.",
                    type = str, default="yes")
    ap.add_argument("-saiphipaca",
                    "--include_saiphipaca",
                    help="Include the saiphipaca adapter. Either 'yes' or 'no'. Default is 'yes'.",
                    type = str, default="yes")
    args = ap.parse_args() # Parse the args
    return args

########################### LOAD MODELS ###########################
# Load the model and tokenizer
def load_model_and_tokenizer(phipaca_arg, saiphipaca_arg):
    print("Loading model and tokenizer...")

    # See if this makes a difference?
    print("Loading the base model...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

    #config = PeftConfig.from_pretrained(adapter)
    #model = PeftModel.from_pretrained(base_model, adapter, config=config)
    print("Adding the adapter(s)...")
    if phipaca_arg == "yes":
        model.load_adapter("alekswael/phipaca", adapter_name="phipaca")
        print("Phipaca adapter loaded.")
    
    if saiphipaca_arg == "yes":
        model.load_adapter("alekswael/saiphipaca", adapter_name="saiphipaca")
        print("Saiphipaca adapter loaded.")

    ########## EXPERIMENTAL IMPLEMENTATIONS ##########
    # peft_config = PeftConfig.from_pretrained(adapter)
    # model.add_adapter(peft_config, adapter_name="phipaca")
    # model.set_adapter("phipaca")
    # model.enable_adapters()

    #phipaca = PeftModel.from_pretrained(model, adapter)
    #print("Merging and unloading...")
    #model = phipaca.merge_and_unload()
    #model.merge_adapter()
    #model = phipaca

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    return model, tokenizer

# Load the embeddings
def load_embeddings():
    print("Loading embeddings...")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE})
    return embeddings

########################### LOAD DATA ###########################
def load_data(task_arg, tokenizer):
    print("Loading data...")
    # Load the document
    loader = PyPDFium2Loader(f"data/benchmark_data/{task_arg}/{task_arg}.pdf")
    document = loader.load()
    # Split the document into chunks
    print("Splitting the document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, # Max input length for the embedding model is 512, so each context chunk should be less than this. Max input length for LLM is 2048, so context chunks + question should be less than this.
        chunk_overlap=50,
        length_function=lambda x: len(tokenizer.tokenize(x))
        )
    texts = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(texts)}")
    return texts

def plot_chunks(texts, tokenizer, task_arg, id_arg):
    # Histogram of the character counts per chunk
    print("Plotting histogram of token counts per chunk...")
    token_data = []
    for x in texts:
        token_count = len(tokenizer.tokenize(x.page_content))
        token_data.append(token_count)
    plt.hist(token_data)
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.title('Histogram of token counts per chunk')
    plt.ylim(0, len(texts))
    plt.savefig(f'./results/chunk_histograms/{task_arg}_histogram_{id_arg}.png')


########################### UPLOAD TO DB ###########################
def upload_to_db(texts, embeddings):
    print("Uploading data to the Chroma database...")
    db_start = time.time()
    # Upload the data to the database
    db = Chroma.from_documents(texts, embeddings)
    db_end = time.time()
    db_runtime = db_end - db_start
    print(f"Database runtime: {db_runtime // 3600} hours {(db_runtime % 3600) // 60} minutes {db_runtime % 60} seconds")
    return db

########################### RETRIEVAL ###########################
def pipeline_setup(tokenizer, model):
    print("Setting up the pipeline...")
    # Text streamer
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)
    # Text pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=False,
        #top_p=0.95,
        repetition_penalty=1.15,
        streamer=streamer,
    )
    # Language model pipeline
    cllm = HuggingFacePipeline(pipeline=text_pipeline)
    return cllm

########################### QA ###########################
def prompt_setup():
    print("Setting up the prompt...")
    # Prompt
    template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n
    ### Instruction: Your role is to answer questions about text excerpts from an academic paper. You will provide clear, accurate explanations and insights related to the content, methodology, results, and implications of the paper. Avoid speculation and stick to the information presented in the paper itself. If a question falls outside the scope of the paper or requires clarification, ask for more details or indicate the limitations of your responses. Your responses should be informative, precise, and tailored to the user's level of understanding. Use the following pieces of context to answer the question at the end.\n\n
    Text: \n
    {context}\n
    ### Input: {question}\n
    ### Response: """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def chain_setup(cllm, db, prompt):
    print("Setting up the QA chain...")
    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=cllm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )
    return qa_chain

########################### OUTPUT ###########################
def get_responses(task_arg, id_arg, qa_chain):
    print("Loading the questions...")
    # Load in the questions from the benchmark data csv file
    df = pd.read_csv(f"data/benchmark_data/{task_arg}/{task_arg}_data.csv", delimiter=";")

    # Loop through the quesionts and generate the responses, and append them to the dataframe
    print("Generating the responses...")

    responses = []
    inference_speed = []
    context_chunks = {}

    for i in range(len(df)):
        # Inference speed test. Start time here
        inf_start = time.time()
        # QA Chain
        response = qa_chain(df["question"][i])
        # Inference speed test. End time here
        inf_end = time.time()
        inf_speed = inf_end - inf_start
        print(f"Inference speed: {inf_speed // 3600} hours {(inf_speed % 3600) // 60} minutes {inf_speed % 60} seconds")
        inference_speed.append(inf_speed)
        # Append the output
        result = response["result"]
        responses.append(result)
        # Append the context chunks
        print("Appending chunks...")
        for c in range(len(response["source_documents"])):
            chunk = response["source_documents"][c].page_content
            # Use the loop variable to create a key for the dictionary
            key = f"context_chunk_{c+1}"
            # If the key is not in the dictionary, add it with an empty list as the value
            if key not in context_chunks:
                context_chunks[key] = []
            # Append the chunk to the list
            context_chunks[key].append(chunk)
            print(f"Question {i+1} of {len(df)}: Chunk {c+1} of {len(response['source_documents'])}")

    # Add the responses to the dataframe
    print("Adding the responses to the dataframe...")
    df["model_response"] = pd.Series(responses)

    # Add the chunks to the dataframe
    print("Adding the chunks to the dataframe...")
    for key, value in context_chunks.items():
        df[key] = pd.Series(value)
    
    # Add the inference speed to the dataframe
    print("Adding the inference speed to the dataframe...")
    df["inference_speed_seconds"] = pd.Series(inference_speed)

    # Save the dataframe to a csv file
    print("Saving the dataframe to a csv file...")
    df.to_csv(f"results/{task_arg}_results_{id_arg}.csv", index=False)

def main():
    start_time = time.time()

    # Parse the arguments
    args = input_parser()
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.include_phipaca, args.include_saiphipaca)
    # Load the embeddings
    embeddings = load_embeddings()
    # Load the data
    texts = load_data(args.task, tokenizer)
    # Plot the chunks
    plot_chunks(texts, tokenizer, args.task, args.save_id)
    # Upload the data to the database
    db = upload_to_db(texts, embeddings)
    # Setup the pipeline
    cllm = pipeline_setup(tokenizer, model)
    # Setup the prompt
    prompt = prompt_setup()
    # Setup the QA chain
    qa_chain = chain_setup(cllm, db, prompt)
    # Get the responses
    get_responses(args.task, args.save_id, qa_chain)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime // 3600} hours {(runtime % 3600) // 60} minutes {runtime % 60} seconds")

if __name__ == "__main__":
    main()
