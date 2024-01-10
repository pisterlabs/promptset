import argparse
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from typing import List, Tuple, Union
from consts import CHROMA_SETTINGS, PERSIST_DIRECTORY

args = args.parse_args()

args.ArgumentParser()

args.add_argument("--source_dir", type=str, default=SOURCE_DIRECTORY)
args.add_argument("--persist_dir", type=str, default=PERSIST_DIRECTORY)
args.add_argument(
    "--device_type",
    type=str,
    default="cuda",
    choices=["cuda", "cpu", "hip", "xla", "ort", "tpu", "mkldnn"],
    help="The compute power that you have",
)


def load_model(device_type: str):
    """
    Select a model from huggingface.
    """

    model_id = "TheBloke/vicuna-7B-1.1-HF"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(model_id)

    logging.info(f"Loading model {model_id} on {device_type}")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    cage_llm = HuggingFacePipeline(pipeline=pipe)

    logging.info(f"cage_llm loaded on {device_type}")
    return cage_llm


def main():
    logging.info(f"Running on: {args.device_type}")

    embedding = HuggingFaceInstructEmbeddings(
        # "hkunlp/instruct-large" model_name is default for the embeddings in langchain
        model_name="hkunlp/instruct-large",
        model_kwargs={"device": args.device_type},
    )

    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding,
        client_settings=CHROMA_SETTINGS,
    )

    retriever = vectordb.as_retrieval()

    llm = load_model(args.device_type)

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    with open("user_input.log", "w") as file:
        while True:
            query = input("\nEnter a question: ")

            # Write the query to the log file
            file.write(query + "\n")

            if query == "quit":
                break

            # Get the answer from the QA
            res = qa(query)
            answer, docs = res["result"], res["source_documents"]

            # Print the answer
            print(f"\n\n > Question:")
            print(query)
            print(f"\n\n > Answer:")
            print(answer)

            ## Print the relevant sources used for the answer
            print("---------------------SOURCE DOCUMENTS---------------------")

            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)

            print("---------------------SOURCE DOCUMENT---------------------")


if __name__ == "__main__":
    main()
