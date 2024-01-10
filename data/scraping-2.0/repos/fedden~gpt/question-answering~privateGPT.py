#!/usr/bin/env python3
import torch
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Prompt

import constants


def load_model(model_id: str):
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "10GB" for i in range(n_gpus)}
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        cache_dir="/shared/llm/huggingface",
        load_in_4bit=True,
        device_map="sequential",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=8192,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


def main():
    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)
    db = Chroma(
        persist_directory=constants.PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=constants.CHROMA_SETTINGS,
    )
    retriever = db.as_retriever(search_kwargs={"k": constants.TARGET_SOURCE_CHUNKS})
    llm = load_model(model_id="timdettmers/guanaco-33b-merged")
    if constants.USE_PROMPT:
        with open("prompt.txt", "r") as stream:
            prompt_template: str = stream.read()
            prompt = PromptTemplate(
                template=prompt_template, input_variables=["summaries", "question"]
            )
            chain_type_kwargs = {"prompt": prompt}
    else:
        chain_type_kwargs = None
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        chain_type="stuff",
        chain_type_kwargs=chain_type_kwargs,
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    # Interactive questions and answers
    console = Console()
    while True:
        console.clear()
        new_lines: str = "\n" * ((console.height // 2) - 1)
        question: str = Prompt.ask(f"{new_lines}  Ask anything about the DL team")
        # Get the answer from the chain
        res = qa({"question": question})
        answer, docs = res["answer"], res["source_documents"]
        qa_panel = Panel(f"[b]Question:[/b] {question}\n[yellow]{answer.rstrip()}")
        # Print the relevant sources used for the answer
        document_panels = []
        for document in docs:
            document_panels.append(
                Panel(
                    document.page_content,
                    title=(
                        f"[grey89][u][i]Source Document: "
                        f"{document.metadata['source']}[/i][/grey89][/u]"
                    ),
                )
            )
        layout = Layout(size=10)
        layout.split_row(
            qa_panel,
            Columns(document_panels),
        )
        console.clear()
        console.print(layout)
        input("")


if __name__ == "__main__":
    main()
