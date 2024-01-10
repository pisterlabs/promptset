import os
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig
from langchain import LLMChain, PromptTemplate
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import (RetrievalQA)

import chromadb


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]


def format_prompt(system_prompt: str, user_prompt: str):
    """format prompt based on: https://huggingface.co/spaces/mosaicml/mpt-30b-chat/blob/main/app.py"""

    system_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    user_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    assistant_prompt = f"<|im_start|>assistant\n"

    return f"{system_prompt}{user_prompt}{assistant_prompt}"


def generate(
    llm: AutoModelForCausalLM,
    generation_config: GenerationConfig,
    system_prompt: str,
    user_prompt: str,
):
    """run model inference, will return a Generator if streaming is true"""

    return llm(
        format_prompt(
            system_prompt,
            user_prompt,
        ),
        **asdict(generation_config),
    )


if __name__ == "__main__":
    print("loading mpt config")
    config = AutoConfig.from_pretrained("mosaicml/mpt-30b-chat", context_length=8192)
    print("loading pretrained quanitized model")
    llm = AutoModelForCausalLM.from_pretrained(
        os.path.abspath("./models/mpt-30b-chat.ggmlv0.q4_1.bin"),
        model_type="mpt",
        config=config,
    )

    system_prompt = "A conversation between a user and an LLM-based AI assistant named Local Assistant. Local Assistant gives helpful and honest answers."
    print("creating generation config")
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        max_new_tokens=512,  # adjust as needed
        seed=42,
        reset=False,  # reset history (cache)
        stream=True,  # streaming per word/token
        threads=int(os.cpu_count() / 2),  # adjust for your CPU
        stop=["<|im_end|>", "|<"],
    )
    print("Setting up prompt")
    user_prefix = "[user]: "
    assistant_prefix = f"[assistant]:"

    system_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    context_prompt = "<|im_start|>context\n{context}<|im_end|>\n"
    user_prompt = "<|im_start|>user\n{question}<|im_end|>\n"
    assistant_prompt = f"<|im_start|>assistant\n"

    promptTemplate = f"{system_prompt}{context_prompt}{user_prompt}{assistant_prompt}"
    # template for an instruction with no input
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=promptTemplate
    )
    print("loading langchain things")
    
    collection_name = "open-amt-cloud-toolkit"
    local_directory = "kb-oact"
    persist_directory = os.path.join(os.getcwd(), local_directory)
    embedding = OpenAIEmbeddings()
    # get an existing collection
    print("loading docs collection")
    client = chromadb.PersistentClient(path="./kb-oact")
    kb_db = Chroma(client=client, collection_name=collection_name, persist_directory=persist_directory,
                embedding_function=embedding)
    print("setting retriever settings")
    retriever = kb_db.as_retriever()
    retriever.search_kwargs = {"k": 5}
    chain_type_kwargs = {"prompt": prompt}
    #llm = CTransformers(model="./models/mpt-30b-chat.ggmlv0.q4_1.bin",config=asdict(generation_config),model_type="mpt",callbacks=[StreamingStdOutCallbackHandler()])
    print("creating chain")
    #llm_chain = LLMChain(prompt=prompt, llm=llm)
    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       chain_type_kwargs=chain_type_kwargs,
                                       return_source_documents=True
                                       )
    print("running chain")
    response = qa_chain({"query": "What is the difference between CCM and ACM?"})
    print(response)

    # while True:
    #     user_prompt = input(user_prefix)
    #     generator = generate(llm, generation_config, system_prompt, user_prompt.strip())
    #     print(assistant_prefix, end=" ", flush=True)
    #     for word in generator:
    #         print(word, end="", flush=True)
    #     print("")
