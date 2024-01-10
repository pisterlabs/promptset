# Created by scalers.ai for Dell Inc
import os

import torch
from fastapi import FastAPI, Response, status
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from optimum.intel import OVModelForCausalLM
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

app = FastAPI()


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def get_docsearch():
    """Return text embeddings of context."""
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma(
        persist_directory="./db/", embedding_function=embeddings
    )
    return docsearch


def create_hf_pipeline(model_name):
    """Create HF pipeline for text generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = OVModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        top_p=0.95,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )

    # # Create a custom llm using HF pipeline wrapper
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    return hf_pipeline


def create_prompt():
    """Prompt template for LLM."""

    prompt_template = """<s><<SYS>>
    You are a chatbot having a conversation with a human in drive-thru pickup area of a pharmacy.

    Given the following extracted parts of a long document, chat history and a question, just answer the users question.
    First ask for order number, date of birth and then check whether payment is done or not for confirmation.<</SYS>>

    {context}

    {chat_history}
    [INST]Question: {human_input}[/INST]
    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "human_input", "context"],
    )

    return prompt


def create_qa():
    """Create question answering chain."""
    memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", input_key="human_input"
    )
    qa = load_qa_chain(
        llm=HF_PIPELINE,
        chain_type="stuff",
        memory=memory,
        prompt=PROMPT,
    )
    return qa


@app.get("/generate", status_code=200)
def process_query(query: str, http_response: Response):
    """Predic api call for text generation."""
    if qa:
        docs = docsearch.similarity_search(query)
        answer = qa(
            {"input_documents": docs, "human_input": query},
            return_only_outputs=True,
        )
        print(answer)
        return str(answer["output_text"])
    else:
        http_response.status_code = status.HTTP_400_BAD_REQUEST
        response = {"response": "Context not set"}
        return response


@app.get("/health")
async def health_check():
    """Health checkup API."""
    return {"status": "healthy"}


MODEL_NAME = "llama-2-7b-chat-ov"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 250))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.2))
PROMPT = create_prompt()
HF_PIPELINE = create_hf_pipeline(MODEL_NAME)
qa = create_qa()
docsearch = get_docsearch()
