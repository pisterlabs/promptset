#! /usr/bin/python3
import time
from flask import Flask, Response
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import Optional
from threading import Thread
import torch.cuda

# model_id = "garage-bAInd/Platypus2-7B"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

cache = {}


class CustomLLM(LLM):
    streamer: Optional[TextIteratorStreamer] = None
    history = []

    def _call(self, prompt, stop=None, run_manager=None) -> str:
        self.history = []
        self.question = ""
        self.streamer = TextIteratorStreamer(
            tokenizer=tokenizer, skip_prompt=True, timeout=5
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        kwargs = dict(
            input_ids=inputs.input_ids,
            max_new_tokens=500,
            streamer=self.streamer,
            pad_token_id=tokenizer.eos_token_id,
        )
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        return ""

    @property
    def _llm_type(self) -> str:
        return "custom"

    def stream_tokens(self):
        for token in self.streamer:
            time.sleep(0.05)
            cache[self.question] += token
            yield token

    def query_cache(self, question):
        """Check the cache for a response to the given question."""
        return cache.get(question)

    def update_cache(self, question, response):
        """Update the cache with the new question-response pair."""
        cache[question] = response

    def build_context(self):
        """Construct the context string from the cache."""
        context_parts = []
        for question, answer in cache.items():
            context_parts.append(f"Question: {question}\nAnswer: {answer}")
        return "\n".join(context_parts)


tokenizer.pad_token_id = model.config.eos_token_id

template = """You are a virtual assistant named Lumi.
{context}
Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)
llm = CustomLLM()
chain = prompt | llm

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "This is the base address for Lumi!"


@app.route("/query/<question>", methods=["GET"])
def query(question):
    print("Question asked: {}".format(question))
    context_string = llm.build_context()
    llm.question = question
    chain.invoke(input=dict({"context": context_string, "question": question}))
    return Response(llm.stream_tokens(), mimetype="text/plain")
