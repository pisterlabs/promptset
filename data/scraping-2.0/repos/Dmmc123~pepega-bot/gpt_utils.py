import os
import openai
from sentence_transformers import SentenceTransformer
from transformers import GPT2TokenizerFast
from pynndescent import NNDescent
import pickle
import json


class AnswerGenerator:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_path="index.pkl",
        paragraphs_path="paragraphs.json",
        tokenizer_name="gpt2",
        completion_model="text-davinci-002",
    ):
        # initializing the paragraph/question encoder and tokenizer
        self.encoder = SentenceTransformer(model_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        # connecting to openai
        openai.api_key = os.environ["openai-api-token"]
        # loading the index and corresponding paragraphs
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(paragraphs_path) as f:
            self.paragraphs = json.load(f)
        # storing info prompt construction
        self.prompt_params = {"max_section_len": 500, "sep": "\n* "}
        self.prompt_params["sep_len"] = len(
            self.tokenizer.tokenize(self.prompt_params["sep"])
        )
        # parameters for querying the gpt
        self.gpt_params = {
            "temperature": 0.0,
            "max_tokens": 300,
            "model": completion_model,
        }

    def _prepare_prompt(self, question):
        # encode the question
        q_emb = self.encoder.encode(question)
        # get indices of top matches for paragraphs
        most_relevant_document_sections = self.index.query([q_emb])[0][0]
        # add contexts until we run out of space.
        chosen_sections = []
        chosen_sections_len = 0
        for section_index in most_relevant_document_sections:
            paragraph = self.paragraphs[section_index]
            # update current length
            chosen_sections_len += (
                len(self.tokenizer(paragraph)) + self.prompt_params["sep_len"]
            )
            if chosen_sections_len > self.prompt_params["max_section_len"]:
                break
            # add section to prompt context
            chosen_sections.append(
                self.prompt_params["sep"] + paragraph.replace("\n", " ")
            )
        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

    def __call__(self, question):
        # get prompt
        prompt = self._prepare_prompt(question)
        # query the qpt
        response = openai.Completion.create(prompt=prompt, **self.gpt_params)
        # return the needed part of response
        return response["choices"][0]["text"].strip(" \n")
