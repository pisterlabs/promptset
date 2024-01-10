# AI_Model.py

import os
import openai
import re
import spacy
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class AIModel:
    def __init__(self, model):
        self.model = model

    def __init__(self, model_engine="text-davinci-003"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_engine = model_engine
        self.tokenizer = openai.Tokenizer(engine=model_engine)
        self.nlp = spacy.load("en_core_web_sm")

    def prepare_prompt(self, prompt):
        # Add a prefix for context
        prompt = Config.PROMPT_PREFIX + prompt

        # Tokenize the prompt
        tokenized_prompt = self.tokenizer.tokenize(prompt)

        # Truncate if necessary
        if len(tokenized_prompt) > Config.MAX_LENGTH:
            tokenized_prompt = tokenized_prompt[-Config.MAX_LENGTH:]

        # Convert to a batch
        batched_prompt = [tokenized_prompt]

        return batched_prompt

    def generate_response(self, batched_prompt):
        try:
            response = openai.Completion.create(
                engine=self.model_engine,
                prompt=batched_prompt,
                max_tokens=Config.MAX_LENGTH,
                n=1,
                stop=None,
                temperature=0.5,
            )

            response_text = response.choices[0].text.strip()
        except Exception as e:
            print(f"Error: {e}")
            return None

        return response_text

    def postprocess_response(self, response_text):
        # Detokenize
        detokenized_response = self.tokenizer.detokenize(response_text)

        # Capitalize the first letter after a newline or period
        detokenized_response = re.sub(r"(\.|\n)\s*([a-zA-Z])", lambda pat: pat.group(1) + " " + pat.group(2).upper(), detokenized_response)

        # Add a period at the end if the response ends without [.?!], and replace ending comma with a period
        if detokenized_response[-1] not in ".!?":
            detokenized_response = detokenized_response[:-1] + "." if detokenized_response[-1] == "," else detokenized_response + "."

        # Replace multiple spaces/newlines with singular versions
        detokenized_response = re.sub(r"\n+", "\n", detokenized_response)
        detokenized_response = re.sub(r" +", " ", detokenized_response)

        # Capitalize proper nouns using named entity recognition
        doc = self.nlp(detokenized_response)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                detokenized_response = re.sub(rf"\b{ent.text}\b", ent.text.capitalize(), detokenized_response)

        return detokenized_response

    def generate_response(self, prompt):
        # Generate a response given a prompt
        # This function will need to be fleshed out with the specifics of how the AI model generates a response.
        # This could involve pre-processing the prompt, feeding the prompt to the model, post-processing the model's output, etc.
        # Since this is highly dependent on the specifics of the AI model being used, it's left as a placeholder here.
        return self.model.generate(prompt)

class Config:
    MAX_LENGTH = 1024
    PROMPT_PREFIX = "Please write a response to the following: "

