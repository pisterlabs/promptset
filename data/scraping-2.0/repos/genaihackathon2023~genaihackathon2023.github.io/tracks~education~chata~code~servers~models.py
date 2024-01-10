import os
import openai
from utils import write_to_json
from dotenv import load_dotenv
import torch
from utils import QA_Data
# from llama import Llama
from transformers import GPT2LMHeadModel, GPT2Tokenizer

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") # GPT4 & 3.5
openai.organization = os.getenv("OPENAI_ORG_ID") # Organization: cmuwine
qastore = QA_Data()

class Llama():
    def __init__(self, ckpt_dir="", tokenizer_path="", max_seq_len=500, max_batch_size=500):
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            )

    def get_response(self, message):
        results = self.generator.text_completion(
            [message],
            max_gen_len=500,
            temperature=0.8,
            top_p=0.9,
        )
        write_to_json(results[0], file='llama2_response.json')
        return results[0]['generation']

class GPT2():
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def get_response(self, message):
        message = "Q: " + message + "\nA: "
        inputs = self.tokenizer.encode(message, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        write_to_json(response, file='gpt2_response.json')
        return response.split("\nA: ")[1]

class GPT3():
    def __init__(self, gpt="gpt-3.5-turbo", temperature=0.2):
        self.gpt = gpt
        self.temperature = temperature
        self.fix_prompt = []

    def format_qa(self, q, a):
        user = {"role": "user", "content": f"Question: {q}."}
        assistant = {"role": "assistant", "content": f"Answer: {a}."}
        return [user, assistant]

    def format_gpt_prompt(self, q):
        system = {"role": "system", "content": "You are a helpful and experienced teaching assistant of a computer science course."}
        user = {"role": "user", "content": f"Question: {q}."}
        return [system] + self.fix_prompt + [user]

    def ask_gpt(self, q):
        retrieved_answer = qastore.retrieve_answer(q)
        if retrieved_answer is not None:
            print("retrieved answer from db")
            return retrieved_answer
        
        prompt = self.format_gpt_prompt(q)
        response = openai.ChatCompletion.create(
            model=self.gpt,
            temperature=self.temperature,
            messages=prompt
        )
        gpt_answer = response['choices'][0]['message']['content']
        response["prompt"] = prompt
        write_to_json(response, file='gpt3_response.json')
        return gpt_answer

