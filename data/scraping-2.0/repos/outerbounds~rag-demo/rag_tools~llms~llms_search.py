import openai
from transformers import AutoTokenizer
import transformers
import torch

class Llama2ChatWrapper:

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf"):
        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def sample(self, text, max_tokens=200):
        sequences = self.pipeline(
            f'{text}\n',
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=max_tokens,
        )

        # return the first sequence
        for seq in sequences:
            return seq['generated_text']


class ChatGPTWrapper:
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        """
        Currently only handles single message sessions.
        """
        self.model_name = model_name
        openai.api_key = api_key

    def sample(self, text, temperature=0.8, max_tokens=100):
        metadata = {"model": self.model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens}

        # sample from the openai model

        r = openai.ChatCompletion.create(
            messages=[
                {"role": "system", "content": "You are a search engine that replies to query based on context. Answer with "
                                              "information provided within the context otherwise reply with 'I don't know'"},
                {"role": "user", "content": text}
            ],
            **metadata
        )
        return r['choices'][0]["message"]["content"]
