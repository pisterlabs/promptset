import os
import openai
import tiktoken  # OpenAI's fast BPE Tokenizer
from .base_ml_model import BaseMLModel

openai.api_key = os.environ.get('OPENAI_API_KEY')


class SummaryOpenAI(BaseMLModel):
    """# Model that uses OpenAI API call to pick  most important sentences in text"""

    def __init__(self):
        self.name = "Sum-OpenAI-GPT35t"
        # the gpt-3.5-turbo model is a good compromise between speed and quality in my tests (202303)
        # see https://platform.openai.com/docs/models/gpt-3.5 for other options
        self.OPENAI_MODEL = "gpt-3.5-turbo"
        self.TOP_SENTENCES = 4
        self.SUMMARY_LEN = 200
        super().__init__()


    def process(self, input_text, abstractive=False):
        # first, truncate to the max token length of this API
        trunc_text = self.truncate_text(input_text, 4096-self.SUMMARY_LEN-50)

        # GPT prompt. note that the 'without..., without.., without ...'  are key instructions
        if not abstractive:
            prompt = f"Extract the most {self.TOP_SENTENCES} important sentences from the following text without altering " \
                     f"them, without giving unnecessary weight to the start of the text, and without repeating any " \
                     f"information: \"{trunc_text}\""
        else:
            prompt = f"Please summarize the following text in a few sentence: \"{trunc_text}\""

        response = openai.ChatCompletion.create(
            model=self.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.SUMMARY_LEN,
            n=1,
            stop=None,
            temperature=0.5,
        )
        output = response.choices[0].message['content'].strip()
        if not abstractive:
            sents = []
            for s in output.splitlines():
                if s.startswith("1. ") or s.startswith("2. ") or s.startswith("3. ") or s.startswith("4. "):
                    s = s[3:]
                sents.append(s)
            output = sents
        return output

    # Truncate the text if it exceeds the token limit
    def truncate_text(self, text, max_tokens):
        enc = tiktoken.encoding_for_model(self.OPENAI_MODEL)
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            truncated_text = enc.decode(tokens)
            return truncated_text
        return text
