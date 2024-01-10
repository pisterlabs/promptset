import os, math, logging
import openai
from util import nlp


class GPT3Wrapper:

    def __init__(self, model_name: str, api_key: str, max_return_tokens=1024, temperature=0.5):
        openai.api_key = api_key
        self._model_name = model_name
        self._max_return_tokens = max_return_tokens
        self._temperature = temperature
        self._text_len_limit = self._max_return_tokens * 3
        self._nlputil = nlp.NLPUtil()

    def summarise_to_file(self, src_file: str, dst_file: str, overwrite: bool = False) -> bool:
        if not overwrite and os.path.exists(dst_file):
            logging.warning(f"File '{dst_file}' already exists. Skipping summarization.")
            return False
        try:
            with open(dst_file, 'w') as fh:
                pass
        except Exception as e:
            logging.error(f"Cannot write to destination '{dst_file}'.")
            logging.error(e)
            return False

        text = self.summarize(src_file)
        with open(dst_file, 'w') as f:
            f.write(text)
        return True

    def summarize(self, text: str) -> str:
        if len(text) < self._text_len_limit:
            return self._summarise_small_text(text)
        else:
            return self._summarise_large_text(text)

    def _summarise_large_text(self, text: str) -> str:
        """ Note: This only works if the text can be split up into smaller chunks by punctuation. """

        def split_text_into_chunks(text: str, chunk_length_limit: len) -> list:
            chunks = []
            tmp, tmp_len = [], 0
            for chunk in self._nlputil.split_text_by_punctuation(text):
                if chunk:
                    if len(chunk) + tmp_len > chunk_length_limit:
                        chunks.append("".join(tmp))
                        tmp, tmp_len = [], 0
                    tmp.append(chunk)
                    tmp_len += len(chunk)
            if tmp:
                chunks.append("".join(tmp))
            return chunks

        summaries = []
        chunks = split_text_into_chunks(text, self._text_len_limit)
        for nr, chunk in enumerate(chunks, start=1):
            if chunk:
                summary = self._summarise_small_text(chunk)
                if not summary:
                    logging.warning(f"Could not summarize text chunk #{nr}. Aborting.")
                    break
                summaries.append(summary)
        return "\n\n".join(summaries)

    def _summarise_small_text(self, text: str) -> str:
        prompt = f"{text}\n\nTl;dr"  # adding Tl;dr to prompt GPT-3 to create a summary
        response = openai.Completion.create(
            model=self._model_name,
            #messages=[{'role': 'user', 'content': prompt}],
            prompt=prompt,
            max_tokens=self._max_return_tokens,
            n=1,
            temperature=self._temperature,
        )
        summary = response["choices"][0]["text"]
        if summary.startswith(":"):
            summary = summary[1:]
        summary = summary.strip()
        return summary
