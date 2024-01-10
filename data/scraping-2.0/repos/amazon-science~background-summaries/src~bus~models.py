import logging
import re
import time

import openai
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class QAGenerator:
    def __init__(self) -> None:
        pass

    def _parse_outputs(self, txt: str):
        output = re.search(
            (
                r"(?s)"
                r"1[\.|\)](?P<Q1>.*)"
                r"2[\.|\)](?P<Q2>.*)"
                r"3[\.|\)](?P<Q3>.*)"
                r"4[\.|\)](?P<Q4>.*)"
                r"5[\.|\)](?P<Q5>.*)"
            ),
            txt,
        )
        if output is None:
            # split by `?`
            output = [_item.strip(" \n") for _item in re.findall(r"([^\?]+\?)", txt)]
            output = output[:5]
            if len(output) < 5:
                # found fewer than 5 questions
                # add empty strings
                output += [""] * (5 - len(output))
            return output

        return [output.group(f"Q{idx}").strip(" \n") for idx in range(1, 6)]

    def _generate(self):
        return NotImplementedError

    def _get_chunks(self, data: list, batch_size: int):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]


class GPT(QAGenerator):
    def __init__(
        self,
        model_name_or_path: str = "gpt-3.5-turbo",
        temperature: float = 0,
        max_src_len: int = 3696,
        max_tgt_len: int = 400,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.temperature = temperature
        self.model = model_name_or_path
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.question_template = (
            "Update: {update}\n"
            "Imagine you read the above update about a news story. "
            "You have no prior information about the story. "
            "Generate five background questions you might have about the story."
        )
        self.answer_template = (
            "Background: {background}\n\n"
            "Questions: {questions}\n\n"
            "For each question, list answers from the background text when available. "
            'Say "unanswerable" if the question is not answered in the background text.'
        )

    def _generate(self, docs: list[str]):
        out = []
        for doc in tqdm(docs):
            messages = [
                {
                    "role": "user",
                    "content": doc,
                },
            ]
            completion = None
            while completion is None:
                try:
                    completion = openai.ChatCompletion.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=messages,
                    )
                except Exception as E:
                    SLEEP = 240
                    logger.warning(f"found exception: {E}")
                    logger.warning(f"sleeping for {SLEEP} seconds")
                    time.sleep(SLEEP)
            out += [completion["choices"][0]["message"]["content"].strip("\n")]
        return out

    def _concat(self, docs: list[str]) -> str:
        return "\n".join([f"{idx+1}. {doc}" for idx, doc in enumerate(docs)])

    def _truncate_docs(self, docs: list[str]) -> list[str]:
        truncated_docs = []
        for doc in docs:
            truncated_doc = doc
            tokens = self.tokenizer(truncated_doc)["input_ids"]
            while len(tokens) > self.max_src_len:
                splits = re.split(
                    r"(Date: [0-9]{4}-[0-9]{2}-[0-9]{2})", truncated_doc, 2
                )
                if len(splits) == 1:
                    # input doesn't follow Date: ... Article: ... format
                    truncated_doc_ids = self.tokenizer(
                        truncated_doc, truncation=True, max_length=self.max_src_len
                    )["input_ids"]
                    truncated_doc_toks = self.tokenizer.batch_decode(
                        truncated_doc_ids, skip_special_tokens=True
                    )
                    truncated_doc = " ".join(truncated_doc_toks)
                else:
                    truncated_doc = "".join(splits[3:])
                tokens = self.tokenizer(truncated_doc)["input_ids"]
            truncated_docs += [truncated_doc]
        return truncated_docs

    def gen_questions(self, contexts: list[str]) -> list[list[str]]:
        prompts = [self.question_template.format(update=doc) for doc in contexts]
        questions: list[str] = self._generate(prompts)
        questions: list[list[str]] = [self._parse_outputs(_item) for _item in questions]
        return questions

    def gen_answers(
        self, contexts: list[str], questions: list[list[str]]
    ) -> list[list[str]]:
        """
        gpt-3.5 allows extracting all answers at once
        """
        assert len(contexts) == len(questions)
        truncated_contexts = self._truncate_docs(contexts)
        prompts = [
            self.answer_template.format(
                background=context, questions=self._concat(context_questions)
            )
            for context, context_questions in zip(truncated_contexts, questions)
        ]
        answers: list[str] = self._generate(prompts)
        answers: list[list[str]] = [self._parse_outputs(_item) for _item in answers]
        return answers
