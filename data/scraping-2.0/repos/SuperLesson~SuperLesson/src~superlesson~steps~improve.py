import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import cast

from superlesson.storage import Slides
from superlesson.storage.store import Store
from superlesson.storage.utils import diff_words

from .step import Step, step

logger = logging.getLogger("superlesson")


@dataclass
class Prompt:
    body: str
    slide: int = 0


class Replace:
    def __init__(self, slides: Slides):
        self._replacements_path = slides.lesson_root / "replacements.txt"
        self.slides = slides

    @step(Step.replace, Step.merge)
    def bogus_words(self):
        if not self._replacements_path.exists():
            logger.warning(
                f"{self._replacements_path} doesn't exist, so no replacements will be done"
            )
            return

        pattern = re.compile(r'\s*([^"]*[^"\s])')

        lines = self._replacements_path.read_text().split("\n")
        for line in lines:
            if line.strip() == "":
                continue
            words = line.split("->")
            if len(words) != 2:
                logger.warning(f"Invalid line: {line}")
                continue
            word = cast(re.Match, pattern.search(words[0])).group(1)
            rep = cast(re.Match, pattern.search(words[1])).group(1)
            logger.debug("Replacing %s with %s", word, rep)
            for slide in self.slides:
                slide.transcription = re.sub(
                    r"\b%s\b" % re.escape(word),
                    rep,
                    slide.transcription,
                    flags=re.IGNORECASE,
                )


class Improve:
    def __init__(self, slides: Slides):
        from dotenv import load_dotenv

        load_dotenv()

        self.slides = slides

    @staticmethod
    def _diff_gpt(before: str, after: str):
        file1 = Store.temp_save(before)
        file2 = Store.temp_save(after)

        diff_words(file1, file2)

    @step(Step.improve, Step.merge)
    def punctuation(self):
        context = """O texto a seguir precisa ser preparado para impressão.
        - formate o texto, sem fazer modificações de conteúdo.
        - corrija qualquer erro de digitação ou de grafia.
        - faça as quebras de paragrafação que forem necessárias.
        - coloque as pontuações adequadas.
        - a saída deve ser somente a resposta, sem frases como
        - "aqui está o texto revisado e formatado".
        - NÃO FAÇA NENHUMA MODIFICAÇÃO DE CONTEÚDO, SOMENTE DE FORMATAÇÃO.
        """

        # margin = 20  # to avoid errors
        # GPT-3.5-turbo-1106 takes in at most 16k tokens, but it only outputs 4k tokens, so we use
        # it as the maximum and ignore margin and context size for now
        max_input_tokens = 2**12  # - self._count_tokens(context)) // 2 - margin
        logger.debug(f"Max input tokens: {max_input_tokens}")

        logger.debug("Splitting into prompts")
        prompts = self._split_into_prompts(
            [slide.transcription for slide in self.slides],
            max_input_tokens,
        )

        bench_start = time.time()

        completions = asyncio.run(self._complete_with_chatgpt(prompts, context))

        bench_duration = time.time() - bench_start
        logger.info(f"ChatGPT requests took {bench_duration} to finish")

        last_slide = 0
        improved_transcription = []
        for prompt, completion in zip(prompts, completions, strict=True):
            text = prompt.body
            similarity_ratio = self._calculate_difference(text, completion)
            # different = 0 < similarity_ratio < 1 = same
            if similarity_ratio < 0.40:
                logger.info("The text was not improved by ChatGPT-3.5-turbo.")
                logger.debug(f"Similarity: {similarity_ratio}")
                logger.debug(f"Diffing GPT output for slide {prompt.slide}")
                if logger.isEnabledFor(logging.DEBUG):
                    self._diff_gpt(text, completion)
                continue
            if prompt.slide != last_slide:
                self.slides[last_slide].transcription = " ".join(improved_transcription)
                improved_transcription = []
                last_slide = prompt.slide
            improved_transcription.append(completion)

    @staticmethod
    def _count_tokens(text: str) -> int:
        import tiktoken

        # Based on: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        tokens_per_message = 4
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text)) + tokens_per_message

    @classmethod
    def _split_into_prompts(
        cls, transcriptions: list[str], max_tokens: int
    ) -> list[Prompt]:
        prompts = []
        for i, transcription in enumerate(transcriptions):
            chunks = []
            for period in cls._split_in_periods(transcription):
                tokens = cls._count_tokens(period)
                if tokens > max_tokens:
                    logger.debug(f"Splitting text with {tokens} tokens")
                    words = period.split()
                    chunks.extend(cls._merge_chunks(words, max_tokens))
                else:
                    chunks.append(period)

            merged = cls._merge_chunks(chunks, max_tokens)
            for text in merged:
                prompts.append(Prompt(text, i))

        return prompts

    @classmethod
    def _split_in_periods(cls, text: str) -> list[str]:
        import re

        splits = re.split(r"([.?!])", text)
        if len(splits) == 1:
            return splits

        if splits[0] == "":
            splits = splits[1:]

        if splits[-1] == "":
            splits = splits[:-1]

        # merge punctuation with previous sentence
        # we iterate over len(splits) - 1 because, if it's even,
        # we get the same result, and if it's odd, we skip the
        # invalid iteration at the end
        periods = []
        for i in range(0, len(splits) - 1, 2):
            periods.append(splits[i] + splits[i + 1])

        if len(splits) % 2 == 1:
            periods.append(splits[-1])

        return periods

    @classmethod
    def _merge_chunks(cls, chunks: list[str], max_tokens: int) -> list[str]:
        merged = []
        total_tokens = 0
        start = 0
        for i, chunk in enumerate(chunks):
            tokens = cls._count_tokens(chunk)
            if total_tokens + tokens > max_tokens:
                merged.append(" ".join(chunks[start:i]))

                start = i
                total_tokens = tokens
            else:
                total_tokens += tokens
        merged.append(" ".join(chunks[start:]))

        return merged

    @classmethod
    async def _complete_with_chatgpt(
        cls, prompts: list[Prompt], context: str
    ) -> list[str]:
        from openai import AsyncOpenAI, OpenAIError

        try:
            client = AsyncOpenAI()
        except OpenAIError as e:
            msg = "Please review README.md for instructions on how to set up your OpenAI token"
            raise Exception(msg) from e

        tasks = [
            asyncio.create_task(cls._complete_prompt(client, prompt.body, context))
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks)
        completions = []
        for result in results:
            if result is None:
                break
            cast(str, result)
            completions.append(result)
        logger.debug("ChatGPT completed prompt successfully")

        return completions

    @classmethod
    async def _complete_prompt(cls, client, prompt: str, context: str) -> str | None:
        import openai

        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]
        logger.debug("Completing prompt: %s", prompt)
        try:
            completion = await client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                # model="gpt-4",
                messages=messages,
                n=1,
                temperature=0.1,
            )
            logger.debug("ChatGPT response: %s", completion.choices[0].message.content)
            return completion.choices[0].message.content
        except openai.RateLimitError:
            return None

    @staticmethod
    def _calculate_difference(paragraph1, paragraph2):
        import difflib

        words1 = paragraph1.split()
        words2 = paragraph2.split()
        return difflib.SequenceMatcher(None, words1, words2).ratio()
