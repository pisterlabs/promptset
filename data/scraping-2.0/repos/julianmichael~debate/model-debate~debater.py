# Adapted from: https://github.com/akbir/debate

import re
import time
from textwrap import dedent
from typing import List
from tenacity import RetryError
from fastapi import HTTPException

import aiohttp
import prompts
from chat_client import ChatClient


class Debater():

    def __init__(self, story: str, answers: List[str], temperature: float, debater_idx: int, turn_type: str,
                 client: ChatClient):
        super().__init__()
        self.story = story
        self.answers = answers
        self.debater_idx = debater_idx
        self.turn_type = turn_type
        self.name = prompts.NAMES[debater_idx] if turn_type != "single debater" else "Consultant"
        self.results = []
        self.client = client
        self.temperature = temperature

    def prepare_transcript(self, history, turn_type: str, debater_idx: int):
        # TODO: check length of transcript and cut it down to length if possible
        # tip: use something computed from self.client.max_context_size
        separator = "\n---------------------------------------------\n"
        if turn_type == "single debater":
            opening_prompt = f"""
Here is the current transcript of the Consultation:

{self.story}
One of these two answers is correct:
Answer A: {self.answers[0]}
Answer B: {self.answers[1]}

Consultant: I claim the answer is {self.answers[debater_idx]}{separator}\n"""
        else:
            opening_prompt = f"""
Here is the current transcript of the debate:

{self.story}
{prompts.NAME_1}: I claim the answer is {self.answers[0]}
{prompts.NAME_2}: I claim the answer is {self.answers[1]}{separator}\n"""
        history_str = separator.join([f"{name}: {argument}" for name, argument in history])
        transcript = opening_prompt + history_str + separator + f"{self.name}: "
        return transcript

    def check_output_length(self, output: str, char_limit: int, quote_char_limit: int):
        num_output_chars = len(output)
        pattern = r"<quote>(.*?)</quote>"
        matches = re.findall(pattern, output, re.DOTALL)
        num_quote_chars = sum([len(match) for match in matches])
        if num_output_chars > char_limit:
            return "total", num_output_chars, num_quote_chars
        # don't bother with quote limit. more quotes good
        # elif num_quote_chars > quote_char_limit:
        #     return "quote", num_output_chars, num_quote_chars
        return "ok", num_output_chars, num_quote_chars

    async def run_single_turn(self, history, char_limit: int, quote_char_limit: int, turn_type: str):
        word_limit = char_limit / 5
        quote_limit = quote_char_limit / 5
        if turn_type == "single debater":
            rules = prompts.single_debater_rules(word_limit, quote_limit)
            few_shot_examples = prompts.single_few_shot_examples()
        elif turn_type in {"sequential", "simultaneous"}:
            rules = prompts.debate_rules(word_limit, quote_limit, turn_type == "simultaneous")
            # TODO: add examples for debates
            few_shot_examples = prompts.debate_few_shot_examples()
        else:
            raise ValueError(f"Invalid turn type: {turn_type}")

        if turn_type == "single debater":
            self.private = prompts.private_consultation_prompt(self.name, word_limit, quote_limit)
        else:
            self.private = prompts.private_debate_prompt(self.name, word_limit, quote_limit)
        self.position = self.private + f"You argue that the answer is: '{self.answers[self.debater_idx]}'"
        system_prompt = "\n".join([rules, few_shot_examples, self.position])
        transcript = self.prepare_transcript(history, turn_type, self.debater_idx)
        async with aiohttp.ClientSession() as session:
            with open("last_system_prompt.txt", "w") as f:
                f.write(system_prompt)
            with open("last_transcript_prompt.txt", "w") as f:
                f.write(transcript)

            with open("last_prompt.txt", "w") as f:
                f.write(system_prompt + "\n" + "\n\n-------- END SYSTEM PROMPT ------------------\n\n" + transcript)

            output_length_check = ""
            num_output_chars, num_quote_chars = 0, 0
            num_length_retries = 0
            ending_prompt = f"Complete the next turn of debate as your role of {self.name}:"
            while output_length_check != "ok" and num_length_retries < 3:
                if output_length_check == "total":
                    ending_prompt = f"""You just tried to respond by saying:\n\n{response}\n\nbut this was too long.
Your response contained {num_output_chars} characters, but the character limit is {char_limit}.
Please shorten your response, completing the next turn of debate as your role of {self.name}:"""
                elif output_length_check == "quote":
                    ending_prompt = f"""You just tried to respond by saying:\n\n{response}\n\nbut you exceeded the quote limit.
Your response contained {num_quote_chars} quote characters, but the quote limit is {quote_char_limit}.
Please reduce your quote usage to be under the limit, completing the next turn of debate as your role of {self.name}:"""
                with open("last_ending_prompt.txt", "w") as f:
                    f.write(ending_prompt)
                    print(ending_prompt)

                try:
                    response = await self.client.chat_completion_with_backoff_async(
                        session=session,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": transcript,
                            },
                            {
                                "role": "user",
                                "content": ending_prompt,
                            },
                        ],
                        temperature=self.temperature,
                    )
                except RetryError:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded from OpenAI API")
                output_length_check, num_output_chars, num_quote_chars = self.check_output_length(
                    response, char_limit, quote_char_limit)
                num_length_retries += 1
                time.sleep(0.3)
        return response
