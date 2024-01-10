import copy
import time
from typing import List

import openai
from termcolor import colored

from commands.embed import get_embedding
from model.brains.modeling_memory import MemoryRetrievalBrain
import logging


logging.basicConfig(level=logging.INFO)


def request(
    model,
    messages,
    temperature=1.0
):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            break
        except openai.error.RateLimitError as e:
            logging.warning(str(e))
            logging.warning("Retrying...")
            time.sleep(1)

    return response


def format_memories(memories):
    formated_memories = ""
    for (i, m) in enumerate(memories):
        formated_memories += f"Memory {i+1}: {m}\n"
    return formated_memories


class GPTChatModel:
    def __init__(self, memory_brain: str, system="You are a helpful assistant.", temperature=0.3, no_brain=False, incontext_example: List[List[str]]=None):
        self.model = "gpt-3.5-turbo"
        self.no_brain = no_brain
        if not no_brain:
            self.memory_brain = MemoryRetrievalBrain(memory_brain)
            introduction_id = "00000000-0000-0000-0000-000000000000"
            introduction = self.memory_brain.remember_with_ids([introduction_id])[0]
            if introduction is None:
                introduction = "AutoLearn-GPT is an experimental project and model to explore " \
                                "whether GPT models can learn by themselves. The model can learn " \
                                "new things and memorize them while interacting with humans."
                self.memory_brain.memorize(introduction, id=introduction_id)

            logging.info(colored(introduction, "blue"))

        self.short_term_memories = []
        self.system = system
        self.temperature = temperature
        self.conv = [
            {"role": "system", "content": system}
        ]
        self.generate_template = copy.deepcopy(self.conv)
        if incontext_example is not None:
            for ex in incontext_example:
                self.generate_template.append({'role': "user", "content": ex[0]})
                self.generate_template.append({'role': "assistant", "content": ex[1]})

    def clear_history(self):
        self.conv = [
            {"role": "system", "content": self.system}
        ]

    # one turn chat with retrieval to memory brain, do not save conversation
    def generate_with_memory(self, user_input: str, temperature=None):
        if self.no_brain:
            raise RuntimeError("This model does not contain a memory brain.")
        logging.info(colored("Model: I am trying to remember related content...", "yellow"))
        memories = self.memory_brain.remember(user_input)
        # answering_with_memory_prompt = "The following are some memories you just recalled for the question \"{question}\". " \
        #                                "These memories consist of queries paired with explanation.\n{formated_memories}\n\n" \
        #                                "You should respond with a python dictionary, which contains the " \
        #                                "following keys.\n\"has_answer\": a boolean variable indicates whether the memories contain the answer.\n\"answer\": " \
        #                                "a answer string for the question \"{question}\". You should generate it from the memories if they contain the answer, otherwise generate it by yourself. Becareful that the memories may not fully contain the answer so you should use them as supplement.\n" \
        #                                "\"explanation\": a string to explain why previous keys should be set that way."
        answering_with_memory_prompt = "Try to answer the question \"{question}\" " \
                                       "The following are some memories you just recalled. You may use the information from them.\n\n{formated_memories}"
        formated_memories = format_memories(memories)
        # logging.info(f"Memories are:\n{formated_memories}")
        # print(answering_with_memory_prompt.format(question=user_input, formated_memories=formated_memories))
        response = self.generate(
            answering_with_memory_prompt.format(question=user_input, formated_memories=formated_memories))

        return response

    # chat and save conversation to history
    def chat(self, user_input, temperature=None):
        self.conv.append({"role": "user", "content": user_input})
        response = request(
            model=self.model,
            messages=self.conv,
            temperature=self.temperature if temperature is None else temperature
        )
        self.conv.append({
            "role": "assistant",
            "content": response["choices"][0]["message"]["content"]
        })

        return response["choices"][0]["message"]["content"]

    # one turn chat without saving conversation to history
    def generate(self, user_input, temperature=None):
        generate_template = copy.deepcopy(self.generate_template)
        generate_template.append({"role": "user", "content": user_input})
        response = request(
            model=self.model,
            messages=generate_template,
            temperature=self.temperature if temperature is None else temperature
        )

        return response["choices"][0]["message"]["content"]


