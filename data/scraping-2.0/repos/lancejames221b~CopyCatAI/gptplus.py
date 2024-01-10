import openai
import json
import os, json
from openai import APIError
import configparser
import re
from notification import *
import tiktoken
from requests.exceptions import RequestException, Timeout
import uuid


def guid_generator():
    return str(uuid.uuid4())


TIMEOUT_SECONDS = 60

home_dir = os.path.expanduser("~")
bundle_dir = os.path.join(home_dir, "Library", "Application Support", "CopyCat")
models_path = os.path.join(bundle_dir, "models.json")


def manage_memory(messages, model_name, max_tokens=None):
    with open(models_path, "r") as f:
        models = json.load(f)

        if model_name not in models:
            raise ValueError(f"Invalid model name: {model_name}")

        if not max_tokens:
            max_tokens = models[model_name]["token_size"] * 0.95  # 95% of the maximum

    # Ensure no single message exceeds max_tokens
    for i, message in enumerate(messages):
        if len(message["content"]) > max_tokens:
            # Truncate the message
            messages[i]["content"] = message["content"][:max_tokens]

    total_tokens = sum(len(message["content"]) for message in messages)

    while total_tokens > max_tokens:
        # Remove the first message that is an answer
        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                total_tokens -= len(message["content"])
                messages.pop(i)
                break

    return messages


def parse_token_error(error_msg):
    max_tokens_pattern = r"maximum context length is (\d+) tokens"
    actual_tokens_pattern = r"your messages resulted in (\d+) tokens"

    max_tokens_match = re.search(max_tokens_pattern, error_msg)
    actual_tokens_match = re.search(actual_tokens_pattern, error_msg)

    if max_tokens_match and actual_tokens_match:
        max_tokens = int(max_tokens_match.group(1))
        actual_tokens = int(actual_tokens_match.group(1))
        return max_tokens, actual_tokens
    else:
        # Handle the case when the regex search doesn't find a match
        return None, None


def calculate_cost(prompt_tokens, completion_tokens, total_tokens, model=None):
    with open(models_path, "r") as f:
        models = json.load(f)

    if model in models:
        system_prompt_price_per_token = (
            models[model]["input_price_per_1k_tokens"] / 1000
        )
        response_price_per_token = models[model]["output_price_per_1k_tokens"] / 1000
        total_price = (
            prompt_tokens * system_prompt_price_per_token
            + completion_tokens * response_price_per_token
        )
        return total_price
    else:
        return 0


class OpenAIMemory:
    def __init__(
        self,
        memory_file=bundle_dir + "/memory.json",
        config_file=bundle_dir + "/config.ini",
    ):
        self.memories = {}
        self.memory_file = memory_file
        self.config_file = config_file
        self.load_memory()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add_to_memory(self, system_prompt, message, ai_response=False):
        if not message.strip():
            return

        if system_prompt not in self.memories:
            self.memories[system_prompt] = {
                "system_prompt": system_prompt,
                "messages": [],
            }
            self.add_system_prompt(system_prompt)

        if ai_response:
            role = "assistant"
        else:
            role = "user"

        if (
            role == "user"
            and self.memories[system_prompt]["messages"]
            and self.memories[system_prompt]["messages"][-1]["content"] == message
        ):
            # If the last message in memory has the same content, do not add it again
            return

        self.memories[system_prompt]["messages"].append(
            {"role": role, "content": message}
        )
        self.save_memory()

    def add_system_prompt(self, system_prompt):
        if system_prompt in self.memories:
            self.memories[system_prompt]["messages"].insert(
                0, {"role": "system", "content": system_prompt}
            )

    def clear_memory(self, system_prompt):
        if system_prompt in self.memories:
            del self.memories[system_prompt]
            self.save_memory()

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memories, f, indent=4)

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                self.memories = json.load(f)

    def get_memory_keys(self):
        return list(self.memories.keys())

    def generate_response(
        self,
        system_prompt,
        prompt,
        model="gpt-3.5-turbo",
        tokens=None,
        temperature=0.8,
        use_memory=True,
    ):
        memory = (
            self.memories[system_prompt]["messages"]
            if use_memory and system_prompt in self.memories
            else []
        )

        messages = []
        if memory:
            for i, m in enumerate(memory):
                if i == 0:
                    messages.append({"role": "system", "content": m["content"]})
                else:
                    messages.append(
                        {"role": m["role"].lower(), "content": m["content"]}
                    )
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=tokens,
                n=1,
                stop=None,
                temperature=temperature,
                stream=False,
            )
        except openai.error.InvalidRequestError as error:
            raise Exception(error)

        ai_response = response["choices"][0]["message"]["content"].strip()
        # Update the token counts here
        self.prompt_tokens += response["usage"][
            "prompt_tokens"
        ]  # Update with actual prompt token count
        self.completion_tokens += response["usage"][
            "completion_tokens"
        ]  # Update with actual completion token count
        self.total_tokens += (
            self.prompt_tokens + self.completion_tokens
        )  # Update total tokens

        if use_memory:
            self.add_to_memory(system_prompt, prompt)
            self.add_to_memory(system_prompt, ai_response, ai_response=True)

        return (
            ai_response,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        )


class CostManager:
    def __init__(self, config_file, memory_path):
        self.config_file = config_file  # Store the config_file as an instance variable
        self.memory_path = memory_path  # Store the memory_path as an instance variable
        self.config = configparser.ConfigParser(strict=False, interpolation=None)
        self.config.read(config_file)
        self.total_costs = self.load_cost()

    def load_cost(self):
        cost = float(self.config.get("OpenAI", "total_costs"))
        return cost

    def save_cost(self, costs, total_costs, total_tokens):
        try:
            costs = float(costs)
        except ValueError:
            print("Invalid value for costs:", costs)
            return

        try:
            total_costs = float(total_costs)
        except ValueError:
            print("Invalid value for total_costs:", total_costs)
            return

        self.config.set("OpenAI", "costs", str(costs))
        self.config.set("OpenAI", "total_costs", str(total_costs))
        self.config.set("OpenAI", "total_tokens", str(total_tokens))

        with open(self.config_file, "w") as config_file:
            self.config.write(config_file)

    def update_total_cost(self, cost):
        self.total_costs += cost

    def process_request(
        self,
        system_prompt,
        question,
        model,
        use_memory=True,
        tokens=None,
        temperature=0.8,
    ):
        openai_memory = OpenAIMemory(self.memory_path, self.config_file)
        if use_memory and system_prompt not in openai_memory.memories:
            openai_memory.add_to_memory(system_prompt, system_prompt)

        messages = (
            openai_memory.memories[system_prompt]["messages"] if use_memory else []
        )
        if not messages and system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages = manage_memory(
            messages,
            model,
            max_tokens=tokens,
        )  # Call manage_memory here

        try:
            (
                response,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            ) = openai_memory.generate_response(
                system_prompt,
                question,
                model=model,
                use_memory=use_memory,
                tokens=tokens,
                temperature=temperature,
            )
        except APIError as error:
            print(f"OpenAI API Error: {str(error)}")
            response = ""
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        cost = calculate_cost(
            prompt_tokens,
            completion_tokens,
            total_tokens,
            model,
        )

        self.total_costs += cost
        self.save_cost(cost, self.total_costs, total_tokens)
        print("Cost:", cost)
        print("Prompt tokens:", prompt_tokens)
        print("Completion tokens:", completion_tokens)
        print("Total tokens:", total_tokens)
        print("Total Costs:", self.total_costs)

        if use_memory:
            memory_content = "\n\n".join(
                [
                    m["content"]
                    for m in openai_memory.memories[system_prompt]["messages"]
                ]
            )
        else:
            memory_content = ""

        return {
            "system_prompt": system_prompt,
            "question": question,
            "memory": memory_content,
            "model": model,
            "response": response,
            "total_tokens": total_tokens,
            "cost": cost,
            "total_costs": self.total_costs,
        }
