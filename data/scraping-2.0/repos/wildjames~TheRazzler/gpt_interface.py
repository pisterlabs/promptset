import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import openai
from openai.error import RateLimitError
import tiktoken
from modelsinfo import COSTS

logger = logging.getLogger(__name__)


def clean_filename(filename):
    # Replaces non-alphanumeric characters (except for periods, hyphens and underscores) with an underscore
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    # Replaces any remaining forward slashes with an underscore
    filename = filename.replace("/", "_")
    return filename


def create_chat_message(role, content):
    """
    Create a chat message with the given role and content.

    Args:
    role (str): The role of the message sender, e.g., "system", "user", or "assistant".
    content (str): The content of the message.

    Returns:
    dict: A dictionary containing the role and content of the message.
    """
    return {"role": role, "content": content}


@dataclass
class SignalAI:
    """The SignalAI class, which contains all the logic for the responder AI.

    This class will keep track of the AI's cost, and the user's budget, and will stop when the budget is exceeded.
    """

    model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 400
    prompt_filename: str = "prompt.txt"
    prompt_profile_filename: str = "prompt_profile.txt"
    niceness: float = 0.5
    profile_fname_template: str = "profile_{group}_{name}.txt"
    profile_model: str = "gpt-3.5-turbo"
    last_profiled: int = 0
    total_cost_filename: str = "total_cost.txt"

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0
    total_budget: float = 0
    debug: bool = False
    razzler_rate: float = 0.1
    razzler_image_rate: float = 0.1

    enc: Dict[str, tiktoken.Encoding] = None  # type: ignore

    def __post_init__(self):
        self.enc = {
            "voice": tiktoken.encoding_for_model(self.model),
            "profile": tiktoken.encoding_for_model(self.profile_model),
        }
        self.get_total_cost()

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0

    def get_profile_fname(self, group, name):
        profile_fname = self.profile_fname_template.format(group=group, name=name)
        profile_fname = profile_fname.replace(" ", "")
        profile_fname = clean_filename(profile_fname)
        profile_fname = os.path.join("profiles", profile_fname)

        return profile_fname

    def get_profile(self, group, name):
        """
        Get the profile for the given group and name.

        Args:
        group (str): The group of the profile.
        name (str): The name of the profile.

        Returns:
        str: The profile text.
        """
        profile_fname = self.get_profile_fname(group=group, name=name)

        if os.path.exists(profile_fname):
            with open(profile_fname, "r") as f:
                profile = f.read()
        else:
            profile = "Unkown"

        profile = "{}: \n".format(name) + profile
        return profile

    def create_chat_completion(
        self,
        messages: list,  # type: ignore
        model: str = None,
        spender: str = None,
    ) -> str:
        """
        Create a chat completion and update the cost.
        Args:
        messages (list): The list of messages to send to the API.
        model (str): The model to use for the API call.
        temperature (float): The temperature to use for the API call.
        max_tokens (int): The maximum number of tokens for the API call.
        Returns:
        str: The AI's response.
        """
        if model is None:
            model = self.model
        logger.debug(f"[GPTInterface] Using model: {model}")
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except RateLimitError:
            # Fall back on GPT 3.5
            response = self.create_chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                spender=spender,
            )
            
        if self.debug:
            logger.debug(f"[GPTInterface] Response: {response}")
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, self.model, spender)
        return response

    def create_image_completion(self, text: str, spender=None) -> str:
        """
        Create an image completion and update the cost.

        Args:
        text (str): The text to send to the API.

        Returns:
        str: The AI's response.
        """
        try:
            response = openai.Image.create(
                prompt=text,
                n=1,
                size="1024x1024",
            )
            logger.info(f"[GPTInterface] Response: {response}")
            image_url = response["data"][0]["url"]

            self.update_cost(1, 0, "image", spender)
        except:
            image_url = "https://cdn.openart.ai/stable_diffusion/42f53e9b69daeaef0d2e7b29f9cb938e2e385496_2000x2000.webp"

        return image_url

    def embedding_create(
        self,
        text_list: List[str],
        model: str = "text-embedding-ada-002",
        spender: str = None,
    ) -> List[float]:
        """
        Create an embedding for the given input text using the specified model.

        Args:
        text_list (List[str]): Input text for which the embedding is to be created.
        model (str, optional): The model to use for generating the embedding.

        Returns:
        List[float]: The generated embedding as a list of float values.
        """
        response = openai.Embedding.create(input=text_list, model=model)

        self.update_cost(response.usage.prompt_tokens, 0, model, spender)
        return response["data"][0]["embedding"]

    def get_this_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        this_cost = (
            prompt_tokens * COSTS[model]["prompt"]
            + completion_tokens * COSTS[model]["completion"]
        ) / 1000
        return this_cost

    def update_cost(self, prompt_tokens, completion_tokens, model, spender=None):
        this_cost = self.get_this_cost(prompt_tokens, completion_tokens, model)

        # Load the cost JSON file
        spending: Dict = json.load(open(self.total_cost_filename, "r"))
        spending["total_cost"] += this_cost

        if spender not in spending.keys():
            spending[spender] = 0.0

        spending[spender] += this_cost
        json.dump(spending, open(self.total_cost_filename, "w"))
        logger.info(
            f"[GPTInterface] OpenAI call cost ${this_cost:.3f}. Total running cost: ${self.total_cost:.3f} out of a budget of ${self.total_budget:.3f}"
        )

    def get_spending(self, spender):
        spending: Dict = json.load(open(self.total_cost_filename, "r"))
        return 0.0 if spender not in spending.keys() else spending[spender]

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        """
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """

        try:
            with open(self.total_cost_filename, "r") as f:
                self.total_cost = json.load(f)["total_cost"]
        except FileNotFoundError:
            self.total_cost = 0

        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
        float: The total budget for API calls.
        """
        return self.total_budget
