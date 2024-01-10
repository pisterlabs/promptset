import re
import openai

from conf import OPEN_AI_API_KEY, MODEL_ID, TOKEN_LIMIT
from logger import get_logger

from .base import Filter

logger = get_logger("gpt_filter")


class MakeAIDoTheFiltering(Filter):
    """Leverages GPT-3 to to check if a line is a category or not. This honestly doesn't work very well, but it's a cool idea.

    Args:
        conf_threshold (float, optional): The maximum threshold for a line to be considered a processed using GPT-3, reducing this value will use less tokens.
        Defaults to 1.
    """

    def __init__(self, weight: float, conf_threshold: float = 1):
        self.set_openai_api_key()
        self.weight = weight
        self.conf_threshold = conf_threshold

    @staticmethod
    def set_openai_api_key():
        if OPEN_AI_API_KEY:
            openai.api_key = OPEN_AI_API_KEY

    def apply(self, lines):
        if not OPEN_AI_API_KEY:
            logger.warning(
                "OPEN_AI_API_KEY not set, the AI based filter will be skipped"
            )
            return

        base_prompt = self.create_base_prompt()
        probabilities = self.extract_probabilities(base_prompt, lines)

        for p in probabilities:
            self.update_line_confidence(p, lines)

    @staticmethod
    def create_base_prompt():
        return """Classify restaurant menu strings as probable (100) or improbable (0) menu categories. 'Polevky', 'Wafle', 'Vino' are examples of categories, while specific items or prices aren't. Output should follow \\d+: \\d{1,3} format, with the first number being the ID and the second the category likelihood percentage.

        Example input:
        0: Soups
        1: Wafle
        2: Wafle s grilovanou slaninou
        3: 5.99,-
        4: Hot drinks
        Example output based on the above input:
        0: 99
        1: 75
        2: 15
        3: 01
        4: 95

        provide a similar output for the following all of the following lines:
        """

    def extract_probabilities(self, base_prompt, lines):
        re_get_probabilities = re.compile(r"\d+: \d{1,3}")
        id = 0
        probabilities = []
        while id < len(lines):
            prompt, id = self.create_prompt(id, base_prompt, lines)
            content = self.get_content_from_openai(prompt)
            probabilities += re_get_probabilities.findall(content)
        return probabilities

    def create_prompt(self, id, base_prompt, lines):
        prompt = base_prompt + "\n\n"

        for i in range(id, len(lines)):
            id += 1
            line = lines[i]
            # no reason to use up tokens on lines that are already filtered out
            if line.analysis.category_confidence < self.conf_threshold:
                continue

            prompt += f"\n{i}: {line.text}"

            #  1 token is approximately 4 characters
            if len(prompt) / 4 > TOKEN_LIMIT:
                break

        logger.info(
            f"Sending approximately {len(prompt) / 4} tokens to GPT-3, this may take a while"
        )
        return prompt, id

    @staticmethod
    def get_content_from_openai(prompt):
        response = openai.ChatCompletion.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["choices"][0]["message"]["content"]  # type: ignore

    def update_line_confidence(self, p, lines):
        index, probability = p.split(": ")
        index = int(index)
        probability = float(probability)
        # calculate the confidence modifier based on the probability, and scale it's effect based on the weight
        confidence_multiplier = 1 - (1 - probability / 100) * self.weight

        old_confidence = lines[index].analysis.category_confidence
        lines[index].analysis.category_confidence *= confidence_multiplier
        new_confidence = lines[index].analysis.category_confidence

        # Uncomment this line to enable detailed logging.
        # logger.debug(f"'{lines[index].text}'\n\t{probability=}, {self.weight=}, {confidence_multiplier=}\n\t{old_confidence}->{new_confidence}")
