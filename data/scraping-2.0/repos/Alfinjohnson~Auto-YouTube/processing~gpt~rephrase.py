import openai
import logging

from utilities.const import YOUR_OPENAI_API_KEY, GPT_MODEL

# Configure OpenAI API
openai.api_key = YOUR_OPENAI_API_KEY

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('rephraser.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Rephrase:
    logger.info(f"Inside class Rephrase")

    def __init__(self, content):
        self.content = content

    def rephrase_with_gpt(self):
        prompt = f"Rephrase the following sentence:\n\"{self.content}\"\n\nRewritten Sentence:"
        response = openai.Completion.create(
            engine=GPT_MODEL,
            prompt=prompt,
            max_tokens=64,
            temperature=0.6,
            n=1,
            stop=None,
        )
        rephrased_sentence = response.choices[0].text.strip()
        return rephrased_sentence

    def rephrase_sentence(self):
        logger.info(f"Received sentence: {self.content}")
        rephrased_sentence = self.rephrase_with_gpt()
        logger.info(f"Rephrased sentence: {rephrased_sentence}")
        return rephrased_sentence


# rephrase = Rephrase(content)
# rephrase_response = rephrase.rephrase_sentence()
# json_obj = {"title": title, "content": rephrase_response}
