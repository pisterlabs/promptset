from abc import ABC, abstractmethod
import logging
import openai
from transformers import pipeline
from thefuzz import fuzz, process

logger = logging.getLogger("main").getChild(__name__)

API_KEY = "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"


class TextClassifier(ABC):
    """
    Abstract class for text classifier
    """

    @abstractmethod
    def predict(self, text, labels):
        """
        Given a text and a list of labels, predict the label for the text
        """
        pass

    @abstractmethod
    def predict_batch(self, texts, labels):
        """
        Given a list of texts and a list of labels, predict the labels for the texts
        """
        pass


class GPTClassifier(TextClassifier):
    """
    GPT-3 Text Classifier
    given a list of text, predict the label for each text
    """

    def __init__(self, model="gpt-3.5-turbo-instruct") -> None:
        openai.api_key = API_KEY
        self.model = model
        self.prompt = """
            You will be provided with a description of a transaction,
            and your task is to classify its cateogry as  one of the below.
            In your answer, ONLY write the category name, do not write anything else!

            CATEGORIES

            ---

            Description: DESCRIPTION

            ---
            Cateogry is:
        """.strip()
        self.max_tokens = 5

    def predict(self, text, labels):
        logger.info("GPT Predicting category for %s", text)
        prompt = self.prompt.replace("CATEGORIES", ", ".join(labels))
        prompt = prompt.replace("DESCRIPTION", text)
        response = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=self.model,
        )
        logger.debug("GPT Response: %s", response)
        predicted_label = response.choices[0].text.strip()

        logger.info("GPT Predicted label: %s", predicted_label)
        return predicted_label

    def predict_batch(self, texts, labels):
        logger.info("GPT Predicting category for %s", texts)
        prompts = []
        for text in texts:
            prompt = self.prompt.replace("CATEGORIES", ", ".join(labels))
            prompt = prompt.replace("DESCRIPTION", text)
            prompts.append(prompt)

        response = openai.Completion.create(
            prompt=prompts,
            max_tokens=self.max_tokens,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=self.model,
        )
        logger.debug("GPT Response: %s", response)
        # The response is sometimes Category: cateogry, sometimes Category is: cateogry, so handle the cases
        predicted_labels = []
        for choice in response.choices:
            predicted_label = choice.text.strip()
            predicted_label = predicted_label.replace("Category:", "")
            predicted_label = predicted_label.replace("Category is:", "")
            predicted_label = predicted_label.strip()
            predicted_labels.append(predicted_label)
        logger.info("GPT Predicted labels: %s", predicted_labels)
        return predicted_labels


class SimpleClassifier(TextClassifier):
    """
    This classifier will be used to classify the text using NLP techniques.
    It uses a pre-trained model from huggingface.
    """

    def __init__(self, model="facebook/bart-large-mnli") -> None:
        self.model = model
        self.pipe = None  # only initialize the pipeline when needed

    def predict(self, text, labels):
        logger.info("Simple Predicting category for %s", text)
        if not self.pipe:
            self.pipe = pipeline("zero-shot-classification", model=self.model)
        logger.debug("Simple Labels: %s", labels)
        result = self.pipe(text, labels)
        predicted_label = result["labels"][0]
        logger.info("Simple Predicted label: %s", predicted_label)
        return predicted_label

    def predict_batch(self, texts, labels):
        logger.info("Simple Predicting categories for %s", texts)
        if not self.pipe:
            self.pipe = pipeline("zero-shot-classification", model=self.model)
        result = self.pipe(texts, labels)
        logger.debug("Simple Result: %s", result)
        predicted_labels = [r["labels"][0] for r in result]
        logger.info("Simple Predicted labels: %s", predicted_labels)
        return predicted_labels


def fuzzy_search(text, labels, threshold=85, scorer=fuzz.token_set_ratio):
    """
    Given a text and a list of labels, find the label that best matches the text
    """
    if not labels:
        logger.warning("No labels provided for %s, returning None", text)
        return None

    logger.info("Fuzzy Predicting label for %s", text)
    logger.debug("Fuzzy Labels: %s", labels)
    best_label, best_score = process.extractOne(text, labels, scorer=scorer)
    logger.debug("Fuzzy Best label: %s. score %s", best_label, best_score)
    if best_score < threshold:
        logger.warn("Fuzzy score is below threshold: %s", best_score)
        return None
    logger.info("Fuzzy Predicted label: %s", best_label)
    return best_label
