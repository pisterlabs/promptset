import openai
import numpy as np
from hate_speech_detection.classifier_model import ClassifierModel


class GPT3TextInference(ClassifierModel):
    def __init__(self):
        super(GPT3TextInference, self).__init__()

    # Define the function to detect hate speech
    def hate_word_count(self, sentence):
        # Extract the hate speech words using regular expressions
        # hate_speech_words = re.findall(r"\b(?!not\b)\w+", sentence.lower())
        hate_speech_words = ["nigger", "nigga", "negro", "muslim", "Black", "whore", "fuck", "cuck", "terrorist",
                             "asshole", "cunt", "fucker", "kill", "bomb", "shoot", "commies", "leftist", "trump", "white", "blonde", "dead"]
        num_words = 0
        for word in sentence.split():
            if word.lower() in hate_speech_words:
                num_words += 1

        # num_words = len(hate_speech_words)
        total_words = len(sentence.split())
        percentage = num_words / total_words * 100
        return percentage
    # Define the function to detect hate speech

    def detect_hate_speech(self, sentence):
        # Classify the sentence as either hate speech or not hate speech using GPT-3
        prompt = f"Is it Hate speech ? reply in yes or no :\n{sentence}\n"
        print(f'prompt = {prompt}')
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0,
            max_tokens=5,  # Increase max_tokens to include the entire classification
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        classification = str(response.choices[0].text.strip())
        print(classification)
        if 'yes' in classification.lower():
            return True
        else:
            return False

    def detect_social_bias(self, sentence):
        return False
        # Classify the sentence as either hate speech or not hate speech using GPT-3
        prompt = f"Is following sentence targeting any social group, nation, race, ethnicity, gender, religion ? reply in yes or no :\n{sentence}\n"
        print(f'prompt = {prompt}')
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0,
            max_tokens=5,  # Increase max_tokens to include the entire classification
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        classification = str(response.choices[0].text.strip())
        print(classification)
        if 'yes' in classification.lower():
            return True
        else:
            return False

    def predict(self, inputs):
        if np.isscalar(inputs):
            return self.detect_hate_speech(inputs), self.detect_social_bias(inputs)
        else:
            return [self.detect_hate_speech(inp) for inp in inputs], [self.detect_social_bias(inp) for inp in inputs]
