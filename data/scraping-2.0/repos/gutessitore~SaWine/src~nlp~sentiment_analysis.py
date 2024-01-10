import openai
import os


class ReviewClassifier:
    def __init__(self):
        self.prompt = "Decide whether a wine review's sentiment is positive, neutral, or negative.\n\nreview: \"{}\"\nSentiment:"
        self.model = "text-davinci-003"
        self.temperature = 0
        self.max_tokens = 60
        self.top_p = 1.0
        self.frequency_penalty = 0.5
        self.presence_penalty = 0.0
        self.__authenticate()

    def __authenticate(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def get_sentiment(self, review):
        prompt = self.prompt.format(review)

        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )

        if not response['choices']:
            raise Exception("OpenAI API request failed: {}".format(response))

        sentiment = response['choices'][0]['text'].strip().lower()
        return sentiment
