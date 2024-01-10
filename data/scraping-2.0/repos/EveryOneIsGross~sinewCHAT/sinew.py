## I ran out of credits so I am using GPT4ALL server routing in this code.
## summary is a bit broken and summary keyword is returning 1st word still

import numpy as np
import json
import openai
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from rake_nltk import Rake

class OpenAIAgent:
    def __init__(self, model):
        openai.api_base = "http://localhost:4891/v1"
        openai.api_key = "not needed for a local LLM"
        self.model = model

    def generate_response(self, prompt, previous_response=None, temperature=0.5):
        messages = [
            {"role": "system", "content": "Respond directly to the prompt. Do not ask a question. Summarize as best as possible."},
            {"role": "user", "content": prompt}
        ]

        if previous_response:
            messages.append({"role": "assistant", "content": previous_response})

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=50,
            frequency_penalty=0.9,
            n=1, 
            stop=None,
            temperature=temperature
        )

        return response['choices'][0]['message']['content']

    def generate_summary(self, prompt, temperature=0.5, max_tokens=1024):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Your task is to summarize the following text:"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response['choices'][0]['message']['content'].strip()

class Neuron:
    def __init__(self, agent):
        self.agent = agent
        self.sia = SentimentIntensityAnalyzer()
        self.responses = []
        self.sentiment_scores = []
        self.keywords = []  # Initialize an empty list for keywords
        self.should_include = True
        self.previous_responses = []
        self.last_positive_response = ""
        self.rake = Rake()

    def process(self, input_data, feedback_output=None):
        response = self.agent.generate_response(input_data)
        sentiment_score = self.sia.polarity_scores(response)['compound']
        self.responses.append(response)
        self.sentiment_scores.append(sentiment_score)
        self.keywords.append(self.extract_keywords(response, input_data))  # Append keywords to the list

        # Update last positive response if the sentiment score is positive
        if sentiment_score > 0:
            self.last_positive_response = response

        return response, sentiment_score


    def extract_keywords(self, text, input_data):
        self.rake.extract_keywords_from_text(text)
        ranked_keywords = self.rake.get_ranked_phrases_with_scores()
        relevant_keywords = [keyword for score, keyword in ranked_keywords if keyword in input_data]

        return relevant_keywords[:5]  # Extract the top 5 relevant keywords


class NeuralModel:
    def __init__(self, agent):
        self.agent = agent
        self.neurons = [Neuron(agent) for _ in range(5)]
        self.attention_weights = np.random.rand(5)
        self.feedback_weights = np.random.rand(5)
        self.learning_rate = 0.01
        self.sia = SentimentIntensityAnalyzer()

    def process_input(self, input_data):
        query_sentiment = self.sia.polarity_scores(input_data)['compound']
        self.attention_weights /= np.sum(self.attention_weights)

        sentiment_scores = []
        adjusted_responses = []

        for neuron in self.neurons:
            response, sentiment_score = neuron.process(input_data)
            sentiment_scores.append(sentiment_score)

            if sentiment_score < -0.5:
                new_prompt = f"Building on your thoughts: '{neuron.last_positive_response}', could you provide a more positive perspective on {input_data}?"
                response, sentiment_score = neuron.process(new_prompt)
                sentiment_scores[-1] = sentiment_score

            adjusted_responses.append(response)

        feedback_output = np.dot(self.feedback_weights, sentiment_scores)
        attended_output = np.dot(self.attention_weights, sentiment_scores)

        for neuron in self.neurons:
            neuron.sentiment_scores.append(feedback_output)

        # Set a constant temperature for the summary
        adjusted_summary = self.agent.generate_summary(' '.join(adjusted_responses), temperature=0.5)
        tokens = word_tokenize(adjusted_summary)
        self.summary = adjusted_summary
        self.summary_keyword = tokens[0] if tokens else ""

        data = {
            'prompt': input_data,
            'sentiment_scores': [neuron.sentiment_scores[-1] for neuron in self.neurons],
            'adjusted_responses': adjusted_responses,
            'adjusted_summary': adjusted_summary,
            'adjusted_summary_keyword': self.summary_keyword,
            'attended_output': attended_output,
            'feedback_output': feedback_output,
            'attention_weights': self.attention_weights.tolist(),
            'feedback_weights': self.feedback_weights.tolist(),
            'query_sentiment': query_sentiment,
            'sentiment_scores': sentiment_scores,
            'responses': [neuron.responses[-1] for neuron in self.neurons],
            'keywords': [neuron.keywords[-1] for neuron in self.neurons],
            'summary': self.summary,
            'summary_keyword': self.summary_keyword
        }

        self.save_conversation(data)

        error = query_sentiment - attended_output
        self.attention_weights -= self.learning_rate * error
        self.feedback_weights -= self.learning_rate * error

        return attended_output

    def save_conversation(self, data):
        try:
            with open('conversation.json', 'r') as f:
                conversations = json.load(f)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            conversations = []

        conversations.append(data.copy())

        with open('conversation.json', 'w') as f:
            json.dump(conversations, f, indent=4)


agent = OpenAIAgent("wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0")
neural_model = NeuralModel(agent)

while True:
    input_data = input("Enter a prompt: ")
    output = neural_model.process_input(input_data)
    print(f"Output: {output}")
    print(f"Summary: {neural_model.summary}")
    print(f"Summary Keyword: {neural_model.summary_keyword}")

    continue_prompt = input("Do you want to continue? (yes/no): ")
    if continue_prompt.lower() != 'yes':
        break
