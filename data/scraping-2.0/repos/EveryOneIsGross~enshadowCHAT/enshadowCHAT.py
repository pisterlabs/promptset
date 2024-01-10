import numpy as np
import openai
import concurrent.futures
import json
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
import dotenv
import os
import re
from tenacity import retry, stop_after_attempt, wait_fixed


dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize sentiment analyzer and keyword extractor
sia = SentimentIntensityAnalyzer()
rake = Rake()
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai_api(prompt, temperature):
    return openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=temperature
    )

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai_chat_api(prompt):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a holistic summarizer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

class Chatbot:
    def __init__(self, base_temperature):
        self.temperature = base_temperature
        self.history = []
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    def get_embeddings(self, text):
        return self.model.encode(text)

    def get_previous_context(self, current_embedding):
        if self.history:
            previous_embeddings = [embedding for _, _, embedding in self.history]
            similarities = cosine_similarity([current_embedding], previous_embeddings)
            most_similar_index = np.argmax(similarities)
            return self.history[most_similar_index][0]
        return None

    def get_response(self, prompt):
        current_embedding = self.get_embeddings(prompt)
        
        previous_context = self.get_previous_context(current_embedding)
        if previous_context:
            prompt += ' ' + previous_context

        if self.history:
            last_sentiment = self.history[-1][1]
            if last_sentiment['compound'] < 0:
                self.temperature += 0.1
            else:
                self.temperature = max(self.temperature - 0.1, 0.1)

        response = call_openai_api(prompt, self.temperature)
        text = response.choices[0].text.strip()
        sentiment = sia.polarity_scores(text)
        self.history.append((text, sentiment, current_embedding))

        return text, sentiment, current_embedding

class rankBOT:
    def __init__(self):
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    def rank_responses(self, responses, prompt):
        response_embeddings = self.model.encode(responses)
        prompt_embedding = self.model.encode([prompt])
        similarities = cosine_similarity(response_embeddings, prompt_embedding)
        ranked_indices = np.argsort(similarities.flatten())
        worst_response_index = ranked_indices[0]
        return worst_response_index

class HolisticSum:
    def __init__(self):
        self.question_history = []

    def update_history(self, question):
        self.question_history.append(question)

    def create_holistic_summary(self, best_response, shadow_response):
        history_prompt = " ".join(self.question_history)
        prompt = f"The response to the question is: '{best_response}'. However, on a deeper, unconscious level, the response could also be interpreted as: '{shadow_response}'. The conversation so far has been: '{history_prompt}'."
        
        response = call_openai_chat_api(prompt)
        
        return response.choices[0].message.content.strip()

class AnswersSum:
    def __init__(self):
        self.conversation_history = []

    def update_conversation(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def summarize_answer(self, response, answer_type):
        self.update_conversation("system", f"You are a {answer_type} summarizer.")
        prompt = f"The {answer_type} response to the question was: '{response}'."
        self.update_conversation("user", prompt)

        summary = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.conversation_history,
            max_tokens=250
        )

        return summary.choices[0].message.content.strip()


class Grid:
    def __init__(self, size, base_temperature):
        self.size = size
        self.initial_grid = [[Chatbot(base_temperature) for _ in range(size)] for _ in range(size)]
        self.shadow_grid = [[Chatbot(base_temperature) for _ in range(size)] for _ in range(size)]
        self.summary_bot = rankBOT()

    def process_prompt(self, prompt):
        initial_responses = []
        shadow_responses = []
        initial_bots = [bot for row in self.initial_grid for bot in row]
        shadow_bots = [bot for row in self.shadow_grid for bot in row]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            initial_futures = [executor.submit(bot.get_response, prompt) for bot in initial_bots]
            shadow_futures = [executor.submit(bot.get_response, prompt) for bot in shadow_bots]

            for future in initial_futures:
                try:
                    data = future.result()
                    initial_responses.append(data)
                except Exception as exc:
                    print('An initial bot generated an exception: %s' % exc)

            for future in shadow_futures:
                try:
                    data = future.result()
                    shadow_responses.append(data)
                except Exception as exc:
                    print('A shadow bot generated an exception: %s' % exc)

        initial_worst_response_index = self.summary_bot.rank_responses([text for text, _, _ in initial_responses], prompt)
        shadow_worst_response_index = self.summary_bot.rank_responses([text for text, _, _ in shadow_responses], prompt)

        if initial_worst_response_index is not None:
            initial_worst_response = initial_responses[initial_worst_response_index]
        else:
            initial_worst_response = (None, None, None)

        if shadow_worst_response_index is not None:
            shadow_worst_response = shadow_responses[shadow_worst_response_index]
        else:
            shadow_worst_response = (None, None, None)

        return initial_responses, shadow_responses, initial_worst_response, shadow_worst_response


def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

def extract_keywords(text):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def save_embeddings(embedding, filename):
    np.save(filename, embedding)

def load_embeddings(filename):
    return np.load(filename)

while True:
    prompt = input("\nEnter your prompt (type 'quit' to quit): ")
    if prompt.lower() == 'quit':
        break

    sentiment_scores = analyze_sentiment(prompt)
    keywords = extract_keywords(prompt)

    grid = Grid(2, 0.5)
    initial_responses, shadow_responses, initial_worst_response, shadow_worst_response = grid.process_prompt(prompt)

    initial_worst_response_text, initial_worst_response_sentiment, initial_worst_response_embedding = initial_worst_response
    shadow_worst_response_text, shadow_worst_response_sentiment, shadow_worst_response_embedding = shadow_worst_response

    if initial_worst_response_text is not None:
        shadow_grid = Grid(2, 1.5)
        initial_responses, shadow_responses, _, _ = shadow_grid.process_prompt(initial_worst_response_text)
    else:
        shadow_responses = []

    holistic_summary_bot = HolisticSum()

    best_response_index = grid.summary_bot.rank_responses([text for text, _, _ in initial_responses], prompt)
    shadow_best_response_index = shadow_grid.summary_bot.rank_responses([text for text, _, _ in shadow_responses], prompt)

    best_response = initial_responses[best_response_index][0]
    shadow_best_response = shadow_responses[shadow_best_response_index][0]

    final_summary = holistic_summary_bot.create_holistic_summary(best_response, shadow_best_response)

    # Create an instance of AnswersSum and use it to re-summarize the best and shadow responses
    answers_summary_bot = AnswersSum()
    best_response_summary = answers_summary_bot.summarize_answer(best_response, "best")
    shadow_response_summary = answers_summary_bot.summarize_answer(shadow_best_response, "shadow")

    #print("\nKeywords: ", keywords)
    #print("\nSentiment Scores: ", sentiment_scores)
    #print("\nBest Response: \n", best_response)
    print("\nBest Response Summary: \n", best_response_summary)
    #print("\nShadow Best Response: \n", shadow_best_response)
    print("\nShadow Response Summary: \n", shadow_response_summary)
    print("\nFinal summary: \n", final_summary)

    save_embeddings(grid.initial_grid[0][0].history[-1][2], 'current_embedding.npy')
    save_embeddings(initial_worst_response_embedding, 'worst_response_embedding.npy')

    # Save results to JSON
    with open('results.json', 'a') as json_file:
        data = {
            'prompt': prompt,
            'keywords': keywords,
            'sentiment_scores': sentiment_scores,
            'best_response': best_response,
            'best_response_summary': best_response_summary,
            'shadow_best_response': shadow_best_response,
            'shadow_response_summary': shadow_response_summary,
            'final_summary': final_summary,
        }
        json.dump(data, json_file)
        json_file.write("\n")
