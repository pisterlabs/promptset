#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# h/t: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

import nltk
import random
import torch
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from openai import OpenAI
from textblob import TextBlob
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

OPENAI_API_KEY = "openai_api_key"
api_key = OPENAI_API_KEY

df_1 = pd.read_csv("/content/drive/MyDrive/opendi/CS_Chatbot_newaccount (1).csv")
prompts_1 = df_1["Prompt"].tolist()
responses_1 = df_1["Response_Llama_70B_Chat"].tolist()
ideal_responses_1 = df_1["Ideal Response"].tolist()

"""
### Fitness Function
### Relevance Score
### use Bert model as embedding
"""

# Load pre-trained Sentence-BERT model
model_name = "sentence-transformers/bert-base-nli-mean-tokens"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# genetic algorithm
class GeneticAlgorithm():
    def encode(texts, tokenizer, model):
    # Tokenize and encode sequences in the batch
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Mean of the token embeddings to get sentence embedding
        embeddings = torch.mean(model_output.last_hidden_state, dim=1)
        return embeddings

    def semantic_similarity(response, ideal_response, tokenizer, model):
        # Encode both response and ideal response
        response_embedding = encode([response], tokenizer, model)
        ideal_response_embedding = encode([ideal_response], tokenizer, model)
        # Compute cosine similarity
        similarity = 1 - cosine(response_embedding[0], ideal_response_embedding[0])
        return similarity

    """#### Informativeness Score"""

    # Updated key information for all categories
    # Define keywords for each category
    category_keywords = {
        'Account Creation & Setup': ['sign up', 'register', 'account'],
        'Order Process & Management': ['order', 'shipping', 'delivery'],
        'Payment & Pricing': ['payment', 'price', 'cost'],
        'Navigation & Product Search': ['find', 'search', 'product'],
        'Account Management & Support': ['reset', 'password', 'support', 'account']
    }

    # Function to infer the category of a prompt based on keywords
    def infer_category(prompt):
        for category, keywords in category_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                return category
        return 'Unknown'  # Default category if no keywords match

    ## Adjusted informativeness score function
    def informativeness_score(response, category):
        # Check if key information for the given category is present in the response
        for key_phrase in category_keywords.get(category, []):
            if key_phrase in response:
                return 1
        return 0

    """#### Engagment Score"""

    def engagement_score(response):
        # Simple heuristic: check if the response asks a question or provides a call-to-action
        if '?' in response or 'click' in response or 'select' in response or 'visit' in response.lower():
            return 1
        return 0

    def sentiment_score(text):
        # Create a TextBlob object
        blob = TextBlob(text)
        # Return the polarity score of the text
        return blob.sentiment.polarity

    def combined_engagement_score(response, sentiment_weight=0.5, engagement_weight=0.5):
        # Suppose both sentiment_score and engagement_score return a value between 0 and 1
        sentiment = sentiment_score(response)  # This is your new sentiment function
        engagement = engagement_score(response)  # This is your existing engagement function

        # Normalize both scores if they aren't already
        normalized_sentiment = (sentiment + 1) / 2  # To transform from [-1, 1] to [0, 1]
        normalized_engagement = engagement  # Assuming this is already between 0 and 1

        # Calculate combined score with respective weights
        combined_score = (normalized_sentiment * sentiment_weight) + (normalized_engagement * engagement_weight)
        return combined_score

    """#### Overall Fitness Score"""

    # Calculate overall fitness for each response
    def fitness_score(responses):
        fitness_scores = []
        for i in range(len(responses)):
            # Calculate each component of fitness
            relevance = semantic_similarity(responses[i], ideal_responses_1[i], tokenizer, model)

            # Infer the category for each prompt and use it in informativeness_score
            category = infer_category(prompts_1[i])  # Use the inferred category
            informativeness = informativeness_score(responses[i], category)

            engagement = combined_engagement_score(responses[i])

            # Weights for each component (example weights)
            weight1, weight2, weight3 = 0.4, 0.4, 0.2
            overall_fitness = weight1 * relevance + weight2 * informativeness + weight3 * engagement

            fitness_scores.append(overall_fitness)
    return fitness_scores

    # Display fitness scores
    fitness_score(responses_1)

    fitness_values = fitness_score(responses_1)

    sorted(fitness_values,reverse=True)[:5]

    indexed_fitness = [(index, value) for index, value in enumerate(fitness_values)]
    sorted_indexed_fitness = sorted(indexed_fitness, key=lambda x: x[1], reverse=True)
    # Select the indices of the top 5 fitness values
    top_5_indices = [index for index, value in sorted_indexed_fitness[:5]]


    # Use these indices to select the corresponding items from prompts_1
    top_5_prompts = [prompts_1[index] for index in top_5_indices]
    top_5_prompts

    """### The other components of GA"""

    def select_parents(population, fitness_values):
        # Roulette wheel selection
        total_fitness = sum(fitness_values)
        selection_probabilities = [f / total_fitness for f in fitness_values]
        return random.choices(population, weights=selection_probabilities, k=len(population) // 2)

    """#### Crossover"""

    def crossover(parents):
        children = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = parent1 if random.random() < 0.5 else parent2
            children.append(child)
        return children

    """Because crossover by combining parts of parent prompts or using synonyms gave me combinations sometime they may not make logical or grammatical sense.e.g.'Why was my I report an issue with the website?' I'm going to only use simple crossover fuction and leave mutation function to generate more creative prompts.

    #### Mutation

    https://github.com/openai/openai-python

    #### Generate overall fitness scores
    """

    nltk.download('stopwords')

    # Get English stopwords
    stop_words = set(stopwords.words('english'))

    top_prompts = top_5_prompts

    # Filter out stopwords and count words
    word_counts = Counter(
        word.lower() for prompt in top_5_prompts for word in prompt.split() if word.lower() not in stop_words
    )

    # Identify most common words without stopwords
    common_words = word_counts.most_common(20)
    common_words

    # Identify most common words or themes
    common_words = word_counts.most_common(10)
    print("Common themes:", common_words)

    # Enhance intent recognition based on these themes
    def recognize_intent(user_input):
        # Simplified intent recognition logic
        if "password" in user_input:
            return "change_password"
        elif "account" in user_input:
            return "create_account"
        elif "update" in user_input:
            # Additional check to distinguish between different types of updates
            if "address" in user_input:
                return "update_address"
            elif "account" in user_input:
                return "update_account_info"
        # Add more conditions based on common themes
        else:
            return "unknown"


    # Test the function
    print("Recognized intent:", recognize_intent(top_5_prompts[0][0]))

    # Provided common themes with some words having trailing punctuation
    common_words_with_punctuation = [('account,', 3), ('steps', 3), ('generate', 2), ('user', 2),
                                    ('account', 2), ('efficiently?', 2), ('initiate', 2), ('follow?', 2),
                                    ('establish', 1), ('take?', 1)]

    # Function to clean and combine counts for words with and without trailing punctuation
    def clean_and_combine_counts(word_counts):
        cleaned_counts = {}
        for word, count in word_counts:
            # Remove punctuation at the end of the word (if any)
            clean_word = word.rstrip(',?!.')
            # Combine counts for the clean word
            if clean_word in cleaned_counts:
                cleaned_counts[clean_word] += count
            else:
                cleaned_counts[clean_word] = count
        return cleaned_counts

    # Clean and combine counts
    cleaned_word_counts = clean_and_combine_counts(common_words_with_punctuation)

    # Create a word cloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white', contour_width=3, contour_color='steelblue').generate_from_frequencies(cleaned_word_counts)

    # Plot the WordCloud image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis labels
    plt.tight_layout(pad=0)
    plt.show()

    # Extracted top words from the word cloud
    top_words = [word for word, count in common_words_with_punctuation[:8]]

    def gpt_mutate(sentence, api_key, top_words, model="gpt-3.5-turbo", temperature=1, max_tokens=60):
        client = OpenAI(api_key=api_key)

        try:
            # Instruction to include top words or themes
            instruction = "Rewrite the following sentence to be more engaging and include themes related to: " + ", ".join(top_words) + "."
            prompt = f"{instruction} Original sentence: '{sentence}'"

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            mutated_sentence = response.choices[0].message.content.strip()
            return mutated_sentence
        except Exception as e:
            print(f"Error during GPT-3 mutation: {e}")
            return sentence  # Return original sentence in case of an error

    num_prompts = len(prompts_1)
    population_size = 20
    num_generations = 30

    # Initialize population with random solutions
    population = [random.randint(0, num_prompts-1) for _ in range(population_size)]
    best_prompts = []

    for generation in range(num_generations):
        # Calculate fitness for each solution
        fitness_values = [fitness_values[sol] for sol in population]
        # Selection
        parents = select_parents(population, fitness_values)
        # Crossover
        children = crossover(parents)
        # Mutation
        gpt_mutate(children,api_key,top_words)
        # Create new generation
        population = children + parents
        # Optional: Print best solution in this generation
        best_sol = population[fitness_values.index(max(fitness_values))]
        # print(f"Generation {generation}: Best Prompt - {prompts_1[best_sol]}")
        best_prompts.append((prompts_1[best_sol],max(fitness_values)))

    # Final best solution
    best_overall_sol = population[fitness_values.index(max(fitness_values))]
    # Final best solution
    best_prompt_and_score = [prompts_1[best_overall_sol],max(fitness_values)]
    
    return best_prompt_and_score
