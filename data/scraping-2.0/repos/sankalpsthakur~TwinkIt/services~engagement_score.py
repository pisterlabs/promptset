import openai
import requests
import json
from earlybird import preprocess_tweet, load_earlybird_model
from tensorflow import keras

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Load Earlybird model
earlybird_model = load_earlybird_model('earlybird_trained_model_path')

def generate_tweets(prompt, n=5):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=280,
        n=n,
        stop=None,
        temperature=0.8,
    )
    
    return [choice.text.strip() for choice in response.choices]

def predict_engagement_scores(tweets, model):
    # Preprocess the tweets and extract features
    features = [preprocess_tweet(tweet) for tweet in tweets]

    # Get engagement score predictions for the tweets
    engagement_scores = model.predict(features)
    
    return engagement_scores

def optimize_tweets(prompt, model, iterations=3, top_k=2):
    generated_tweets = generate_tweets(prompt)

    for _ in range(iterations):
        engagement_scores = predict_engagement_scores(generated_tweets, model)
        top_tweets = sorted(zip(generated_tweets, engagement_scores), key=lambda x: -x[1])[:top_k]

        # Generate new variations of top-k tweets
        new_tweets = []
        for tweet, _ in top_tweets:
            new_variations = generate_tweets(prompt + " " + tweet)
            new_tweets.extend(new_variations)
        
        generated_tweets = new_tweets

    # Final evaluation of the generated tweets
    final_engagement_scores = predict_engagement_scores(generated_tweets, model)
    best_tweet = max(zip(generated_tweets, final_engagement_scores), key=lambda x: x[1])
    
    return best_tweet

prompt = "Write a tweet about the benefits of AI in healthcare."
best_tweet, engagement_score = optimize_tweets(prompt, earlybird_model)

print("Best tweet:", best_tweet)
print("Engagement score:", engagement_score)
