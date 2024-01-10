import openai
import pandas as pd

# Set up OpenAI API client
openai.api_key = "YOUR_API_KEY"

# Define function to detect spam reviews using GPT-3 model
def detect_spam_review(review_text):
    # Fine-tune GPT-3 for spam detection
    model_engine = "text-davinci-002"
    prompt = "Is this review spam or not?\n\nReview: " + review_text + "\n\nAnswer:"
    completions = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1)
    answer = completions.choices[0].text.strip()
    
    # Return True if the answer is "spam", otherwise False
    if answer.lower() == "spam":
        return True
    else:
        return False