import pandas as pd
import openai
from dotenv import load_dotenv
import os
import textwrap

# Load your OpenAI API key from an environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the data
df = pd.read_csv('rekkles.csv')

# Create a new DataFrame with alternating rows merged
df['Title'] = df['Title'].iloc[::2].reset_index(drop=True)
df['Content'] = df['Content'].iloc[1::2].reset_index(drop=True)

# Drop any rows that still have missing data
df = df.dropna()

# Define a function to use GPT-3 for sentiment analysis
def sentiment_analysis(text):
    # Chunk the text into smaller pieces of approximately 3000 characters each
    chunks = textwrap.wrap(text, width=3000)

    analysis_results = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "system", "content": "This is a sentiment analysis assistant."},
                {"role": "user", "content": chunk},
            ],
            temperature=0.3,
            max_tokens=12000
        )
        analysis_results.append(response['choices'][0]['message']['content'].strip())
    
    # Combine the analysis results from all chunks
    return " ".join(analysis_results)

# Apply the sentiment analysis to the 'Content' column
df['Sentiment_Analysis'] = df['Content'].apply(sentiment_analysis)

# Print the sentiment analysis
print(df['Sentiment_Analysis'])
