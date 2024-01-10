# Reads responses to free text questions and uses GPT to analyse sentiment
# Writes output to file

import openai
import json
import pandas as pd


from tqdm import tqdm

def sentiment(file_path, secrets_path):
    # Load API key from secrets.json
    with open(secrets_path) as f:
        secrets = json.load(f)
    openai.api_key = secrets['OPENAI_API_KEY']

    # Load the data
    data = pd.read_csv(file_path)

    print("Analysing sentiment using text-davinci-003. This may take some time ...")

    # Columns to analyze
    columns_to_analyze = [
        'What are things that make your work enjoyable and fulfilling? Have these things become more or less common in your recent experience? Are there factors that make your role eu-stressful (stressful and challenging, but in a good way)? Are these factors more or less present in your recent experience?',
        'What do you experience in your work that makes it (dis)stressful, unpleasant, and/or unfulfilling to perform your role? How commonplace are these experiences?',
        'Do you have any thoughts about how your work could be redesigned to be less (dis)stressful?',
        'If you could make one single positive improvement to your work or workplace; one that would make the most difference to you, what would it be?',
        'Do you have any other comments about your work situation or role? Is there anything previous questions failed to allow you to express regarding workplace stressors?'
    ]

    # Iterate through each row
    for i in tqdm(range(len(data)), desc="Progress"):  # Add progress bar here
        row = data.iloc[i]
        # Iterate through each column
        for column in columns_to_analyze:
            text = row[column]
            if pd.isnull(text):
                continue

            # Generate the sentiment
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=f"This is a sentiment classification task. The question was: '{column}'. The response was: '{text}'. Classify the sentiment of the response only using a single word: Positive, Negative, or Neutral.",
                temperature=0,
                max_tokens=60,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            sentiment = response.choices[0].text.strip()

            # Update the cell with the sentiment
            data.at[i, column] = f"({sentiment}). {text}"



    # Save the updated data back to the CSV file
    data.to_csv(file_path, index=False)

    print("Sentiment analysis complete.\nCheck manually before continuing")



