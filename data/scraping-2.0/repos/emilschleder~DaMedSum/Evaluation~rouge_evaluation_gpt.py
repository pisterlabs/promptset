from openai import OpenAI
from evaluate import load
import pandas as pd

# Load the CSV file
df = pd.read_csv('evaluation/data/rouge_summaries_test.csv')

# Initialize the Rouge metric
rouge = load("rouge")

# Initialize the OpenAI API
client = OpenAI(api_key='<TOKEN>')
model = "gpt-3.5-turbo"

# Function to generate summary using ChatGPT 3.5-turbo
def generate_summary_chatgpt(text):
    response = client.chat.completions.create(
      model = model,  
       messages=[ { "role": "user", 
                    "content": f"Skriv et resume af den følgende tekst på dansk. Resumeet må ikke være længere end maksimalt 60 ord:\n\n{text}"} ]
    )
    return response.choices[0].message.content

# Initialize the list for the results
results = []

# Generate summaries for each text in the dataset
for index, row in df.iterrows():
    text, true_summary = row['text'], row['summary']
    generated_summary = generate_summary_chatgpt(text)
    print(f"Generated summary for text{index+1}... = {generated_summary}")
    print(len(generated_summary))
    scores = rouge.compute(predictions=[generated_summary], references=[true_summary])

    # Save the scores and the generated summary
    results.append({
        "Model": model,
        "TextID": f'text{index}',
        "ROUGE-1": scores["rouge1"] * 100,
        "ROUGE-2": scores["rouge2"] * 100, 
        "ROUGE-L": scores["rougeL"] * 100, 
        "ROUGE-Lsum": scores["rougeLsum"] * 100, 
        "summary": generated_summary,
        "text_length": len(text),
        "summary_length": len(generated_summary),
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
df_means = results_df.groupby('Model')[['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum']].mean()

# Print the results
print(df_means)
print(results_df)

# Save the results in csv files
results_df.to_csv('evaluation/results/all/summary_evaluation_results_gpt.csv', index=False)
df_means.to_csv('evaluation/results/mean/summary_evaluation_results_gpt_means.csv', index=False)