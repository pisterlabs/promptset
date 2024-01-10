#Create a openai based summarizer that can summarize a given text. the text will be above the token limit for openai (4k) so you must split it into chunks (each size 4k) and then summarize each chunk and then combine the chunks into one summary.

import openai

# Define your OpenAI API key and model ID
api_key = "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"
model_id = "gpt-3.5-turbo"  # You can adjust the model as needed

# Initialize the OpenAI API client
openai.api_key = api_key

# Function to summarize a given text
def summarize_text(text):
    chunk_size = 4000  # Maximum token limit per request
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    for chunk in chunks:
        # Generate a summary for each chunk
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
       {"role": "user", "content": chunk}
    ],
            max_tokens=3000,  # You can adjust this limit for desired summary length
            temperature=0.7,  # You can adjust the temperature parameter          what does temp mean
        )
        summary_chunk = response.choices[0].message.content
        summaries.append(summary_chunk)

    # Combine the summaries into one final summary
    final_summary = ' '.join(summaries)
    return final_summary

# Read the long text from a TXT file
with open("text.txt", "r") as file:
    long_text = file.read() #whats this for

# Call the summarize_text function with the long text
final_summary = summarize_text(long_text)

# Save the final summary to a text file
with open("summary.txt", "w") as output_file:
    output_file.write(final_summary)

# Print the final summary
print(final_summary)
