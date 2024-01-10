import openai
from langchain.llms import OpenAI
import os

# Initialize OpenAI (GPT-4) with your API key
openai.api_key = os.environ['OPENAI_API_KEY']
llm = OpenAI(model="gpt-4")

def split_text(text, chunk_size=500):
    # Split the text into chunks of 'chunk_size'
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def iteratively_summarize(text, llm, initial_summary="", max_tokens=150):
    chunks = split_text(text)
    current_summary = initial_summary

    for chunk in chunks:
        # Combine the current summary with the next chunk
        prompt = f"Update the summary: \"{current_summary}\" with the following text: \"{chunk}\""

        # Generate the updated summary
        completion = llm.generate(prompt, max_tokens=max_tokens)
        current_summary = completion.choices[0].text.strip()

    return current_summary

# Example usage
long_text = "Your long text here..."
summary = iteratively_summarize(long_text, llm)
print("Final Summary:", summary)