import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarise_text(text, max_length=2048):
    chunk_size = max_length - 200  # Account for the instructions and other tokens
    chunks = split_into_chunks(text, chunk_size)
    summaries = []

    for chunk in chunks:
        try:
            summary = summarise_chunk(chunk)
            summaries.append(summary)
        except openai.error.InvalidRequestError as e:
            print(f"Error: {e}")
            print("Trying with a smaller chunk size...")
            smaller_chunk_size = chunk_size // 2
            smaller_chunks = split_into_chunks(chunk, smaller_chunk_size)

            for smaller_chunk in smaller_chunks:
                try:
                    summary = summarise_chunk(smaller_chunk)
                    summaries.append(summary)
                except openai.error.InvalidRequestError as e:
                    print(f"Error: {e}")
                    print("Skipping this chunk...")

    final_summary = " ".join(summaries)
    final_summary = summarise_chunk(final_summary, 2)

    return final_summary


def split_into_chunks(text, chunk_size):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def summarise_chunk(chunk, num_sentences=1):
    prompt = f"Co jest przedmiotem zamówienia w: \n{chunk}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    summary = response.choices[0].text.strip()
    return summary

if __name__ == "__main__":
    text = "input_text_here"
    summary = summarise_text(text)
    print(summary)
