import openai
from concurrent.futures import ThreadPoolExecutor
import tiktoken
from sentence_transformers import SentenceTransformer, util
import torch
import openai
from youtube_transcript_api import YouTubeTranscriptApi

# Add your own OpenAI API key
openai.api_key = ""

model = SentenceTransformer("bert-base-nli-mean-tokens")
lines = []


def load_text(file_path):
    with open(file_path, "r") as file:
        return file.read()


def save_to_file(responses, output_file):
    with open(output_file, "w") as file:
        for response in responses:
            file.write(response + "\n")


# Change your OpenAI chat model accordingly
def call_openai_api(chunk):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be given the full text transcript of a YouTube video. Clean the data by splitting the entire transcript into individual sentences while maintaining correct grammar, and make paragraphs by grouping 1-3 sentences that are similar in content together. Return these paragraphs.",
            },
            {"role": "user", "content": f"YOUR DATA TO PASS IN: {chunk}."},
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0]["message"]["content"].strip()


def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(" ".join(encoding.decode(words[i : i + tokens])))
    return chunks


def process_chunks(input_file, output_file):
    text = load_text(input_file)
    chunks = split_into_chunks(text)

    # Processes chunks in parallel
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_openai_api, chunks))

    save_to_file(responses, output_file)


# Specify your input and output files
if __name__ == "__main__":
    print("starting transcript...")

    # transcribe given youtube video
    transcript_dict = YouTubeTranscriptApi.get_transcript("gIBEnSIM7W4")
    transcript_stringified = ""

    for t in transcript_dict:
        transcript_stringified += t["text"] + " "

    print("transcript: " + transcript_stringified)

    # write transcription to file
    f = open("input.txt", "a")
    f.write(transcript_stringified)
    f.close()

    # use infinitegpt to clean and group transcript
    input_file = "input.txt"
    output_file = "output.txt"

    print("cleaning data with infinitegpt...")

    process_chunks(input_file, output_file)

    print("done cleaning!")

    # open output.txt and parse into string array
    with open("./output.txt", "r") as f:
        for line in f:
            if line.strip() != "":
                lines.append(line.strip())

    print(lines)
    print("parsed data into string array!")

    # create embeddings with sentence transformers
    embeddings = model.encode(lines)

    print("created embeddings with sentence transformers")

    # create query embedding
    query = "What does Rich Harris think about using Typescript in libraries?"
    query_embedding = model.encode(query)

    print("query is =>" + query)

    print("performing semantic search now...")

    # perform semantic search to find relevant text
    hits = util.semantic_search(query_embedding, embeddings, top_k=3)
    hits = hits[0]

    print("done with semantic search!")

    print(
        "crafting chatgpt response with relevant cleaned paragraphs from transcript..."
    )

    # search relevant paragraph
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be given a query sent by a user and the relevant content for the query retrieved by semantic serach. Generate a context-aware and brief response to the query that is conversationally accurate. Do not be wordy; be concise!",
            },
            {
                "role": "user",
                "content": f"QUERY: {query}, CONTENT: {lines[hits[0]['corpus_id']] + lines[hits[0]['corpus_id'] + 1]}",
            },
        ],
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    print("all done!")

    print(response.choices[0]["message"]["content"].strip())
