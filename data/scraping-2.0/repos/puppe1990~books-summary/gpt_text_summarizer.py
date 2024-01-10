import openai
import sys
import re
import time

OPENAI_API_KEY = 'your-api-key-here'
openai.api_key = OPENAI_API_KEY


def tokenize(text: str):
    response = openai.Completion.create(
        engine="text-davinci-002", prompt=text, n=1, max_tokens=0, temperature=0)
    return response['usage']['total_tokens']


def split_text(text: str, max_tokens: int = 1500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for word in words:
        word_tokens = count_tokens(word)
        if current_chunk_tokens + word_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_chunk_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_chunk_tokens += word_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def summarize_text(text: str, max_length: int = 80) -> str:
    prompt = f"Please summarize the following text:\n\n{text}\n\n. Summary:"
    response = openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        max_tokens=max_length,
        n=1,
        stop=None,
        temperature=0.5,
    )

    time.sleep(5)  # Wait for 5 seconds
    summary = response.choices[0].text.strip()
    return summary


def count_tokens(text: str) -> int:
    return len(re.findall(r'\w+', text))


def generate_blog_post(summaries: list, max_length: int = 1000) -> str:
    blog_post = ""
    for summary in summaries:
        prompt = f"Create a blog post using the following summary:\n\n{summary}\n\nBlog post:"
        response = openai.Completion.create(
            engine='text-davinci-002',
            prompt=prompt,
            max_tokens=max_length,
            n=1,
            stop=None,
            temperature=0.5,
        )

        time.sleep(5)  # Wait for 5 seconds
        blog_post += response.choices[0].text.strip() + "\n\n"

    return blog_post


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} input_file")
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        input_text = f.read()

    chunks = split_text(input_text)
    summaries = [summarize_text(chunk) for chunk in chunks]
    print("Combined Summary:")
    for summary in summaries:
        print(summary)

    blog_post = generate_blog_post(summaries)
    print("\nGenerated Blog Post:")
    print(blog_post)

    # Save the summaries and blog post to a text file
    with open('summary.txt', 'w') as f:
        f.write("Combined Summary:\n")
        for summary in summaries:
            f.write(summary + '\n\n')

        f.write("\nGenerated Blog Post:\n")
        f.write(blog_post)
