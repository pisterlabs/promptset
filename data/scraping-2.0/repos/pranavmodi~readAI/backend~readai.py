from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of linked list in programming."}
  ]
)

def read_book_chapter(book_summary, chapter_text):
    ## For every chapter, get summary from openai and add to context
    system_prompt = "You are a book reader, skilled in reading chapters and summarizing them. You will be provided a summary of earlier chapters in the book. Generate a new summary incorporating the current chapter. If the summary is empty, just wait for the first chapter text"
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt + book_summary},
            {"role": "user", "content": chapter_text}
        ]
    )
    ccMessage = completion.choices[0].message
    cumulative_book_summary = ccMessage.content

    return cumulative_book_summary