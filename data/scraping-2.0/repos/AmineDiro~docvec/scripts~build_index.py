import json
import os
import struct
from itertools import chain
from re import split

import wikipediaapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

file_path = "page_content.txt"
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        text_content = f.read()
else:
    wiki_html = wikipediaapi.Wikipedia(
        user_agent="MyProjectName (merlin@example.com)",
        language="en",
        # extract_format=wikipediaapi.ExtractFormat.HTML,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )
    page = wiki_html.page("Python_(programming_language)")
    text_content = page.text
    with open(file_path, "w") as f:
        f.write(page.text)

content = [c for c in text_content.split("\n") if len(c) > 0]
split_content = list(chain(*[text_splitter.split_text(c) for c in content]))

# Embed content
model = SentenceTransformer("./gte-small")
embeddings = model.encode(split_content, batch_size=16)

# Save content list
with open("../data/index_content.txt", "w") as f:
    f.write("\n".join(split_content))

# Save Index embedings
data = embeddings.flatten().tolist()
buffer = struct.pack(f"<{len(data)}f", *data)
with open("../data/index_embedding.bin", "wb") as f:
    f.write(buffer)
