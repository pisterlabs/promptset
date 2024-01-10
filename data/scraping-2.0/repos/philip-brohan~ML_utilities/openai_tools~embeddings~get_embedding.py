#!/usr/bin/env python

# Convert a text string into a token set and then an embedding.

#import pandas as pd
import openai
import tiktoken

from openai.embeddings_utils import get_embedding

from openai_tools.config.access import get_key
openai.api_key = get_key()

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

tokenizer = tiktoken.get_encoding(embedding_encoding)

# Text to use
prompt = (
    "Amidst the winter backdrop, the sailor emanated an aura of mystique. "
    + "His tunic, a rich hue of midnight blue, "
    + "draped effortlessly over a well-built frame. His bare arms"
    + "displayed interwoven maritime and oriental tattoos. From his intricately stitched belt "
    + "hung a scimitar. Its curved blade shimmered, ornate patterns etched onto its surface, "
    + "its hilt was encrusted with turquoise and carnelian."
)

# This works locally
tokens = tokenizer.encode(prompt)

# And it's reversible
original = tokenizer.decode(tokens) 

# This is an openai call - needs the API key set
# This is not reversible
embedding = get_embedding(prompt, engine=embedding_model)

