from langchain.text_splitter import SpacyTextSplitter

# Load a long document

with open('LLM.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()
    
# Instantiate the SpacyTextSplitter with the desired chunk size

text_splitter = SpacyTextSplitter(
    chunk_size=500,
    chunk_overlap=20
)

# Split the text using SpacyTextSplitter

texts = text_splitter.split_text(sample_text)

# Print the first chunk
print(texts[0])