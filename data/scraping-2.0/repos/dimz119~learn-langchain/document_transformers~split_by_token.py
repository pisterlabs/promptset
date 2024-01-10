from langchain.text_splitter import CharacterTextSplitter


# required `pip install tiktoken`
with open('./state_of_the_union.txt') as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(state_of_the_union)
print(len(texts))
print(texts[0])
