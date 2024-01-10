from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('example_data/state_of_the_union.txt') as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print(texts[1])

output = text_splitter.split_text(state_of_the_union)[:2]
print(output)
