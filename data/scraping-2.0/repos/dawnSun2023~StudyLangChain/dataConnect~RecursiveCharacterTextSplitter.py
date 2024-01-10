with open('../quest2.txt', encoding='utf-8') as f:
    state_of_the_union = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 200,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)
#返回的是document
texts = text_splitter.create_documents([state_of_the_union])
print(texts)
print(texts[0])
print(texts[1])
#返回的是个数组
tx = text_splitter.split_text(state_of_the_union)[:2]
print(tx)