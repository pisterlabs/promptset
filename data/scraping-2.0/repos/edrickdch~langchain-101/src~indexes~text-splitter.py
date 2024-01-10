from langchain.text_splitter import RecursiveCharacterTextSplitter


with open("example_data/state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)
texts = text_splitter.create_documents([state_of_the_union])
print(f"\nFirst chunk: {texts[0]}\n")
print(f"Second chunk: {texts[1]}\n")
