from langchain.text_splitter import RecursiveCharacterTextSplitter


with open('./state_of_the_union.txt') as f:
    state_of_the_union = f.read()

# default split on are ["\n\n", "\n", " ", ""]
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20, # max overlap
    length_function = len,
    add_start_index = True,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
"""
page_content='Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and' metadata={'start_index': 0}
"""
print(texts[1])
"""
page_content='of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.' metadata={'start_index': 82}
"""