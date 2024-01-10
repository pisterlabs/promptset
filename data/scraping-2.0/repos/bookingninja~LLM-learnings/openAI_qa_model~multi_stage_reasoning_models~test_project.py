# Working with larger documents
# Instead of a local text document, we'll download the complete works of Shakespeare using the `GutenbergLoader` that works with the Gutenberg project: https://www.gutenberg.org

from langchain.document_loaders import GutenbergLoader

loader = GutenbergLoader(
    "https://www.gutenberg.org/cache/epub/100/pg100.txt"
)  # Complete works of Shakespeare in a txt file

all_shakespeare_text = loader.load()


# TEMPLATE (assume some of the early commands are executed from @example_project.py 

# text_splitter = <FILL_IN> #hint try chunk sizes of 1024 and an overlap of 256 (this will take approx. 10mins with this model to build our vector database index)
# texts = <FILL_IN>

# model_name = <FILL_IN> #hint, try "sentence-transformers/all-MiniLM-L6-v2" as your model
# embeddings = <FILL_IN>
# docsearch = <FILL_IN>

#USE THIS 
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
texts = text_splitter.split_documents(all_shakespeare_text)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
  model_name=model_name, cache_folder=DA.paths.datasets
) #using a pre-cached model 
chromadb_index = Chroma.from_documents(
    texts, embeddings, persist_directory=DA.paths.working_dir
)

####MORE TO COME
