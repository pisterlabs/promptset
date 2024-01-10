from llmlib.openai import OpenAiWrapper

llm = OpenAiWrapper()

llm.create_local_embeddings_files_in_dir("./data/")

a = llm.query_local_embeddings("definition of sports")
print(a)

a = llm.query_local_embeddings("activities in sports")
print(a)