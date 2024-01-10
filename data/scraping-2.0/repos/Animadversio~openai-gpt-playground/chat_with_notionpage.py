import os
from notion_client import Client
from langchain.document_loaders import PyPDFLoader, BSHTMLLoader, NotionDBLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
from langchain.chat_models import ChatOpenAI
import textwrap
from os.path import join
import pickle as pkl

# os.environ["NOTION_TOKEN"] =
# os.environ["OPENAI_API_KEY"] =
#%%
database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
notion = Client(auth=os.environ["NOTION_TOKEN"])
loader = NotionDBLoader(os.environ["NOTION_TOKEN"], database_id)
#%%
entries_return = notion.databases.query(database_id=database_id, filter={
      "property": "Name", "title": { "contains": "Progressive" }})
#%%
# print (entries_return)
def print_entries(entries_return):
    # formating the output, so Name starts at the same column
    # pad the string to be 36 character
    print("id".ljust(36), "\t", "Name",)
    for entry in entries_return["results"]:
        print(entry["id"], "\t", entry["properties"]["Name"]["title"][0]["plain_text"], )


print_entries(entries_return)
#%%
# clean up metadata
def clean_metadata(metadata):
    metadata_new = {}
    for k, v in metadata.items():
        if v is None or v == []:
            continue
        metadata_new[k] = metadata[k]
    return metadata_new
#%%
page_entry = entries_return["results"][0]
page_title = page_entry["properties"]["Name"]["title"][0]["plain_text"]
doc = loader.load_page(page_entry["id"])
#%%
doc.metadata = clean_metadata(doc.metadata)
page_save_name = page_title.replace("|", " ").replace(":", " ").replace(" ", "_")
pages = RecursiveCharacterTextSplitter().split_documents([doc])
#%%
# wrap the text to 80 characters and print it
print(textwrap.fill(pages[0].page_content, 80))
#%%
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory=page_save_name, )
vectordb.persist()
#%%
notion_qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                    vectordb.as_retriever(), return_source_documents=True, max_tokens_limit=4097)
#%%
qa_path = page_save_name + "_qa_history"
os.makedirs(qa_path, exist_ok=True)
#%%
def save_qa_history(query, result, qa_path,):
    uid = 0
    while os.path.exists(join(qa_path, f"QA{uid:05d}.pkl")):
        uid += 1
    pkl.dump((query, result), open(join(qa_path, f"QA{uid:05d}.pkl"), "wb"))

    pkl_path = join(qa_path, "chat_history.pkl")
    if os.path.exists(pkl_path):
        chat_history = pkl.load(open(pkl_path, "rb"))
    else:
        chat_history = []
    chat_history.append((query, result))
    pkl.dump(chat_history, open(pkl_path, "wb"))

    with open(os.path.join(qa_path, "QA.md"), "a", encoding="utf-8") as f:
        f.write("\n**Question:**\n\n")
        f.write(query)
        f.write("\n\n**Answer:**\n\n")
        f.write(result["answer"])
        f.write("\n\nReferences:\n\n")
        for doc in result["source_documents"]:
            f.write("> ")
            f.write(doc.page_content[:250])
            f.write("\n\n")
        f.write("-------------------------\n\n")

#%%
# query = "What's the difference of novel stimuli and familiar stimuli in their response?"
query = "How does the paper define novel and faimilar stimuli in their method?"
query = "Why do they decide to use the intermittent stimuli set? What's conclusion about them? "
query = "How is this paper related to sleep and memory consolidation?"
query = "What is difference between the neurons with different time constants?"
query = "How is the neural response adaptation related to sleep?"
result = notion_qa({"question": query, "chat_history": ""})

print("Answer:")
print(textwrap.fill(result["answer"], 80))
save_qa_history(query, result, qa_path)
# print("\nReferences")
# for doc in result["source_documents"]:
#     print(doc.page_content[:100])
#     print("\n")
