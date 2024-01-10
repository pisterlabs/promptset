# %%
# On jupyter notebook you can uncomment the below lines to install the packages
# !pip install openai
# !pip install langchain
# !pip install pypdf
# !pip install chromadb
#%%
import os
from notion_client import Client
# os.environ["NOTION_TOKEN"] =
# os.environ["OPENAI_API_KEY"] =
#%%
def QA_notion_blocks(Q, A, refs=()):
    """
    notion.blocks.children.append(page_id, children=QA_notion_blocks("Q1", "A1"))

    :param Q: str question
    :param A: str answer
    :param refs: list of str references
    :return:
    """
    ref_blocks = []
    for ref in refs:
        ref_blocks.append({'quote': {"rich_text": [{"text": {"content": ref}}]}})
    return [
        {'paragraph': {"rich_text": [{"text": {"content": f"Question:"}, 'annotations': {'bold': True}}, ]}},
        {'paragraph': {"rich_text": [{"text": {"content": Q}}]}},
        {'paragraph': {"rich_text": [{"text": {"content": f"Answer:"}, 'annotations': {'bold': True}}, ]}},
        {'paragraph': {"rich_text": [{"text": {"content": A}}]}},
        {'toggle': {"rich_text": [{"text": {"content": f"Reference:"}, 'annotations': {'bold': True}}, ],
                    "children": ref_blocks, }},
        # {'paragraph': {"rich_text": [{"text": {"content": f"Reference:"}, 'annotations': {'bold': True}}, ]}},
        # *ref_blocks,
        {'divider': {}}, # 'divider': {}
    ]
#%%
from langchain.document_loaders import PyPDFLoader, BSHTMLLoader, NotionDBLoader, UnstructuredURLLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
from langchain.vectorstores import Chroma  # for the vectorization part
from langchain.chains import ChatVectorDBChain  # for chatting with the pdf
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)
from langchain.chat_models import ChatOpenAI
import textwrap
from os.path import join
import pickle as pkl
import urllib.request
#%%
# pdf_path = "2304.02642.pdf"
arxiv_id = "1906.04358"
pdf_path = f"{arxiv_id}.pdf"
# loader = PyPDFLoader(pdf_path)

urlname = f"Progressive neuronal plasticity in primate visual cortex during stimulus familiarization"
url = 'https://www.science.org/doi/full/10.1126/sciadv.ade4648'
html_path = f"{urlname}.html"

response = urllib.request.urlopen(url)
webContent = response.read().decode('UTF-8')
with open(html_path, 'w') as f:
    f.write(webContent)

loader = BSHTMLLoader(html_path)
pages = loader.load_and_split()
print(pages[0].page_content)

#%%
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory=arxiv_id, )
vectordb.persist()
#%%

notion = Client(auth=os.environ["NOTION_TOKEN"])

database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
entries_return = notion.databases.query(database_id=database_id, filter={
      "property": "Name", "title": { "contains": arxiv_id }})

entry = entries_return["results"][0]
page_id = "c32a0e2d-cb6e-4c26-8168-71197aaeb082"  # entry["id"]
#%%
qa_path = pdf_path.replace(".pdf", "_qa_history")
os.makedirs(qa_path, exist_ok=True)
pkl_path = os.path.join(qa_path, "chat_history.pkl")
if os.path.exists(pkl_path):
    chat_history = pkl.load(open(pkl_path, "rb"))
else:
    print("No chat history found, creating new one")
    chat_history = [] # if not os.path.exists(pkl_path) else pkl.load(open(pkl_path, "rb"))

pdf_qa_new = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                    vectordb.as_retriever(), return_source_documents=True, max_tokens_limit=4097)

#%%
query = "For robotics purpose, which algorithm did they used, PPO, Q-learning, etc.?"

result = pdf_qa_new({"question": query, "chat_history": ""})
#%
print("Answer:")
print(textwrap.fill(result["answer"], 80))
print("\nReferences")
for doc in result["source_documents"]:
    print(doc.page_content[:100])
    print("\n")

# save Q&A chat history
with open(os.path.join(qa_path, "query.txt"), "a") as f:
    f.write(query)
    f.write("\n\n")

with open(os.path.join(qa_path, "QA.txt"), "a", encoding="utf-8") as f:
    f.write("\nQuestion:\n")
    f.write(query)
    f.write("\nAnswer:\n")
    f.write(result["answer"])
    f.write("\n\nReferences:\n")
    for doc in result["source_documents"]:
        f.write(doc.page_content[:100])
        f.write("\n")
    f.write("\n")
chat_history.append((query, result))
pkl.dump(chat_history, open(pkl_path, "wb"))


answer = result["answer"]
refdocs = result['source_documents']
refstrs = [refdoc.page_content[:250] for refdoc in refdocs]
notion.blocks.children.append(page_id, children=QA_notion_blocks(query, answer, refstrs));
#%%

loader = UnstructuredURLLoader(urls=["https://www.cnn.com/2023/03/30/politics/donald-trump-indictment/index.html",
                           "https://www.cnn.com/politics/live-news/trump-indictment-stormy-daniels-news-04-03-23/index.html",
                           "https://www.cnn.com/politics/live-news/donald-trump-court-charges-04-05-23/index.html"])

# docs = loader.load()
pages = loader.load_and_split()

#%%
save_dir = r"trump_indicted"
qa_path = save_dir + "_qa_history"
os.makedirs(qa_path, exist_ok=True)
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory=save_dir, )
vectordb.persist()
html_qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                            vectordb.as_retriever(), return_source_documents=True, max_tokens_limit=4096)
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
# query = "Trump 为何被指控？指控的内容是什么？"
# query = "Trump 被指控与1月6日围攻美国国会的人员有关吗？"
query = "Trump与情妇的指控具体是什么内容，是何时发生的事件？"
result = html_qa({"question": query, "chat_history": ""})
print("Answer:")
print(textwrap.fill(result["answer"], 45))
save_qa_history(query, result, qa_path)
