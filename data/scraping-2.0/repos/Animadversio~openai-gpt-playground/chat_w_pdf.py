# !curl -o paper.pdf https://arxiv.org/pdf/2303.13519.pdf
# !curl -o 1906.04358.pdf https://arxiv.org/abs/1906.04358
#%%
# On jupyter notebook you can uncomment the below lines to install the packages
# !pip install openai
# !pip install langchain
# !pip install pypdf
# !pip install chromadb
#%%
# download the pdf from arxiv
from os.path import join
import requests
def download_pdf(arxiv_id, save_root=""):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    r = requests.get(url, allow_redirects=True,)
    open(join(save_root, f"{arxiv_id}.pdf"), 'wb').write(r.content)


download_pdf("1906.04358")
#%%
import os
# os.environ["OPENAI_API_KEY"] =
# os.environ["NOTION_TOKEN"] =
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
from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
from langchain.chat_models import ChatOpenAI
import textwrap
from os.path import join
import pickle as pkl
# pdf_path = "2304.02642.pdf"
# arxiv_id = "1906.04358"
# pdf_path = f"{arxiv_id}.pdf"
pdf_path = f"sciadv.ade4648.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
print(pages[0].page_content)

"""
# Output:
Learning and Veriﬁcation of Task Structure in Instructional Videos
Medhini Narasimhan1;2, Licheng Yu2, Sean Bell2, Ning Zhang2, Trevor Darrell1
1UC Berkeley,2Meta AI
https://medhini.github.io/task_structure
Abstract
Given the enormous number of instructional videos
available online, learning a diverse array of multi-step task
models from videos is an appealing goal. We introduce
......
"""
#%%
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory=pdf_path.replace(".pdf", ""), )
vectordb.persist()
#%%
import os
from notion_client import Client
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
# query = "How is this method different from other image generative models like diffusion?"
# query = "How is this method using the image encoding model? how is it related to diffusion models?"
# query = "Is this model related to Stable Diffusion model, can it be applied there?"
# query = "How is transformer or attention mechanism used in this segmentation model? and is there a language input to this model?"
# query = "How is this method related to traditional methods of image segmentation? e.g. U-Net, Mask-RCNN, etc."
# query = "How is this method related to traditional methods of image segmentation? e.g. U-Net, Mask-RCNN, etc."
# query = "As a foundation model, how is self-supervised learning used in this model? What's the objective? how is it different from previous self-supervised learning in vision"
# query = "What is the supervised learning task for this foundation model? where did they get the data from more than previous segmentation models?"
# query = "Can you be more specific about how their iterative data collection cycle is made?"
# query = "How is this method optimized if it's not optimizing the weights?"
# query = "How is the neural network topology parameterized in this model?"
# query = "How is this method related to Evolutionary Algorithms?"
# query = "How is this method different from NeuroEvolution of Augmenting Topologies (NEAT)?"
# query = "What's the major application of this method?"
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

# Output
'''
How is this method using the image encoding model? how is it related to diffusion models?

PersistentDuckDB del, about to run persist
Persisting DB to disk, putting it in the save folder .

Answer:
This method uses an image encoding model to capture high-level identifiable
semantics of objects, producing an object-specific embedding with only a single
feed-forward pass. This acquired object embedding is then passed to a text-to-
image synthesis model for subsequent generation. The text-to-image synthesis
model used is a diffusion model called Imagen, which is trained with a denoising
objective. The object embedding and the text embedding are used as the two
conditions for object-specific generation. The framework aims to blend an
object-aware embedding space into a well-developed text-to-image model under the
same generation context. The authors investigate different network designs and
training strategies and propose a regularized joint training scheme with an
object identity preservation loss. The overall goal is to generate images of
customized objects specified by users while avoiding the lengthy optimization
typically required by previous approaches.
'''

"""
Is this model related to Stable Diffusion model, can it be applied there?

Answer:
There is no mention of Stable Diffusion model in the provided context, so it is
unclear whether this model is related to it or can be applied there. It would be
best to seek further information on Stable Diffusion model and its applicability
to this model.
"""