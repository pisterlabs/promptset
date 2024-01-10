from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

#embeddings模型加载，如复现请下载m3e-base
model_name = "/home/bingxing2/home/scx6d3v/.cache/modelscope/hub/thomas/m3e-base"
model_kwargs = {'device': 'cuda'}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

def create_kb(file_path):
#暂时用不到
    return


def do_search(query="",
              save_path=r"res/kb/m3e-base/index.faiss",
              top_k=3,
              filter_dict=None,
              score_threshold=None,
              ):
    db = FAISS.load_local(save_path, embeddings)
    #词向量
    embedding_vector = embeddings.embed_query(query)
    result=db.similarity_search_with_score_by_vector(embedding_vector,k=top_k,filter=filter_dict,score_threshold=score_threshold)
    return result


def create_context_prompt(result:list)->list|str:
    prompt=[]
    prompt_str=""
    i=0
    for data in result:
        i+=1
        content,score=data
        content=dict(content)
        prompt.append((content["page_content"],score))
        prompt_str+="{}、".format(i)+content["page_content"]+"\n"
    return prompt,prompt_str








