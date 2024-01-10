from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from models.chatglm_llm import ChatGLM
from configs.model_config import *
import datetime
from textsplitter import ChineseTextSplitter
from typing import List, Tuple
from langchain.docstore.document import Document
import numpy as np

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 1

# LLM input history length
LLM_HISTORY_LEN = 2


def load_file(filepath):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    print(docs)
    return docs

def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template=PROMPT_TEMPLATE) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def get_docs_with_score(docs_with_score):
    docs=[]
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i-1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists



def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, len(docs)-i)):
                for l in [i+k, i-k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
        id_list = sorted(list(id_set))
        id_lists = seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, scores[0][j]))
        return docs



class LocalDocQA:
    llm: object = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_history_len: int = LLM_HISTORY_LEN,
                 llm_model: str = LLM_MODEL,
                 llm_device=LLM_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 use_ptuning_v2: bool = USE_PTUNING_V2
                 ):
        self.llm = ChatGLM()
        self.llm.load_model(model_name_or_path=llm_model_dict[llm_model],
                            llm_device=llm_device,
                            use_ptuning_v2=use_ptuning_v2)
        self.llm.history_len = llm_history_len

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k


    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None):
        loaded_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath)
                    print(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_file(fullfilepath)
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
        if len(docs) > 0:
            def pre_process_documents(docs : list):
                new_docs = []
                tmp = docs[0]
                for i in docs:
                    if i == tmp:
                        continue
                    tmp.page_content += "\n"
                    tmp.page_content += i.page_content
                new_docs.append(tmp)
                return new_docs
            docs = pre_process_documents(docs)
            if vs_path and os.path.isdir(vs_path):
                print("\n [ymx10010]")
                print(vs_path)
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                vector_store.add_documents(docs)
                print(docs)
            else:
                print("\n [ymx10086]")
                print(docs)
                # docs = [Document(page_content='oepkgs社区软件包引入各版本分支原则：\n- 1. 软件包引入master保护分支，通过对应sig组maintainer或者对应源码仓committer review、approve即可。\n- 2. 软件包引入到oepkgs其他保护分支，需根据valid_release_branch，通过对应sig组的maintainer或者对应源码仓committer review、approve即可。', metadata={'source': '/root/langchain-ChatGLM/content/oepkgs分支管理.md', 'page_number': 1, 'category': 'Title'}), Document(page_content='oepkgs仓库\nextras\n  oepkgs主仓，仓库中的软件包大部分取自其他仓库，以保证用户只需要添加这一个仓库，便能获取到其他所有仓库中大部分的软件包，受限于软件包多版本现状，无法覆盖所有的软件包，一部分extras仓库中无法获取的软件包，可到其他仓库中查找\ncompatible\n  通过拉取来自其他 Linux 发行版仓库中的源码包，在 openEuler 上重新编译构建，并在 openEuler 上通过了安装测试的软件包，compatible 仓库中 c6，c7，c8，f33，f34，f35，f36，rawhide 分别表示软件源码包来源是 centos6，centos7，centos8，fedora33，fedora34，fedora35，fedora36，fedora-rawhide，不同来源可以用来区分软件包的版本号，来源是 centos8 的软件包一般而言能拿到较高版本的软件包\ncontrib\n  来自许多开发者贡献的软件包，contrib 仓库下以软件包类别划分出仓库名，显然，bigdata 仓库中是大数据场景下需要应用的软件包。\nvalid_release_branch:', metadata={'source': '/root/langchain-ChatGLM/content/oepkgs分支管理.md', 'page_number': 1, 'category': 'Title'}), Document(page_content='valid_release_branch:\nextras 仓库管控\n| 分支名 | 解释 |\n|---|---|\n| master | 用于管控 oepkgs 主仓 openEuler-20.03-LTS-SP1/extras 下软件包的引入 |\n| openEuler-20.03-LTS-SP1 | 用于管控 oepkgs 主仓 openEuler-20.03-LTS-SP1/extras 下软件包的引入 |\n| openEuler-20.03-LTS-SP3 | 用于管控 oepkgs 主仓 openEuler-20.03-LTS-SP3/extras 下软件包的引入 |\n| openEuler-22.03-LTS | 用于管控 oepkgs 主仓 openEuler-22.03-LTS/extras 下软件包的引入 |\ncompatible 仓库管控：\n| 分支名 | 解释 |\n|---|---|\n| compatible_c7_openEuler-20.03-LTS-SP3 | 用于管控 oepkgs compatible 仓 openEuler-20.03-LTS-SP3/compatible/c7 下软件包的引入 |\n| compatible_c8_openEuler-20.03-LTS-SP3 | 用于管控 oepkgs compatible 仓 openEuler-20.03-LTS-SP3/compatible/c8 下软件包的引入 |\n| compatible_f35_openEuler-20.03-LTS-SP3 | 用于管控 oepkgs compatible 仓 openEuler-20.03-LTS-SP3/compatible/f35 下软件包的引入 |\ncontrib 仓库管控：\n| 分支名 | 解释 |\n|---|---|\n| contrib_bigdata_openEuler-20.03-LTS-SP3 | 用于管控 oepkgs contrib 仓 openEuler-20.03-LTS-SP3/contrib/bigdata 下软件包的引入 |\n| contrib_bigdata_openEuler-20.03-LTS-SP1 | 用于管控 oepkgs contrib 仓 openEuler-20.03-LTS-SP1/contrib/bigdata 下软件包的引入 |\n| contrib_virtual_openEuler-20.03-LTS-SP3 | 用于管控 oepkgs contrib 仓 openEuler-20.03-LTS-SP3/contrib/virtual 下软件包的引入 |\n| contrib_basic-system_openEuler-20.03-LTS-SP3 | 用于管控 oepkgs contrib 仓 openEuler-20.03-LTS-SP3/contrib/basic-system 下软件包的引入 | \n contrib multi_version 仓库管控（openstack 有 Queens、Rocky、Wallaby 等多版本的软件包引入）\n| 分支名 | 解释 |\n|---|---|\n| Muti-Version_openstack-wallaby_openEuler-20.03-LTS-SP3 | 用于管控 oepkgs contrib 仓 openEuler-20.03-LTS-SP3/contrib/openstack/wallaby 下软件包的引入 |\n| Muti-Version_openstack-rocky_openEuler-20.03-LTS-SP3 | 用于管控 oepkgs contrib 仓 openEuler-20.03-LTS-SP3/contrib/openstack/rocky 下软件包的引入 |\n| Muti-Version_openstack-queens_openEuler-20.03-LTS-SP3 | 用于管控 oepkgs contrib 仓 openEuler-20.03-LTS-SP3/contrib/openstack/queens 下软件包的引入 |', metadata={'source': '/root/langchain-ChatGLM/content/oepkgs分支管理.md', 'page_number': 1, 'category': 'Title'})]
                if not vs_path:
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    def get_knowledge_based_answer(self,
                                   query,
                                   vs_path,
                                   chat_history=[],
                                   streaming=True):
        self.llm.streaming = streaming
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size=self.chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query,
                                                                            k=self.top_k)


        related_docs = get_docs_with_score(related_docs_with_score)

        prompt = generate_prompt(related_docs, query)

        if streaming:
            for result, history in self.llm._call(prompt=prompt,
                                                  history=chat_history):
                history[-1][0] = query
                response = {"query": query,
                            "result": result,
                            "source_documents": related_docs}
                yield response, history
        else:
            result, history = self.llm._call(prompt=prompt,
                                             history=chat_history)
            history[-1][0] = query
            response = {"query": query,
                        "result": result,
                        "source_documents": related_docs}
            return response, history
