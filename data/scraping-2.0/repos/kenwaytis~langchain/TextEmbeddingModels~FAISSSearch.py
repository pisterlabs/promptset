import operator
from typing import Any, Callable, Dict, Optional, List, Tuple
from langchain.vectorstores.faiss import dependable_faiss_import
import numpy as np
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores import VectorStore
from langchain.vectorstores.utils import DistanceStrategy
# from config.config import *
import copy

VECTOR_SEARCH_SCORE_THRESHOLD = 2000
CHUNK_SIZE = 250

class FAISSSearch(FAISS,VectorStore):
    def __init__(
            self, 
            embedding_function: Callable[..., Any], 
            index: Any, 
            docstore: Docstore, 
            index_to_docstore_id: Dict[int, str], 
            normalize_L2: bool = False, 
            ):
        super().__init__(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=normalize_L2,
            )
        self.score_threshold = VECTOR_SEARCH_SCORE_THRESHOLD
        self.chunk_size = CHUNK_SIZE
        self.chunk_content = True
    
    def seperate_list(self, ls: List[int]) -> List[List[int]]:
        """
        输入：一个整数列表
        输出：一个列表的列表
        示例：
            input:[1, 2, 3, 5, 6, 8, 9, 10]
            output:[[1, 2, 3], [5, 6], [8, 9, 10]]
        """
        # TODO： 逻辑优化：该函数假定输入列表已排序并重复
        lists = []
        ls1 = [ls[0]]
        for i in range(1, len(ls)):
            if ls[i - 1] + 1 == ls[i]:
                ls1.append(ls[i])
            else:
                lists.append(ls1)
                ls1 = [ls[i]]
        lists.append(ls1)
        return lists


    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs
    ) -> List[Document]:
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        rearrange_id_list = False
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                # This happens when not enough docs are returned.
                continue
            if i in self.index_to_docstore_id:
                _id = self.index_to_docstore_id[i]
            # 执行接下来的操作
            else:
                continue
            doc = self.docstore.search(_id)
            if (not self.chunk_content) or ("context_expand" in doc.metadata and not doc.metadata["context_expand"]):
                # 匹配出的文本如果不需要扩展上下文则执行如下代码
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue

            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                if "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "forward":
                    expand_range = [i + k]
                elif "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "backward":
                    expand_range = [i - k]
                else:
                    expand_range = [i + k, i - k]
                for l in expand_range:
                    if l not in id_set and 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size or doc0.metadata["source"] != \
                                doc.metadata["source"]:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                            rearrange_id_list = True
                if break_flag:
                    break
        if (not self.chunk_content) or (not rearrange_id_list):
            return docs
        if len(id_set) == 0 and self.score_threshold > 0:
            return []
        id_list = sorted(list(id_set))
        id_lists = self.seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    # doc = self.docstore.search(_id)
                    doc = copy.deepcopy(self.docstore.search(_id))
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs.append(doc)
        return docs

