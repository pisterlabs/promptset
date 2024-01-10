from typing import List

from langchain import FAISS

from tools.text import SimpleTextFilter, EntityTextFragmentFilter, fragment_text, \
    EntityVectorStoreFragmentFilter, SentenceSplitter, LineSplitter
from tools.utils import load_txt


def docs_to_lst(docs):
    lst = []
    for e in docs:
        lst.append(e.page_content)
    return lst


def mem_to_lst(mem):
    lst = []
    for i in range(len(mem)):
        lst.append(mem[i].page_content)
    return lst


def get_related_from_vector_store(query, vs):
    related_text_with_score = vs.similarity_search_with_score(query)
    docs = []
    for doc, score in related_text_with_score:
        docs.append(doc.page_content)
    return docs


def arrange_entities(entity_lst):
    entity_dicts = {}
    for entity in entity_lst:
        # 可能有中文冒号，统一替换为英文冒号
        entity = entity.replace('：', ':')
        k, v = entity.split(":", 1)
        entity_dicts[k] = v
    return entity_dicts


def arrange_entities_name(entity_lst):
    entity_names = []
    for entity in entity_lst:
        # 可能有中文冒号，统一替换为英文冒号
        entity = entity[0].replace('：', ':')
        k, v = entity[0].split(":", 1)
        entity_names.append(k)
    return entity_names


def init_vector_store(embeddings,
                      filepath: str or List[str],
                      textsplitter):
    # docs = load_docs(filepath, textsplitter)
    txt = load_txt(filepath, textsplitter)
    if len(txt) > 0:
        # vector_store = FAISS.from_documents(docs, embeddings)
        vector_store = FAISS.from_texts(txt, embeddings)
        return vector_store
    else:
        return None


class VectorStore:

    def __init__(self, embeddings, path, textsplitter, chunk_size=20, top_k=6):
        self.top_k = top_k
        self.path = path
        self.core = init_vector_store(embeddings=embeddings, filepath=self.path, textsplitter=textsplitter)
        self.chunk_size = chunk_size

    def similarity_search_with_score(self, query):
        if self.core is not None:
            self.core.chunk_size = self.chunk_size
            return self.core.similarity_search_with_score(query, self.top_k)
        else:
            return []

    def get_path(self):
        return self.path


class SimpleStoreTool:

    def __init__(self, info, entity_top_k, history_top_k, event_top_k):
        self.info = info
        self.ai_name = info.ai_name
        self.tsp = SentenceSplitter()
        self.etfs = EntityTextFragmentFilter(tsp=self.tsp, top_k=history_top_k, entity_weight=0.8)

        self.entity_textsplitter = LineSplitter()
        self.history_textsplitter = LineSplitter()
        self.event_textsplitter = LineSplitter()

        self.entity_text_filter = SimpleTextFilter(entity_top_k)
        self.history_text_filter = SimpleTextFilter(history_top_k)
        self.event_text_filter = SimpleTextFilter(event_top_k)

    def load_entity_store(self):
        return load_txt(self.info.entity_path, self.entity_textsplitter)

    def load_history_store(self):
        return load_txt(self.info.history_path, self.history_textsplitter)

    def load_event_store(self):
        return load_txt(self.info.event_path, self.event_textsplitter)

    def get_entity_mem(self, query, store):
        return self.entity_text_filter.filter(query, store)

    def get_history_mem(self, query, store):
        return self.history_text_filter.filter(query, store)

    def get_event_mem(self, query, store):
        return self.event_text_filter.filter(query, store)

    def entity_fragment(self, query, entity_mem):
        entity_dict = arrange_entities(entity_mem)
        return self.etfs.filter(query, entity_dict)

    def dialog_fragment(self, query, dialog_mem):
        dialog_mem = fragment_text(dialog_mem, self.tsp)
        # 再次过滤
        dialog_mem = self.history_text_filter.filter(query, dialog_mem)
        for i, dialog in enumerate(dialog_mem):
            dialog_mem[i] = self.ai_name + '说：' + dialog
        return dialog_mem

    def answer_extract(self, mem, has_ai_name):
        # 提取对话，仅有ai的回答
        splitter = self.ai_name + '说：'
        for i, dialog in enumerate(mem):
            parts = dialog.split(splitter)
            mem[i] = splitter + parts[-1] if has_ai_name else parts[-1]


class VectorStoreTool:
    def __init__(self, info, embeddings, entity_top_k, history_top_k, event_top_k):
        self.info = info
        self.embeddings = embeddings
        self.ai_name = info.ai_name
        self.entity_top_k = entity_top_k
        self.history_top_k = history_top_k
        self.event_top_k = event_top_k

        self.ssp = SentenceSplitter()
        self.etfs = EntityVectorStoreFragmentFilter(tsp=self.ssp, top_k=entity_top_k, entity_weight=0.8)

        self.entity_textsplitter = LineSplitter()
        self.history_textsplitter = LineSplitter()
        self.event_textsplitter = LineSplitter()

    def load_entity_store(self):
        return VectorStore(self.embeddings,
                           self.info.entity_path,
                           top_k=self.entity_top_k,
                           textsplitter=self.entity_textsplitter)

    def load_history_store(self):
        return VectorStore(self.embeddings,
                           self.info.history_path,
                           top_k=self.history_top_k,
                           textsplitter=self.history_textsplitter)

    def load_event_store(self):
        return VectorStore(self.embeddings,
                           self.info.event_path,
                           top_k=self.event_top_k,
                           textsplitter=self.event_textsplitter)

    @staticmethod
    def get_entity_mem(query, store):
        return get_related_from_vector_store(query, store)

    @staticmethod
    def get_history_mem(query, store):
        return get_related_from_vector_store(query, store)

    @staticmethod
    def get_event_mem(query, store):
        return get_related_from_vector_store(query, store)

    def entity_fragment(self, query, entity_mem):
        entity_dict = arrange_entities(entity_mem)
        return self.etfs.filter(query, entity_dict, self.embeddings)

    def dialog_fragment(self, query, dialog_mem):
        dialog_mem = fragment_text(dialog_mem, self.ssp)
        # 再次过滤
        try:
            vs = FAISS.from_texts(dialog_mem, self.embeddings)
            dialog_with_score = vs.similarity_search_with_score(query, self.history_top_k)
        except IndexError:
            return []
        res_lst = []
        for doc in dialog_with_score:
            res_lst.append(self.ai_name + '说：' + doc[0].page_content)
        return res_lst

    def answer_extract(self, mem, has_ai_name):
        # 提取对话，仅有ai的回答
        splitter = self.ai_name + '说：'
        for i, dialog in enumerate(mem):
            parts = dialog.split(splitter)
            mem[i] = splitter + parts[-1] if has_ai_name else parts[-1]
