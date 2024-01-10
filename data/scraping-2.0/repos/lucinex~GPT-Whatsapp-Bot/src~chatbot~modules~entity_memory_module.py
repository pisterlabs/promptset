import os
import numpy as np

# import pandas as pd
import faiss

import pickle

# import Levenshtein
from copy import deepcopy

# from pymongo import MongoClient
from collections import defaultdict

import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

# from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback as ocb

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

# from langchain.vectorstores import FAISS
from langchain.docstore.base import AddableMixin, Docstore
from langchain.vectorstores.utils import maximal_marginal_relevance
from langchain.embeddings.base import Embeddings

from llama_index import SimpleDirectoryReader


from pydantic import BaseModel, Extra
from typing import Any, List
from langchain.embeddings.base import Embeddings

from langchain.chains.base import Memory
from pydantic import BaseModel

import spacy


class ModfiedHuggingFaceInstructEmbeddings(BaseModel, Embeddings):
    """Wrapper around sentence_transformers embedding models.
    To use, you should have the ``sentence_transformers``
    and ``InstructorEmbedding`` python package installed.
    Example:
        .. code-block:: python
            from ulangchain.embeddings import HuggingFaceInstructEmbeddings
            model_name = "hkunlp/instructor-large"
            hf = HuggingFaceInstructEmbeddings(model_name=model_name)
    """

    client: Any  #: :meta private:
    model_name: str = "hkunlp/instructor-large"
    """Model name to use."""
    embed_instruction: str = "Represent the document for retrieval:"
    """Instruction to use for embedding documents."""
    query_instruction: str = (
        "Represent the question for retrieving supporting documents: "
    )
    """Instruction to use for embedding query."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            from InstructorEmbedding import INSTRUCTOR

            self.client = INSTRUCTOR(self.model_name)
        except ImportError as e:
            raise ValueError("Dependencies for InstructorEmbedding not found.") from e

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(
        self, texts: List[str], instruction: str = None
    ) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace instruct model.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        instruct = self.embed_instruction if instruction is None else instruction
        instruction_pairs = [[instruct, text] for text in texts]
        embeddings = self.client.encode(instruction_pairs)
        return embeddings.tolist()

    def embed_query(self, text: str, instruction: str = None) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """

        instruct = self.query_instruction if instruction is None else instruction
        instruction_pair = [instruct, text]
        embedding = self.client.encode([instruction_pair])[0]
        return embedding.tolist()


def mkdir_if_dne(path):
    if os.path.isdir(path):
        return True
    else:
        os.mkdir(path)
        print(f"dir created : {path}")
        return False


# class MongoDocstore(Docstore, AddableMixin):
#     """Simple in memory docstore in the form of a dict."""

#     def __init__(self,comp_name:str,client_col):
#         """Initialize with dict."""
#         self.client_col = client_col
#         self.docs = self.client_col[comp_name]
#         pass


#     def add(self, texts: List[Dict]) -> None:
#         """Add texts to in memory dictionary."""

#         ids = [i["id"] for i in texts]
#         # overlapping = set(texts).intersection(self._dict)
#         if self.check_if_ids_exists(ids):
#             raise ValueError(f"Tried to add ids that already exists")

#         self.docs.insert_many(texts)

#     def search(self, search: str) : # dict[]
#         """Search via direct lookup."""
#         # if search not in self._dict:
#         #     return f"ID {search} not found."
#         # else:
#         #     return self._dict[search]
#         out = self.docs.find_one({"id":search})
#         if out == None:
#           return f"ID {search} not found."
#         else:
#           return {i:j for i,j in out.items() if i!='_id'}

#     def delete(self, search):
#       self.docs.delete_one({"id" : search})
#       pass

#     def check_if_ids_exists(self,ids:List[str]): # if exists return true, else false
#         cursor= self.docs.find({"id":{"$in":ids}})
#         found = False
#         for i in cursor:
#             found=True
#             break
#         return found

# from __future__ import annotations


def dependable_faiss_import() -> Any:
    """Import faiss if available, otherwise raise error."""
    try:
        import faiss
    except ImportError:
        raise ValueError(
            "Could not import faiss python package. "
            "Please it install it with `pip install faiss` "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


class InMemoryDocstore_V2(Docstore, AddableMixin):
    """Simple in memory docstore in the form of a dict."""

    def __init__(self, _dict: Dict[str, str]):
        """Initialize with dict."""
        self._dict = _dict

    def add(self, texts) -> None:
        """Add texts to in memory dictionary."""
        if not self._dict == {}:
            overlapping = set(texts).intersection(self._dict)
            if overlapping:
                raise ValueError(f"Tried to add ids that already exist: {overlapping}")
        # print(texts)

        self._dict.update(texts)

    def search(self, search: str):
        """Search via direct lookup."""
        if search not in self._dict:
            return f"ID {search} not found."
        else:
            return self._dict[search]

    def delete(self, id: str):
        if id not in self._dict:
            return f"ID {id} not found."
        else:
            value = self._dict[id]
            del self._dict[id]
            return value


class FAISS_V3:
    """Wrapper around FAISS vector database.
    To use, you should have the ``faiss`` python package installed.
    Example:
        .. code-block:: python
            from langchain import FAISS
            faiss = FAISS(embedding_function, index, docstore)
    """

    def __init__(
        self,
        embedding_model,
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
    ):
        """Initialize with necessary components."""
        self.embedding_model = embedding_model
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id

    def add_texts(
        self, texts: Dict[str, Iterable[str]], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """
        Update: Provide instruction with InstructorEmbedding
                Different for sources and queries.

        Run more texts through the embeddings and add to the vectorstore.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # Embed and create the documents.
        embeddings = []
        all_texts = []
        instruction, all_texts = list(texts.keys())[0], list(texts.values())[0]
        embeddings = self.embedding_model.embed_documents(texts, instruction)

        documents = []
        for i, text in enumerate(all_texts):
            metadata = metadatas[0] if metadatas else {}
            documents.append({"text": text, "metadata": metadata})
        # Add to the index, the index_to_id mapping, and the docstore.
        starting_len = len(self.index_to_docstore_id)
        self.index.add(np.array(embeddings, dtype=np.float32))
        # Get list of index, id, and docs.
        full_info = [
            (starting_len + i, str(uuid.uuid4()), doc)
            for i, doc in enumerate(documents)
        ]
        # Add information to docstore and index.
        self.docstore.add(
            {
                _id: {"text": doc["text"], "metadata": doc["metadata"]}
                for _, _id, doc in full_info
            }
        )  # gto change

        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)
        return [_id for _, _id, _ in full_info]

    def similarity_search_with_score(self, query: Dict[str, str], k: int = 4):
        """Return docs most similar to query.
        Args:
            query: Dict["Instruction":"Quewry"]
            k: Number of Documents to return. Defaults to 4.
        Returns: -> List[Tuple[Document, float]]
            List of Documents most similar to the query and score for each
        """

        embedding = self.self.embedding_model.embed_query(
            list(query.values())[0], list(query.keys())[0]
        )
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)  # to change
            # if not isinstance(doc, Document):
            #     raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, scores[0][j]))
        return docs

    def similarity_search(self, query: Dict[str, str], k: int = 4, **kwargs: Any):
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self, query: Dict[str, str], k: int = 4, fetch_k: int = 20
    ):
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_model.embed_query(
            list(query.values())[0], list(query.keys())[0]
        )
        _, indices = self.index.search(np.array([embedding], dtype=np.float32), fetch_k)
        # -1 happens when not enough docs are returned.
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(embedding, embeddings, k=k)
        selected_indices = [indices[0][i] for i in mmr_selected]
        docs = []
        # print(selected_indices,self.index_to_docstore_id )
        for i in selected_indices:
            if i != -1:
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                # if not isinstance(doc, Document):
                #     raise ValueError(f"Could not find document for id {_id}, got {doc}")
                docs.append(doc)
        return docs

    def save_local(self, folder_path: str) -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.
        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(self.index, str(path / "index.faiss"))

        # save docstore and index_to_docstore_id
        with open(path / "index.pkl", "wb") as f:
            pickle.dump(self.index_to_docstore_id, f)

        with open(path / "docstore.pkl", "wb") as f:
            pickle.dump(self.docstore._dict, f)

    @classmethod
    def load_local(cls, folder_path: str, embeddings: Embeddings, docstore: Docstore):
        """Load FAISS index, docstore, and index_to_docstore_id to disk.
        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
        """
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(str(path / "index.faiss"))

        # load docstore and index_to_docstore_id
        with open(path / "index.pkl", "rb") as f:
            index_to_docstore_id = pickle.load(f)

        with open(path / "docstore.pkl", "rb") as f:
            doc_dict = pickle.load(f)
        docstore = InMemoryDocstore_V2(doc_dict)

        return cls(embeddings, index, docstore, index_to_docstore_id)


class Entity:
    def __init__(self, entity_name: str, entity_text: str):
        self.entity = entity_name
        self.text = entity_text

    def __add__(self, other):
        if self.entity == other.entity:
            self.text = self.text + other.text
        else:
            print("Cannot add two entity text for different sources")
            return

    def __repr__(self):
        return {"entity_name": self.entity, "text": self.text}


class Entities:
    def __init__(self, entities: Iterable[Entity]):
        self.entities = {}
        self.categorise(entities)

    def categorise(self, entities: Iterable[Entity]):
        if self.entities == {}:
            self.entities = defaultdict(lambda: False)

        for entity in entities:
            if self.entities[entity.entity] == False:
                self.entities[entity.entity] = entity.text
            else:
                self.entities[entity.entity] += f"\n{entity.text}"

    def __getitem__(self, index):
        return self.entities[index]

    def keys(self):
        return self.entities.keys()

    def values(self):
        return self.entities.values()

    def items(self):
        return self.entities.items()


class MemoryModule:
    doc_map_suffix = "entity_mapping.pkl"
    module_name = "InstructSpcayMemory"

    def __init__(self, save_dir):
        self.save_dir = save_dir + "/" + self.module_name
        # memory doc v1
        # self.index = self._get_index()

        self.embed_model = ModfiedHuggingFaceInstructEmbeddings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len,
        )
        self.docstore = InMemoryDocstore_V2({})
        self.index = faiss.IndexFlatL2(768)
        self.index_to_docstore_id = {}
        self.entity_mapping = {}
        self.vecstore = FAISS_V3(
            self.embed_model, self.index, self.docstore, self.index_to_docstore_id
        )

        save_path_bool = mkdir_if_dne(self.save_dir)
        index_path_bool = os.path.isfile(self.save_dir + "/index.faiss")
        print(index_path_bool, save_path_bool)
        if index_path_bool:
            print("Loading previous vmemory store")
            self._reload()

        else:
            print(f"Initializing Memory_module with save to {self.save_dir}")
            print(index_path_bool, save_path_bool)
            # self._initialize()
            # self._update()

    def update(self, entity_name: str, text: str = ""):
        if not text == "":
            if entity_name in self._get_all_indexed_entities():
                self._remove_entity(entity_name)

                self._add_entity(entity_name, text)

        return

    def _remove_entity(self, entity_name):  # delete entity.  Make sure it exists in db
        if entity_name not in self._get_all_indexed_entities():
            print(
                f"Entity: {entity_name}, does not exists inside index ! WIll not do anything"
            )
            return None
        print(f"Removing entity {entity_name}")
        assert len(self.entity_mapping) > 0, "No entities to remove"

        # filter function
        def my_filtering_function(pair):
            key, value = pair
            if value["entity"] == entity_name:
                return True  # filter pair out of the dictionary
            else:
                return False  # keep pair in the filtered dictionary

        file_dict = dict(filter(my_filtering_function, self.entity_mapping.items()))
        doc_mapping = list(file_dict.values())[0]["mapping"]
        doc_id = list(file_dict.keys())[0]
        str_ids, ids = list(doc_mapping.keys()), list(doc_mapping.values())
        # ids need to be shifted
        new_str_to_ids = {
            j: i
            for i, j in self.vecstore.index_to_docstore_id.items()
            if j not in str_ids
        }
        # remove from index
        action = self.vecstore.index.remove_ids(np.array(ids))
        print(action)
        if action is not None:
            for i, j in zip(str_ids, ids):
                # print(i)
                jk = self.vecstore.docstore.delete(i)  # docstore value removal
        else:
            print(f"Didnt delete {entity_name}")
            # action not sucessful
            return

        all_keys = list(self.entity_mapping.keys())
        # print("all keys before delete ", all_keys)
        del self.entity_mapping[doc_id]

        self.vecstore.index_to_docstore_id = {
            e: k[0] for e, k in enumerate(new_str_to_ids.items())
        }
        new_str_to_ids = {j: i for i, j in self.vecstore.index_to_docstore_id.items()}

        # update document mapping
        all_keys = list(self.entity_mapping.keys())

        for e, i in enumerate(all_keys):
            if e != i:
                old_mapping = list(self.entity_mapping[i]["mapping"].keys())
                entity_name = list(self.entity_mapping[i]["entity_name"])
                new_map = {}
                for map in old_mapping:
                    prop_index = new_str_to_ids[map]
                    new_map[map] = prop_index
                self.entity_mapping[e] = {
                    "entity_name": "".join(entity_name),
                    "mapping": new_map,
                }
                del self.entity_mapping[i]
        cur_doc_list_len = (
            0 if self.entity_mapping == None else len(self.entity_mapping.keys())
        )
        self._save_all()
        return

    def _add_entity(self, entity_name, text):
        if entity_name not in self._get_all_indexed_entities():
            if text != None and type(text) == str:
                new_index_to_docstr = self._run_texts(text, [{"entity": entity_name}])

                cur_doc_list_len = (
                    0
                    if self.entity_mapping == None
                    else len(self.entity_mapping.keys())
                )
                print("No. of docs: ", cur_doc_list_len + 1)
                mapping = {
                    cur_doc_list_len: {
                        "entity": entity_name,
                        "mapping": new_index_to_docstr,
                    }
                }
                if type(self.entity_mapping) == dict:
                    self.entity_mapping.update(mapping)
                else:
                    self.entity_mapping = mapping

    def add_entities(self, entity: Entities):
        all_entities = set(entity.keys())
        intersection = all_entities.intersection(self._get_all_indexed_entities())
        # ent_to_index = {j['entity']:i for i,j in self.entity_mapping.items()}
        for ent in intersection:
            print(f"Updating old entity {ent}")
            text = entity[ent]

            def my_filtering_function(pair):
                key, value = pair

                if value["entity"] == ent:
                    return True  # filter pair out of the dictionary
                else:
                    return False  # keep pair in the filtered dictionary

            file_dict = dict(filter(my_filtering_function, self.entity_mapping.items()))
            entity_mapping = list(file_dict.values())[0]["mapping"]
            str_ids, ids = list(entity_mapping.keys()), list(entity_mapping.values())

            old_text = ""
            for i in str_ids:
                node = self.vecstore.docstore.search(i)
                old_text += node["text"]
            new_text = old_text + "\n" + text
            self.update(ent, new_text)

        new_entities = [i for i in all_entities if i not in intersection]
        for i in new_entities:
            entity_name = i
            text = entity[i]
            self._add_entity(entity_name, text)

        self._save_all()
        # print(mapping)
        return

    def _run_texts(self, text, meta):
        # return new ids inserted
        texts = self.splitter.split_text(text)
        instruction = f"Represent information about Entity: {meta[0]['entity']} to be retirved later."
        new_doc_indexes = self.vecstore.add_texts({instruction: texts}, meta)
        new_doc_indexes = {
            j: i
            for i, j in self.vecstore.index_to_docstore_id.items()
            if j in new_doc_indexes
        }
        return new_doc_indexes

    def _get_all_indexed_entities(self):
        if self.entity_mapping != {}:
            entity_names = [i["entity"] for i in self.entity_mapping.values()]
            return entity_names
        else:
            return []

    def _save_all(self):
        self.vecstore.save_local(self.save_dir)

        doc_map_path = self.save_dir + "/" + self.doc_map_suffix
        with open(doc_map_path, "wb") as b:
            pickle.dump(self.entity_mapping, b)

    def _reload(self):
        doc_map_path = self.save_dir + "/" + self.doc_map_suffix
        with open(doc_map_path, "rb") as f:
            self.entity_mapping = pickle.load(f)
        self.vecstore = FAISS_V3.load_local(
            self.save_dir, self.embed_model, self.docstore
        )

    def search(self, query, k: int = 4, fetch_k: int = 20) -> Dict[str, str]:
        len_index = len(self.vecstore.index_to_docstore_id)
        if len_index == 0:
            return {}
        n_k = len_index if len_index > k else k
        total_k = len_index if len_index > fetch_k else fetch_k
        res = self.vecstore.max_marginal_relevance_search(query, k=n_k, fetch_k=total_k)
        result = []

        for i in res:
            entity = Entity(entity_name=i["metadata"]["entity"], entity_text=i["text"])
            result.append(entity)
        entities = Entities(result)
        return entities.entities

    def get_entity_id(self, id: int):
        if self.entity_mapping != {}:
            for i, j in self.entity_mapping.items():
                indexes = list(j["mapping"].values())
                if id in indexes:
                    return i, j["entity"]


class InstructSpacyMemoryModule(Memory, BaseModel):
    """Memory class for storing information about entities."""

    # Define dictionary to store information about entities.

    # Define key to pass information about entities into prompt.
    memory_key: str = "entities"

    entities = {}
    fetch_k = 4

    # def init_db(self,mdb_client,path="./agent_data"):

    #     # self.nlp = spacy.load('en_core_web_lg')
    #     # self.memory = MemoryModule(path,mdb_client)
    #     # self.entities = self.memory._get_all_indexed_entities()

    #     return self

    def __init__(self, path):
        Memory._init__(self)
        self.memory_path = "/InstructEntityMemory"
        BaseModel.__init__(self)
        self.memory = MemoryModule(self.memory_path)
        self.nlp = spacy.load("en_core_web_lg")
        self.entities = self.memory._get_all_indexed_entities()

    def clear(self):
        for i in self.memory._get_all_indexed_entities():
            self.memory._remove_entity(i)
        self.memory._save_all()

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load the memory variables, in this case the entity key."""
        # Get the input text and run through spacy
        input_text = inputs[list(inputs.keys())[0]]
        doc = self.nlp(input_text)
        doc_entities = doc
        # Extract known information about entities, if they exist.
        text = self.get_entities_info(doc.ents, input_text)
        # Return combined information about entities to put into context.
        return {self.memory_key: text}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        # Get the input text and run through spacy
        text = f"Query: {inputs[list(inputs.keys())[0]]}, Response: {outputs[list(inputs.keys())[0]]}"
        # print(outputs)
        doc = self.nlp(text)

        entities = []
        # For each entity that was mentioned, save this information to the dictionary.
        for ent in doc.ents:
            ent_str = str(ent)

            entities.append(Entity(ent_str, text))
        entities = Entities(entities)

        self.memory.add_entities(entities)
        return

    def get_entities_info(self, entities, input_text):
        result = {}
        stored_entities = self.memory._get_all_indexed_entities()
        entities = [str(ent) for ent in entities if str(ent) in stored_entities]

        instruction = (
            f"Represent the Query to retrieve information about {', '.join(entities)}:"
        )
        out = self.memory.search({instruction: input_text}, self.fetch_k)

        if not out == {}:
            text = "\n".join([j for i, j in out.items()])
            return text
        else:
            return "No memory"
