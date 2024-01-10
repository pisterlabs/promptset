import dataclasses
import json
import os
from langchain.docstore.document import Document
from typing import Callable

from analyzers import extraction


def _generate_path_components(hash: str, component_length: int = 12) -> list[str]:
    components = []
    low_i = -2
    high_i = 0
    modifier = 2
    for _ in range(max(component_length, 1)):
        low_i += modifier
        high_i += modifier
        components.append(hash[low_i:high_i])
    components.append(hash[high_i:])
    return components


@dataclasses.dataclass
class DiskStore:
    root_directory: str
    logging_func: Callable[[str], None] = print

    def has_document_content(self, hash: str) -> bool:
        content_path = self._hash_path(hash)
        return os.path.exists(content_path)

    def save_document_content(self, hash: str, documents: list[Document]):
        self.logging_func(f"Saving {hash}")
        content_path = self._hash_path(hash)

        os.makedirs(os.path.dirname(content_path), exist_ok=True)

        with open(content_path, "w+") as f:
            self.logging_func(f"Saving document {os.path.abspath(content_path)}")
            document_dict = {}
            for i, doc in enumerate(documents):
                document_dict[i] = {"c": doc.page_content}
                if doc.metadata:
                    document_dict[i]["m"] = doc.metadata

            json.dump(document_dict, f)

    def restore_document_content(self, hash: str) -> list[Document]:
        self.logging_func(f"Loading document for {hash}")
        content_path = self._hash_path(hash)

        with open(content_path, "r") as f:
            doc_content = json.load(f)

        sorted_doc_content = sorted(doc_content.items())
        return [
            Document(
                page_content=doc["c"],
                metadata=doc.get("m", {}),
            )
            for _, doc in sorted_doc_content
        ]

    def delete_document_content(self, hash: str):
        content_path = self._hash_path(hash)
        os.remove(content_path)

        os.removedirs(os.path.dirname(content_path))

    def has_entity_relations(self, hash: str) -> bool:
        content_path = self._hash_path(hash)
        entity_relations_path = f"{content_path}.entity_relations"
        return os.path.exists(entity_relations_path)

    def load_entity_relations(self, hash: str) -> list[extraction.EntityRelationSchema]:
        content_path = self._hash_path(hash)
        entity_relations_path = f"{content_path}.entity_relations"
        if not os.path.exists(entity_relations_path):
            return []
        with open(entity_relations_path, "r") as f:
            return [extraction.EntityRelationSchema(**x) for x in list(json.load(f))]

    def save_entity_relations(
        self,
        hash: str,
        entity_relations: list[extraction.EntityRelationSchema],
    ):
        content_path = self._hash_path(hash)
        entity_relations_path = f"{content_path}.entity_relations"
        if os.path.exists(entity_relations_path):
            return

        print(f"saving {len(entity_relations)} relations")
        with open(entity_relations_path, "w+") as f:
            json.dump([x.model_dump() for x in entity_relations], f)

    def _hash_path(self, hash: str) -> str:
        return os.path.join(self.root_directory, *_generate_path_components(hash))


def default_store(**kwargs):
    root = os.environ.get("HOME", ".")
    return DiskStore(
        root_directory=kwargs.pop("root_directory", f"{root}/kms/docs"), **kwargs
    )


if __name__ == "__main__":
    # correct length of initial hash shcema
    hash = "1234567890123456" * 4
    docs = [
        Document(page_content="1234", metadata={}),
        Document(page_content="5678", metadata={"test": "1234"}),
    ]
    store = DiskStore(root_directory="root_path")
    store.save_document_content(
        hash,
        docs,
    )

    try:
        restored_docs = store.restore_document_content(hash)
        for i, doc in enumerate(docs):
            print(doc)
            assert docs[i] == doc
    finally:
        store.delete_document_content(hash)
