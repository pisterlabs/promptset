import json
from abc import ABC
from copy import deepcopy
from typing import Dict, Union, List

import tiktoken
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from pycomfort.files import *
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase

def flatten_dict(d: Dict[str, Union[str, int, float, Dict, List]], parent_key: str = '', sep: str = '_', list_sep: Optional[str] = ", ") -> Dict[str, Union[str, int, float]]:
    """
    Flattens a nested dictionary, converting list values to comma-separated strings, and ignoring None values.

    :param d: The dictionary to be flattened
    :param parent_key: The key from the parent dictionary, if any
    :param sep: The separator to be used between parent and child keys
    :param list_sep: The separator to be used when joining list items into a string, if None - just drops the lists
    :return: The flattened dictionary
    """

    items: Dict[str, Union[str, int, float]] = {}  # Initialize empty dictionary to hold result

    for k, v in d.items():  # Iterate over items in input dictionary
        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Generate new key

        if isinstance(v, dict):  # If value is a dictionary, flatten it recursively
            items.update(flatten_dict(v, new_key, sep=sep, list_sep=list_sep))

        elif isinstance(v, list):  # If value is a list, convert to string
            if list_sep is not None:
                list_str = "[ " + list_sep.join(map(str, v)) + " ]"
                if list_str is not None:  # Ignore None values
                    items[new_key] = list_str

        elif v is not None:  # Ignore None values
            items[new_key] = v  # If value is not a dictionary or a list, just add to result

    return items  # Return the flattened dictionary


def paginated_paper_to_documents(folder: Path) -> list[Document]:
    metas: list[Path] = [folder] if "_meta.json" in folder.name else traverse(folder, lambda p: "_meta.json" in p.name)
    docs = []
    for meta in metas:
        parent = meta.parent
        json_data = meta.read_text("utf-8")
        meta_dic = json.loads(json_data)
        paper = parent / meta.name.replace("_meta.json", "")
        if not paper.exists():
            print(f"cannot find paper for {meta}")
        else:
            sorted_files = sorted(paper.glob("*.txt"), key=lambda x: int(x.stem.split('_')[-1]))
            for i, text in enumerate(sorted_files):
                doc = paper_to_document(text, meta_dic, extra={"page": i})
                if doc is not None:
                    docs.append(doc)
    return docs



def papers_to_documents(folder: Path, suffix: str = "", include_meta: bool = True):
    txt = traverse(folder, lambda p: "txt" in p.suffix)
    texts = [t for t in txt if suffix in t.name] if suffix != "" else txt
    docs: List[Document] = []
    for t in texts:
        meta: Path = t.parent / t.name.replace(".txt", "_meta.json")
        if include_meta and meta.exists:
            json_data = meta.read_text("utf-8")
            meta_dic = json.loads(json_data)
            doc = paper_to_document(t, meta_dic)
        else:
            doc = paper_to_document(t)
        if doc is not None:
            docs.append(doc)
    return docs


def paper_to_document(paper: Path, meta: Optional[dict] = None, min_tokens: int = 100, extra: Optional[dict] = None) -> Optional[Document]:
    """
    Turns paper into document, assumes the folder/paper_name is DOI
    :param paper:
    :param meta: additional metadata
    :param min_tokens:
    :param extra: optional extra fields to add
    :return: Documenty
    """
    doi = f"http://doi.org/{paper.parent.name}/{paper.stem}"
    text = paper.read_text(encoding="utf-8")
    if len(text) < min_tokens:
        logger.warning("TOO SHORT TEXT")
        return None
    else:
        new_meta = {} if meta is None else flatten_dict(meta)
        new_meta["source"] = doi
        new_meta["doi"] = doi
        if extra is not None:
            for k, v in extra.items():
                new_meta[k] = v
        return Document(
            page_content=text,
            metadata=new_meta
        )


class SourceTextSplitter(RecursiveCharacterTextSplitter, ABC):
    """
    Class that insludes dois and paging into metadata
    """

    @property
    def chunk_size(self):
        return self._chunk_size

    def create_documents(
            self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            meta = _metadatas[i]
            source: Optional[str] = meta["source"] if "source" in meta else None
            #if "paragraph" in meta and meta["paragraph"] is not None:
            #    source = source + "#paragraph_" + str(meta["paragraph"])
            sub_texts = self.split_text(text)

            for j, chunk in enumerate(sub_texts, start=1):
                new_meta = deepcopy(meta)
                if source is not None:
                    num = str(j)
                    if len(sub_texts) > 1:
                        new_meta["source"] = source + "." + num if "#" in source else source + "#" + num
                    if "doi" not in new_meta:
                        new_meta["doi"] = source
                    if "split" not in new_meta:
                        new_meta["split"] = num
                new_doc = Document(page_content=chunk, metadata=new_meta)
                documents.append(new_doc)
        return documents


class HuggingFaceSplitter(SourceTextSplitter, ABC):

    tokenizer: PreTrainedTokenizerBase

    def __init__(self,
                 model: str = "sentence-transformers/all-mpnet-base-v2", #"thenlper/gte-large"
                 tokens: int = 512, #based on benchmarks
                 tokens_overlap: int = 0,
                 keep_separator: bool = False,
                 add_start_index: bool = False
                 ):

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        def length_function(text: str) -> int:
            return len(self.tokenizer.encode(text, padding=True, truncation=True, return_tensors='pt'))

        super().__init__(chunk_size=tokens, chunk_overlap=tokens_overlap,
                         length_function=length_function,
                         keep_separator=keep_separator,
                         add_start_index=add_start_index)



class OpenAISplitter(SourceTextSplitter, ABC):

    def __init__(self, model: str = "gpt-3.5-turbo-16k",
                 tokens: int = 2000,
                 tokens_overlap: int = 0,
                 keep_separator: bool = False,
                 add_start_index: bool = False
                 ):

        def length_function(text: str) -> int:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))

        super().__init__(chunk_size=tokens, chunk_overlap=tokens_overlap,
                   length_function=length_function,
                   keep_separator=keep_separator,
                   add_start_index=add_start_index)
