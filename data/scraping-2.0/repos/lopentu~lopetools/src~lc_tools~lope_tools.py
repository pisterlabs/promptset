# https://python.langchain.com/en/latest/modules/agents/tools/custom_tools.html#multi-argument-tools
# https://cwngraph.readthedocs.io/en/latest/
import dataclasses
import json
from pathlib import Path
import os
import pickle
import re

from dotenv import load_dotenv
import httpx
from langchain.tools import BaseTool
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from loguru import logger
import opencc
import nh3
from sentence_transformers import SentenceTransformer, util
import torch
import walrus
import weaviate

from CwnGraph import CwnImage
from .mytypes import ContentItem

BASE = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE.joinpath(".env"))
# sys.path.append(str(BASE / "src"))

from api.tagger.schemas import TagOutput  # noqa: E402

cwn = CwnImage.latest()
with BASE.joinpath("data/senseid_to_metadata.json").open("r") as f:
    asbc_freq = json.load(f)

API_URL = "http://localhost:3001"
t2s = opencc.OpenCC("t2s.json")
s2t = opencc.OpenCC("s2t.json")
BASE = Path(__file__).resolve().parent.parent.parent
embedder = SentenceTransformer("distiluse-base-multilingual-cased-v1")
p = str(BASE / "other_data/cwn_definition_embeddings.pkl")
with open(p, "rb") as f:
    data = pickle.load(f)
    def_corpus = data["corpus"]
    def_embeddings = data["embeddings"]

db = walrus.Database(host="localhost", port=6379, db=0)
asbc_index = db.Index("asbc")

WV_CLIENT = weaviate.Client(
    url="http://localhost:8000",
    auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_ADMIN_PASS"]),
    timeout_config=(5, 30),  # (connect timeout, read timeout) # type: ignore
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
)
IGNORE_ATTRS = [
    "media",
    "post_id",
]
PTT_ATTRS = [field.name for field in dataclasses.fields(ContentItem)]
PTT_RETRIEVER = WeaviateHybridSearchRetriever(
    client=WV_CLIENT,
    k=10,
    alpha=0.5,  # weighting for each search algorithm (alpha = 0 (sparse, BM25), alpha = 1 (dense), alpha = 0.5 (equal weight for sparse and dense))
    index_name="ContentItem",
    text_key="text",
    attributes=PTT_ATTRS,  # include these attributes in the 'metadata' field of the search results
)


class ToolMixin:
    def json_dumps(self, d: dict | list) -> str:
        # return json.dumps(nh3.clean(d), default=str, ensure_ascii=False)
        return json.dumps(d, default=str, ensure_ascii=False)


class SenseTagTool(BaseTool, ToolMixin):
    name = "SenseTagger"
    description = "輸入文章可以斷詞和標註詞性SenseID和詞義，輸出為JSON格式。"

    def format_text(self, token_dict: dict) -> str | dict:
        token = token_dict["token"]
        tag = token_dict["tag"]
        gloss = token_dict["gloss"]

        gloss = re.sub(r"\(\d\.\d{4}\)", "", gloss)  # remove probability at the end
        m = re.search(r"\[(?P<sense_id>\d{8})\] (?P<gloss>.+。)", gloss)
        if not m:
            sense_id = "NONE"
            gloss = "NONE"
        else:
            sense_id = m.group("sense_id")
            gloss = m.group("gloss")
            # gloss = f"[SenseID] {m.group('sense_id')} || [詞意] {m.group('gloss')}"

        return {
            "詞": token,
            "詞性": tag,
            "SenseID": sense_id,
            "詞意": gloss,
        }
        # return f"([詞] {token} || [詞性] {tag} || {gloss})"

    def _run(self, text: str) -> str:
        with httpx.Client() as client:
            out = []
            response: TagOutput = client.get(
                f"{API_URL}/tag/", params={"text": text}, timeout=600.0
            ).json()
            # response: TagOutput = client.post(
            #     f"{API_URL}/tag/", data={"text": text}, timeout=600.0
            # ).json()
            response = response["tagged_text"]  # type: ignore
            logger.info(response)
            for sent in response:
                tmp = []
                for tok in sent:
                    tmp.append(
                        # type: ignore
                        self.format_text(tok)
                        # f"([詞] {tok['token']} || [詞性] {tok['tag']} || [詞意] {tok['gloss']})"
                    )
                # out.append("\n".join(tmp))
                out.append(tmp)

            # out = "。".join(out)
            return self.json_dumps(out)

    async def _arun(self, text: str) -> str:
        async with httpx.AsyncClient() as client:
            out = []
            response: TagOutput = (
                await client.get(
                    f"{API_URL}/tag/", params={"text": text}, timeout=600.0
                )
            ).json()
            response = response["tagged_text"]  # type: ignore
            logger.info(response)
            for sent in response:
                tmp = []
                for tok in sent:
                    tmp.append(
                        # type: ignore
                        self.format_text(tok)
                        # f"([詞] {tok['token']} || [詞性] {tok['tag']} || [詞意] {tok['gloss']})"
                    )
                out.append(tmp)

            # out = "。".join(out)
            return self.json_dumps(out)


class QuerySenseBaseTool(BaseTool, ToolMixin):
    def _base_run(self, text: str, search_method: str) -> str:
        res = {}
        senses = cwn.find_senses(**{search_method: f"^{text}$"})
        for sense in senses:
            res[sense.id] = self.expand_sense(sense)

        res = self.json_dumps(res)
        # logger.info(res)
        return res

    async def _base_arun(self, text, search_method: str) -> str:
        res = {}
        senses = cwn.find_senses(**{search_method: text})
        for sense in senses:
            res[sense.id] = self.expand_sense(sense)

        res = self.json_dumps(res)
        return res

    @staticmethod
    def expand_sense(sense):
        res = {}
        # keep = ["definition", "all_examples", "facets", "head_word", "id", "pos"]
        keep = ["definition", "pos", "all_examples"]  # trying to reduce token usage
        attrs = [
            d
            for d in dir(sense)
            if not d.startswith("__") and not d.startswith("_") and (d in keep)
        ]
        for attr in attrs:
            retrieved = getattr(sense, attr)
            if not retrieved:
                continue
            if callable(retrieved):
                retrieved = retrieved()
            if attr == "all_examples":
                retrieved = retrieved[:1]  # too many examples goes over context limit
            res[attr] = retrieved

        return res


class QuerySenseFromLemmaTool(QuerySenseBaseTool):
    name = "QuerySensesFromLemma"
    description = "搜尋CWN所有詞義，找到lemma符合目標詞的數個詞義，可以使用Regular Expression。輸出為JSON格式。"

    def _run(self, text: str) -> str:
        res = self._base_run(text, "lemma")
        return res

    async def _arun(self, text: str) -> str:
        res = await self._base_arun(text, "lemma")
        return res


class QuerySenseFromDefinitionTool(QuerySenseBaseTool):
    name = "QuerySensesFromDefinition"
    description = "搜尋CWN所有詞義，找到definition有出現目標詞的數個詞義，可以使用Regular Expression。輸出為JSON格式。"

    def _run(self, text: str) -> str:
        res = self._base_run(text, "definition")
        return res

    async def _arun(self, text: str) -> str:
        res = await self._base_arun(text, "definition")
        return res


class QuerySenseFromExamplesTool(QuerySenseBaseTool):
    name = "QuerySensesFromExample"
    description = "搜尋CWN所有詞義，找到例句有出現目標詞的數個詞義，可以使用Regular Expression。輸出為JSON格式。"

    def _run(self, text: str) -> str:
        res = json.loads(self._base_run(text, "examples"))
        for k, v in res.items():
            all_examples = v["all_examples"]
            # 原本一個sense的all_examples(list of strings)改成只收錄符合的例句們
            new_examples = []
            for example in all_examples:
                if re.search(text, example):
                    new_examples.append(example)
            v["all_examples"] = new_examples
            res[k] = v

        return self.json_dumps(res)

    async def _arun(self, text: str) -> str:
        res = json.loads(await self._base_arun(text, "examples"))
        for k, v in res.items():
            all_examples = v["all_examples"]
            # 原本一個sense的all_examples(list of strings)改成只收錄符合的例句們
            new_examples = []
            for example in all_examples:
                if re.search(text, example):
                    new_examples.append(example)
            v["all_examples"] = new_examples
            res[k] = v

        return self.json_dumps(res)


class QueryRelationsFromSenseIdTool(BaseTool, ToolMixin):
    name = "QueryRelationsFromSenseId"
    description = "輸入目標詞的SenseID（8位數字） ，得到目標詞的relations，取得特定的語意關係（synonym同義詞、antonym反義詞、hypernym上位詞、hyponym下位詞）。如果已經有標記過的文章，則使用文章中目標詞的SenseID，再去獲得該SenseID的relations。輸出為JSON格式。"
    ignore = ["has_facet", "is_synset", "generic", "nearsynonym"]

    def _run(self, sense_id: str) -> str:
        relations = cwn.from_sense_id(sense_id).relations
        relations = [r for r in relations if r[0] not in self.ignore]
        return self.json_dumps(relations)

    async def _arun(self, sense_id: str) -> str:
        relations = cwn.from_sense_id(sense_id).relations
        relations = [r for r in relations if r[0] not in self.ignore]
        return self.json_dumps(relations)


class QueryAsbcSenseFrequencyTool(BaseTool, ToolMixin):
    name = "QueryAsbcSenseFrequency"
    description = "輸入目標詞義的Sense ID（8個數字），得到目標詞義在中研院平衡語料庫（ASBC）的詞義頻率。"

    def _run(self, sense_id: str) -> str:
        if sense_id not in asbc_freq:
            return self.json_dumps({"sense_info": "查無此詞義。"})
        return self.json_dumps({"sense_info": asbc_freq[sense_id]})

    async def _arun(self, sense_id: str) -> str:
        if sense_id not in asbc_freq:
            return self.json_dumps({"sense_info": "查無此詞義。"})
        return self.json_dumps({"sense_info": asbc_freq[sense_id]})


class QuerySimilarSenseFromCwnTool(BaseTool, ToolMixin):
    name = "QuerySimilarSenseFromCwn"
    description = "輸入釋義，得到在中研院平衡語料庫（ASBC）與目釋義相似的詞義。輸出為JSON格式。"

    def _run(self, query: str) -> str:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, def_embeddings)[0]
        top_results = torch.topk(cos_scores, k=10)
        res = []
        for score, idx in zip(top_results.values, top_results.indices):
            res.append({**def_corpus[idx], "score": float(score)})
        return self.json_dumps(res)

    async def _arun(self, query: str) -> str:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, def_embeddings)[0]
        top_results = torch.topk(cos_scores, k=10)
        res = []
        for score, idx in zip(top_results.values, top_results.indices):
            res.append({**def_corpus[idx], **{"score": float(score)}})
        return self.json_dumps(res)


class QueryPTTSearchTool(BaseTool, ToolMixin):
    name = "PTTSearch"
    description = "輸入目標字串，得到PTT上的文章標題、內文、推文。"
    return_direct = True

    def _run(self, query: str) -> str:
        res = PTT_RETRIEVER.get_relevant_documents(query)
        return self.json_dumps(res)

    def _arun(self, query: str) -> str:
        res = PTT_RETRIEVER.get_relevant_documents(query)
        return self.json_dumps([r.json() for r in res])


class QueryAsbcFullTextTool(BaseTool, ToolMixin):
    name = "QueryAsbcFullText"
    description = (
        "輸入搜尋字串，得到目標字串在中研院平衡語料庫（ASBC）的前後文。"
        "搜尋可以包含可以包含集合操作（例如 AND、OR），布林運算，並使用括號來指示運算優先順序"
    )
    return_direct = True
    top_k = 50
    clean_metadata = True

    def _clean_metadata(self, res: list[str]) -> list[str]:
        cleaned = []
        for line in res:
            tmp = []
            chars = line.split()
            for char in chars:
                tmp.append(char.split("-")[0])
            cleaned.append("".join(tmp))
        # cleaned = "\n".join(cleaned)
        return cleaned

    def _run(self, query: str) -> str:
        res = asbc_index.search(query)[: self.top_k]
        res = [r["content"] for r in res]
        if self.clean_metadata:
            res = self._clean_metadata(res)

        return self.json_dumps(res)

    async def _arun(self, query: str) -> str:
        res = asbc_index.search(query)[: self.top_k]
        res = [r["content"] for r in res]
        if self.clean_metadata:
            res = self._clean_metadata(res)
        return self.json_dumps(res)
