import json

import weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.schema import Document


WEAVIATE_URL = "http://localhost:8080"
client = weaviate.Client(
    url=WEAVIATE_URL,
)
class WeaviateHybridSearchTransformersRetriever(WeaviateHybridSearchRetriever):
    def _create_schema_if_missing(self) -> None:
        class_obj = {
            "class": self._index_name,
            "properties": [{"name": self._text_key, "dataType": ["text"]}],
            "vectorizer": "text2vec-transformers",
        }

        if not self._client.schema.exists(self._index_name):
            self._client.schema.create_class(class_obj)


def remove_quotes_from_string(input_string):
    # Remove single quotes (')
    cleaned_string = input_string.replace("'", "")

    # Remove double quotes (")
    cleaned_string = cleaned_string.replace('"', "")

    return cleaned_string

if __name__ == "__main__":
    with open("./data/readwise_database_KnowledgeAgent.json") as g:
        all_highlights = json.load(g)

    with open("./all_generations.json") as g:
        all_generations = json.load(g)


    class_name = "P2842eba01fcfb2f997160fc4e1af4898"
    class_properties = ["content", "cfiRange", "chapterIndex", "paragraphIndex"]

    # retriever = WeaviateHybridSearchTransformersRetriever(
    #     client, class_name, text_key="topic"
    # )

    retriever = WeaviateHybridSearchTransformersRetriever(
        client=client, index_name="P2842eba01fcfb2f997160fc4e1af4898", text_key="content",
        attributes=["paragraphIndex", "chapterIndex", "cfiRange"], create_schema_if_missing=True
    )
    import hashlib

    all_relationships = []
    nodes = {hashlib.md5(a["text"].encode("utf-8")).hexdigest(): a["text"] for i, a in enumerate(all_highlights[0]["highlights"])}
    print(nodes)
    edges = []
    edge_id = 0

    for g,a in zip(all_generations, all_highlights[0]["highlights"]):
        highlight_hash = hashlib.md5(a["text"].encode("utf-8")).hexdigest()
        for c, text  in g.items():
            print(text)
            try:
                best_results = retriever.get_relevant_documents(remove_quotes_from_string(text))
            except Exception as e:
                print("issue with %s" % text)
                best_results = []
            best_results_text = [ _c.page_content for _c in best_results ]
            for _b in best_results_text:
                _h = hashlib.md5(_b.encode("utf-8")).hexdigest()
                nodes[_h] = _b
                edges.append({"edge_id": edge_id, "from": highlight_hash, "to": _h, "type": c})
                edge_id += 1

    final_json = {
        "nodes":[{"id": id, "text": text} for id, text in nodes.items()],
        "edges": edges
    }

    with open("graph.json", "w") as f:
        json.dump(final_json, f)

    # print(fina)