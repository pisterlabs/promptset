# -*- coding: utf-8 -*-
"""Original file is located at
    https://colab.research.google.com/github/cgint/llama_index_t1/blob/main/LlamaIndex_Simple_Graph_RAG.ipynb

## Graph LLM, Demo Outline

Got it from [Graph_RAG_LlamaIndex_Workshop.ipynb](https://colab.research.google.com/drive/1tLjOg2ZQuIClfuWrAC2LdiZHCov8oUbs#scrollTo=s5LPkzt1YUIN)

### Graph RAG

> Graph RAG with LLM

![Graph RAG](https://github.com/siwei-io/talks/assets/1651790/f783b592-7a8f-4eab-bd61-cf0837e83870)


> Query time workflow:

- Get Key Entities/Relationships related to task
  - LLM or NLP to extract from task string
  - Expand synonyms
- Get SubGraphs
  - Exact matching of "Starting Point"
  - Optionally Embedding based
- Generate answer based on SubGraphs
  - Could be combined with other RAG
  - If KG was built with LlamaIndex, metadata could be leveraged

> Values

- KG is __ of Knowledge:
  - Refined and Concise Form
  - Fine-grained Segmentation
  - Interconnected-structured nature
- Knowledge in (existing) KG is Accurate
- Query towards KG is Stable yet Deterministic
- Reasoning/Info. in KG persist domain knowledge

> Refs:

- https://gpt-index.readthedocs.io/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html
- https://gpt-index.readthedocs.io/en/stable/examples/query_engine/knowledge_graph_rag_query_engine.html
- https://gpt-index.readthedocs.io/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo.html
- https://siwei.io/talks/graph-rag-with-jerry/
- https://www.youtube.com/watch?v=bPoNCkjDmco

### KG Building

![KG Building](https://github.com/siwei-io/talks/assets/1651790/495e035e-7975-4b77-987a-26f8e1d763d2)

> Value

- Game-changer for ROI on adaptation of Graph
  - NLP Competence and efforts
  - Complex Pipelines
- Those "nice to have" graphs can now be enabled by Graph at a small cost

> Refs
- https://gpt-index.readthedocs.io/en/stable/examples/index_structs/knowledge_graph/NebulaGraphKGIndexDemo.html#instantiate-gptnebulagraph-kg-indexes
- https://gpt-index.readthedocs.io/en/stable/examples/query_engine/knowledge_graph_query_engine.html
- https://colab.research.google.com/drive/1G6pcR0pXvSkdMQlAK_P-IrYgo-_staxd?usp=sharing
- https://siwei.io/en/demos/text2cypher/
- https://siwei.io/demo-dumps/kg-llm/KG_Building.ipynb
- https://siwei.io/demo-dumps/kg-llm/KG_Building.html

# How

with LlamaIndex and Simple Graph in memory

## Concepts

REF: https://gpt-index.readthedocs.io/en/stable/getting_started/concepts.html

### RAG

Retrieval Augmented Generation:

![](https://gpt-index.readthedocs.io/en/stable/_images/rag.jpg)

### Indexing Stage

![](https://gpt-index.readthedocs.io/en/stable/_images/indexing.jpg)
- [Data Connectors(LlamaHub)](https://gpt-index.readthedocs.io/en/stable/core_modules/data_modules/connector/root.html)

- Documents
  - Nodes(Chunk)
- Index
  - VectorIndex
  - [KnowledgeGraphIndex](https://gpt-index.readthedocs.io/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo.html), create KG from data, Graph RAG
  - SQLIndex


### Querying Stage

- Query Engine/Chat Engine Agent(input text, output answer)
  - [KnowledgeGraphQueryEngine](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/knowledge_graph_query_engine.html), Text2Cypher Query engine
- Retriever(input text, output nodes)
  - [KnowledgeGraphRAGRetriever](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/knowledge_graph_rag_query_engine.html), for existing KG wired as Graph RAG
- Node Postprocessor(Reranking, filterring nodes)
- Response Synthesizer(input nodes, output answer)

![](https://gpt-index.readthedocs.io/en/stable/_images/querying.jpg)


### Context

REF:
- https://gpt-index.readthedocs.io/en/stable/core_modules/supporting_modules/service_context.html
- https://gpt-index.readthedocs.io/en/stable/api_reference/storage.html



Service context

- LLM
- Embedding Model
- Prompt Helper

Storage context

- Vector Store
- Graph Store

## Key KG related components

- [KnowledgeGraphIndex](https://gpt-index.readthedocs.io/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo.html) is an Index to:
  - Indexing stage:
    - Extract data into KG with LLM or any other callable models
    - Persist KG data into `storeage_context.graph_store`
  - Querying stage:
    - `as_query_engine()` to enable 0-shot Graph RAG
    - `as_retriever()` to create an advanced Graph involving RAG
- [KnowledgeGraphRAGRetriever](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/knowledge_graph_rag_query_engine.html)
  - Instanctiate:
    - Create a `storeage_context.graph_store` as the init argument.
  - Querying stage:
    - pass to `RetrieverQueryEngine` to become a Graph RAG query engine on any existing KG
    - combined with other RAG pipeline

- [KnowledgeGraphQueryEngine](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/knowledge_graph_query_engine.html), Text2Cypher Query engine
  - Instanctiate:
    - Create a `storeage_context.graph_store` as the init argument.
  - Querying stage:
    - Text2cypher to get answers towards the KG in graph_store.
    - Optionally, `generate_query()` only compose a cypher query.

## Preparation

Install Dependencies, prepare for contexts of Llama Index
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install openai llama_index pyvis

# For OpenAI


import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)

from llama_index import (
    VectorStoreIndex,
    KnowledgeGraphIndex,
    ServiceContext,
)

from llama_index.storage.storage_context import StorageContext

import os
import logging
import sys

def display_output_bold(text):
    print(f"\nOutput: \033[1m{text}\033[0m")

def engine_query_print(chat_engine, question):
    print("\n------------------------------------\n")
    print(f"Input: \033[1m{question}\033[0m\n")
    display_output_bold(chat_engine.query(question))
    print("------------------------------------\n")

def engine_chat_print(chat_engine, question):
    print("\n------------------------------------\n")
    print(f"Input: \033[1m{question}\033[0m\n")
    display_output_bold(chat_engine.chat(question))
    print("------------------------------------\n")

be_verbose = False


from llama_index import set_global_service_context

# define LLM
#from llama_index.llms import OpenAI
#os.environ["OPENAI_API_KEY"] = "dummy" # "sk-..."
#os.environ["OPENAI_API_BASE"] = "http://host.docker.internal:1234"
#llm = OpenAI(temperature=0, model="gpt-3.5-turbo") # gpt-3.5-turbo-1106, gpt-4-1106-preview
from llama_index.llms import Ollama

llm_model = "neural-chat" # "gpt-3.5-turbo-1106", "gpt-4-1106-preview"
llm_base_url = "http://host.docker.internal:11434"
print(f"About to instanciate LLM {llm_model} on {llm_base_url} ...")
llm = Ollama(model=llm_model, base_url=llm_base_url, request_timeout=300, temperature=0)

# set global service context
from llama_index.embeddings import FastEmbedEmbedding
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2" # "BAAI/bge-small-en-v1.5"
print(f"About to instanciate Embed Model {embed_model_name} using FastEmbedEmbedding ...")
embed_model = FastEmbedEmbedding(model_name=embed_model_name, cache_dir="/data/fastembed_cache/")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512, embed_model=embed_model)
#service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512, embed_model="local")
set_global_service_context(service_context)

"""## Create a Graph Space

KnowledgeGraphIndex on SimpleGraphStore

## Storage_context with Graph_Store
"""

from llama_index.graph_stores import SimpleGraphStore
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

"""## 🏗️ KG Building with Llama Index

### Preprocess Data with data connectors

with `WikipediaReader`

We will download and preprecess data from:
    https://en.wikipedia.org/wiki/Guardians_of_the_Galaxy_Vol._3
"""

from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()

documents = loader.load_data(pages=['Guardians of the Galaxy Vol. 3'], auto_suggest=False)

"""### Indexing Extract Triplets and Save to KnowledgeGraph

with `KnowledgeGraphIndex`

This call will take some time, it'll extract entities and relationships and store them into KnowledgeGraph
"""
from llama_index import load_index_from_storage
kg_index_storage_dir = '/data/storage_graph'
if not os.path.exists(kg_index_storage_dir):
  print(f"About to build graph-index over {len(documents)} document(s) ...")
  kg_index = KnowledgeGraphIndex.from_documents(
      documents,
      storage_context=storage_context,
      service_context=service_context,
      max_triplets_per_chunk=10,
      include_embeddings=True
  )
  print(f"Storing graph-index to {kg_index_storage_dir} ...")
  kg_index.storage_context.persist(persist_dir=kg_index_storage_dir)
else:
  print(f"Loading graph-index from storage from {kg_index_storage_dir} ...")
  storage_context = StorageContext.from_defaults(persist_dir=kg_index_storage_dir)
  kg_index = load_index_from_storage(
      storage_context=storage_context,
      service_context=service_context
  )


# Assuming kg_index is a NetworkX graph
from pyvis.network import Network
net = Network(notebook=False, directed=True)
net.from_nx(kg_index.get_networkx_graph())
net.save_graph("/data/example.html")

"""## 🧠 Graph RAG

### KG_Index as **Query Engine**
"""
print("====================================")
print("     KG_Index as Query Engine")
print("====================================")

kg_index_query_engine = kg_index.as_query_engine(
    retriever_mode="keyword",
    verbose=be_verbose,
    response_mode="tree_summarize",
)

engine_query_print(kg_index_query_engine, "Who is Rocket?")
engine_query_print(kg_index_query_engine, "who is Lylla?")
engine_query_print(kg_index_query_engine, "who is Groot?")
engine_query_print(kg_index_query_engine, "What challenges do Rocket and Lylla face?")


"""See also here for comparison of text2cypher & GraphRAG
- https://user-images.githubusercontent.com/1651790/260617657-102d00bc-6146-4856-a81f-f953c7254b29.mp4
- https://siwei.io/en/demos/text2cypher/

> While another idea is to retrieve in both ways and combine the context to fit more use cases.

### Graph RAG on any existing KGs

with `KnowledgeGraphRAGRetriever`.

REF: https://gpt-index.readthedocs.io/en/stable/examples/query_engine/knowledge_graph_rag_query_engine.html#perform-graph-rag-query
"""

print("====================================")
print("     KnowledgeGraphRAGRetriever")
print("====================================")

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import KnowledgeGraphRAGRetriever

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=be_verbose,
)

lg_rag_ret_query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever, service_context=service_context
)
engine_query_print(lg_rag_ret_query_engine, "Who is Rocket?")
engine_query_print(lg_rag_ret_query_engine, "who is Lylla?")
engine_query_print(lg_rag_ret_query_engine, "who is Groot?")
engine_query_print(lg_rag_ret_query_engine, "What challenges do Rocket and Lylla face?")


"""### Example of Graph RAG Chat Engine

#### The context mode
"""

print("====================================")
print("           Context mode")
print("====================================")

from llama_index.memory import ChatMemoryBuffer

chat_engine = kg_index.as_chat_engine(
    chat_mode="context",
    memory=ChatMemoryBuffer.from_defaults(token_limit=6000),
    verbose=be_verbose
)

engine_chat_print(chat_engine, "who is Rocket?")
engine_chat_print(chat_engine, "who is Lylla?")
engine_chat_print(chat_engine, "who is Groot?")
engine_chat_print(chat_engine, "do they all know each other?")
engine_chat_print(chat_engine, "But how about Lylla?")
engine_chat_print(chat_engine, "who of them are human?")


"""Above chat_engine won't eval the "them" when doing RAG, this could be resolved with ReAct mode!

We can see, now the agent will refine the question towards RAG before the retrieval.

#### The ReAct mode
"""

print("====================================")
print("           ReAct mode")
print("====================================")

chat_engine = kg_index.as_chat_engine(
    chat_mode="react",
    memory=ChatMemoryBuffer.from_defaults(token_limit=6000),
    verbose=be_verbose
)
engine_chat_print(chat_engine, "who is Rocket?")
engine_chat_print(chat_engine, "who is Lylla?")
engine_chat_print(chat_engine, "who is Groot?")
engine_chat_print(chat_engine, "do they all know each other?")
engine_chat_print(chat_engine, "But how about Lylla?")
engine_chat_print(chat_engine, "who of them are human?")


"""Refs:
- https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
- https://github.com/wey-gu/demo-kg-build/blob/main/graph_rag_chatbot.py
- https://llamaindex-chat-with-docs.streamlit.app/
"""

from IPython.display import HTML

HTML("""
<iframe src="https://player.vimeo.com/video/857919385?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" width="1080" height="525" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" title="chat_graph_rag_demo"></iframe>
""")

"""### Graph RAG with Text2Cypher"""
#
# The following would only make sense when we could set with_graphquery=True, which is not possible with SimpleStorage. But waas possible with NebulaGraph.
#
# graph_rag_retriever_with_graphquery = KnowledgeGraphRAGRetriever(
#     storage_context=storage_context,
#     service_context=service_context,
#     llm=llm,
#     verbose=be_verbose,
#     with_graphquery=False, # otherwise not possible with SimpleStorage
# )

# query_engine_with_graphquery = RetrieverQueryEngine.from_args(
#     graph_rag_retriever_with_graphquery, service_context=service_context
# )

# response = query_engine_with_graphquery.query("Tell me about Rocket?")

# display_output_bold(response)

"""### Combining Graph RAG and Vector Index

REF: https://gpt-index.readthedocs.io/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html

```
                  ┌────┬────┬────┬────┐                  
                  │ 1  │ 2  │ 3  │ 4  │                  
                  ├────┴────┴────┴────┤                  
                  │  Docs/Knowledge   │                  
┌───────┐         │        ...        │       ┌─────────┐
│       │         ├────┬────┬────┬────┤       │         │
│       │         │ 95 │ 96 │    │    │       │         │
│       │         └────┴────┴────┴────┘       │         │
│ User  │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─▶   LLM   │
│       │                                     │         │
└───────┘  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  └─────────┘
    │          ┌──────────────────────────┐        ▲     
    └──────┼──▶│  Tell me ....., please   │├───────┘     
               └──────────────────────────┘              
           │┌────┐ ┌────┐                  │             
            │ 3  │ │ 96 │ x->y, x<-z->b,..               
           │└────┘ └────┘                  │             
            ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
```

#### Vector Index creation
"""
from llama_index import load_index_from_storage
vector_storage_dir = '/data/storage_vector'
if not os.path.exists(vector_storage_dir):
  print(f"About to build vector-index over {len(documents)} document(s) ...")
  vector_index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
  )
  print(f"Storing vector-index to {vector_storage_dir} ...")
  vector_index.storage_context.persist(persist_dir=vector_storage_dir)
else:
  print(f"Loading vector-index from storage from {vector_storage_dir} ...")
  storage_context_vector = StorageContext.from_defaults(persist_dir=vector_storage_dir)
  vector_index = load_index_from_storage(
    service_context=service_context,
    storage_context=storage_context_vector
  )

vector_rag_query_engine = vector_index.as_query_engine()

"""## "Cherry-picked" Examples that KG helps

### Top-K Retrieval, nature of information distribution and segmentation

See more from [here](https://siwei.io/graph-enabled-llama-index/kg_and_vector_RAG.html).

> Tell me events about NASA.

|        | VectorStore                                                  | Knowledge Graph + VectorStore                                | Knowledge Graph                                              |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Answer | NASA scientists report evidence for the existence of a second Kuiper Belt,<br>which the New Horizons spacecraft could potentially visit during the late 2020s or early 2030s.<br>NASA is expected to release the first study on UAP in mid-2023.<br>NASA's Venus probe is scheduled to be launched and to arrive on Venus in October,<br>partly to search for signs of life on Venus.<br>NASA is expected to start the Vera Rubin Observatory, the Qitai Radio Telescope,<br>the European Spallation Source and the Jiangmen Underground Neutrino.<br>NASA scientists suggest that a space sunshade could be created by mining the lunar soil and<br> launching it towards the Sun to form a shield against global warming. | NASA announces future space telescope programs on May 21.<br>**NASA publishes images of debris disk on May 23. NASA discovers exoplanet LHS 475 b on May 25.**<br>NASA scientists present evidence for the existence of a second Kuiper Belt on May 29.<br>NASA confirms the start of the next El Niño on June 8.<br>NASA produces the first X-ray of a single atom on May 31.<br>NASA reports the first successful beaming of solar energy from space down to a receiver on the ground on June 1.<br>NASA scientists report evidence that Earth may have formed in just three million years on June 14.<br>NASA scientists report the presence of phosphates on Enceladus, moon of the planet Saturn, on June 14.<br>NASA's Venus probe is scheduled to be launched and to arrive on Venus in October.<br>NASA's MBR Explorer is announced by the United Arab Emirates Space Agency on May 29.<br>NASA's Vera Rubin Observatory is expected to start in 2023. | NASA announced future space telescope programs in mid-2023,<br>**published images of a debris disk**, <br>and discovered an exoplanet called **LHS 475 b**. |
| Cost   | 1897 tokens                                                  | 2046 Tokens                                                  | 159 Tokens                                                   |



And we could see there are indeed some knowledges added with the help of Knowledge Graph retriever:

- NASA publishes images of debris disk on May 23.
- NASA discovers exoplanet LHS 475 b on May 25.

The additional cost, however, does not seem to be very significant, at `7.28%`: `(2046-1897)/2046`.

Furthermore, the answer from the knwoledge graph is extremely concise (only 159 tokens used!), but is still informative.

> Takeaway: KG gets Fine-grained Segmentation of info. with the nature of interconnection/global-context-retained, it helps when retriving spread yet important knowledge pieces.

### Hallucination due to w/ relationship in literal/common sense, but should not be connected in domain Knowledge

[GPT-4 (WebPilot) helped me](https://shareg.pt/4zbGI5G) construct this question:

> during their mission on Counter-Earth, the Guardians encounter a mysterious artifact known as the 'Celestial Compass', said to be a relic from Star-Lord's Celestial lineage. Who among the Guardians was tempted to use its power for personal gain?

where, the correlation between knowledge/documents were setup in "common sence", while, they shouldn't be linked as in domain knowledge.

See this picture, they could be considered related w/o knowing they shouldn't be categorized together in the sense of e-commerce.

> Insulated Greenhouse v.s. Insulated Cup
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/siwei-io/talks/assets/1651790/81ff9a61-c961-47c1-80fb-8e5bd9c957bc" alt="104946561_0_final" width="45%">
    <img src="https://github.com/siwei-io/talks/assets/1651790/e587d229-3973-4a3a-856e-0b493ad690eb" alt="104946743_0_final" width="45%">
</div>

> Takeaway: KG reasons things reasonably, as it holds the domain knowledge.
"""

engine_query_print(vector_index.as_query_engine(),
"""
during their mission on Counter-Earth, the Guardians encounter a mysterious artifact known as the 'Celestial Compass', said to be a relic from Star-Lord's Celestial lineage. Who among the Guardians was tempted to use its power for personal gain?
"""
)

engine_query_print(kg_index_query_engine,
"""
during their mission on Counter-Earth, the Guardians encounter a mysterious artifact known as the 'Celestial Compass', said to be a relic from Star-Lord's Celestial lineage. Who among the Guardians was tempted to use its power for personal gain?
"""
)

# backup runtime contexts
#!zip -r workshop_dump.zip openrc storage_graph storage_vector

# restore runtime contexts
#!unzip workshop_dump.zip
