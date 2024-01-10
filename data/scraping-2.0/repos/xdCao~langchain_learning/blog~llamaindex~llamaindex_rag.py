import os
import getpass
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode
from llama_index.embeddings import resolve_embed_model
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.response.notebook_utils import display_source_node
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import RecursiveRetriever

os.environ["OPENAI_API_KEY"] = getpass.getpass("输入apiKey: ")

docs = TrafilaturaWebReader().load_data(
    [
        "https://baike.baidu.com/item/ChatGPT/62446358",
        "https://baike.baidu.com/item/恐龙/139019"
    ]
)

print("docs: ", len(docs))
# print(docs[0].text[:1000])

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)

base_nodes = node_parser.get_nodes_from_documents(docs)
print("nodes: ", len(base_nodes))

# print(base_nodes[0])

embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")

llm = OpenAI(model="gpt-3.5-turbo")

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

base_index = VectorStoreIndex(base_nodes, service_context=service_context)
base_retriever = base_index.as_retriever(similarity_top_k=2)
# retrievers = base_retriever.retrieve("恐龙是冷血动物吗")

# for n in retrievers:
#     display_source_node(n, source_length=1500)
response = RetrieverQueryEngine(retriever=base_retriever).query("恐龙是冷血动物吗")
print(str(response))

sub_chunk_sizes = [128, 256, 512]
sub_node_parsers = [
    SimpleNodeParser.from_defaults(chunk_size=c, chunk_overlap=0) for c in sub_chunk_sizes
]

all_nodes = []
for base_node in base_nodes:
    for n in sub_node_parsers:
        sub_nodes = n.get_nodes_from_documents([base_node])
        sub_inodes = [
            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
        ]
        all_nodes.extend(sub_inodes)
    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)

all_nodes_dict = {n.node_id: n for n in all_nodes}

vector_index_trunk = VectorStoreIndex(all_nodes, service_context=service_context)
vector_retriever_chunk = vector_index_trunk.as_retriever(similarity_top_k=2)
retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True
)

response = RetrieverQueryEngine(retriever=retriever_chunk).query("恐龙是冷血动物吗")
