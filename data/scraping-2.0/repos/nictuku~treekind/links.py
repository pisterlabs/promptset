import openai

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from llama_index.vector_stores.types import VectorStoreQuery

import logging
import sys
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('links_template.html')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.log = "debug"


try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    logging.info("Loading index from storage")
    index = load_index_from_storage(storage_context)
    logging.info("Index loaded")
except FileNotFoundError:
    logging.info("Loading documents")
    documents = SimpleDirectoryReader('data', recursive=True,
                                      num_files_limit=1500,
                                      required_exts=[".md"]).load_data()
    logging.info("Loaded %d documents", len(documents))
    logging.info("Building index")
    index = VectorStoreIndex.from_documents(documents)
    logging.info("Index built")
    logging.info("Storing index")
    index.storage_context.persist()
    logging.info("Index stored")

from openai.embeddings_utils import get_embedding, cosine_similarity
retriever = index.as_retriever()

def get_closest_nodes(doc_id):
    # wip
    embeds = index.vector_store.get(doc_id)
    query = VectorStoreQuery(embeds, 2)
    results = index.vector_store.query(query)
    # VectorStoreQueryResult(nodes=None, similarities=[1.0, 0.7893158727497774], ids=['f0830e4a-2742-47eb-8539-f0a25ff55b40', 'df3e4c08-c063-4ec9-b801-40dd23a2140b'])
    
    # return the ids of the closest nodes, as long as they're not the same as the doc_id
    # and similarity is higher than 0.8
    # Remember nodes is None, so we have to work with the two other lists

    nodes = []
    for i, sim in enumerate(results.similarities):
        if sim > 0.8 and results.ids[i] != doc_id:
            nodes.append(results.ids[i])

    return nodes


all_docs = index.docstore.docs.keys()

MAX_DOCS = 2000

# write an HTML file containing a table with all docs and their respective links
# column one is the document content, column two are the links to other docs
# in each line, add an anchor for the doc_id for that row.

def gen_doc_links():
    i = 0
    for doc_id in all_docs:
        nodes = get_closest_nodes(doc_id)
        yield doc_id, nodes
        i += 1
        if i > MAX_DOCS:
            break

# import a markdown to html converter
from markdown import markdown


def gen_doc_links_html():
    for doc_id, nodes in gen_doc_links():
        logging.info("Generating HTML for %s", doc_id)
        node_links = []
        for node in nodes:
            node = index.docstore.get_node(node)
            node_links.append(f'<a href="#{node.id_}">{node.id_}</a>')
        text = index.docstore.get_node(doc_id).text
        markdown_text = markdown(text)
        #yield f'<tr><td id="{doc_id}">{markdown_text}</td><td>{", ".join(node_links)}</td></tr>'
        # if people go to #doc_id they should scroll to this row
        yield f'<tr><td><a id="{doc_id}"></a>{markdown_text}</td><td>{", ".join(node_links)}</td></tr>'
        

links_html = "".join(gen_doc_links_html())

rendered_html = template.render(table=links_html)

with open("md.html", "w") as f:
    f.write(rendered_html)

# now, we can use the links.html file to navigate through the documents
logging.info("Done")
#doc = "162b987d-b61c-47d0-a71b-b631708cc164"
##x = index.vector_store.get(doc)
##
##sim = cosine_similarity(x, x)
##print(sim)
#import pdb;pdb.set_trace()
#print(get_links(doc))
#retriever = index.as_retriever()
#nodes = retriever.retrieve("Qual atividade estou planejando fazer com o Samuel?")
#for node in nodes:
#    print(node)
#    print()
