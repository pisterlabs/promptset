import pandas as pd
import numpy as np
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random
import networkx as nx
import seaborn as sns
from pyvis.network import Network
from helpers.df_helpers import documents2Dataframe, df2Graph, graph2Df

# Constants
DATA_DIR = "Anthropic"
INPUT_DIRECTORY = Path(f"../RAG/data/txt/{DATA_DIR}")
OUTPUT_DIRECTORY = Path(f"./data_output/{DATA_DIR}")
GRAPH_OUTPUT_DIRECTORY = "./docs/index.html"

def load_documents(directory):
    loader = DirectoryLoader(directory, show_progress=True)
    return loader.load()

def split_documents(documents, chunk_size=1500, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    docs = splitter.split_documents(documents)
    print("Split into chunks:", len(docs), "chunks")
    return docs

def process_documents(pages, regenerate=True):
    df = documents2Dataframe(pages)
    if regenerate:
        # concepts_list = df2Graph(df, model='zephyr:latest')
        concepts_list = df2Graph(df, OAI=True)
        dfg1 = graph2Df(concepts_list)
        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)
        dfg1.to_csv(OUTPUT_DIRECTORY/"graph.csv", sep="|", index=False)
        df.to_csv(OUTPUT_DIRECTORY/"chunks.csv", sep="|", index=False)
    else:
        dfg1 = pd.read_csv(OUTPUT_DIRECTORY/"graph.csv", sep="|")
    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4
    return dfg1

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2


def create_graph(dfg1, dfg2):
    dfg = pd.concat([dfg1, dfg2], axis=0)
    dfg = (dfg.groupby(["node_1", "node_2"])
           .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
           .reset_index())
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    G = nx.Graph()
    for node in nodes:
        G.add_node(str(node))
    for index, row in dfg.iterrows():
        G.add_edge(str(row["node_1"]), str(row["node_2"]), title=row["edge"], weight=row['count']/4)
    return G

def colors2Community(communities, palette) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

def visualize_graph(G, communities):
    colors = colors2Community(communities, palette="hls")
    for index, row in colors.iterrows():
        G.nodes[row['node']]['group'] = row['group']
        G.nodes[row['node']]['color'] = row['color']
        G.nodes[row['node']]['size'] = G.degree[row['node']]
    net = Network(notebook=False, cdn_resources="remote", height="900px", width="100%", select_menu=True)
    net.from_nx(G)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    net.show_buttons(filter_=["physics"])
    net.show(GRAPH_OUTPUT_DIRECTORY)

# Main execution
def main():
    documents = load_documents(INPUT_DIRECTORY)
    pages = split_documents(documents)
    dfg1 = process_documents(pages, regenerate=True)
    dfg2 = contextual_proximity(dfg1)
    G = create_graph(dfg1, dfg2)
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    visualize_graph(G, communities)

if __name__ == "__main__":
    main()
