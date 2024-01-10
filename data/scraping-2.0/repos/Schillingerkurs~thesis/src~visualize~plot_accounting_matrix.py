

import os
import openai
from pathlib import Path
import pandas as pd
import sys
import numpy as np
import networkx as nx


import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from itertools import count

HERE = Path(__file__).parent.parent.parent.absolute()



sys.path.insert(0, str(HERE/Path("src", "features")))
import accounting_matrix


def plot_static_graph(G, labels, color_state_map, seed = 13648  ):
    
    plt.figure(figsize=(20,20))
    # Set seed for reproducibility and increase distance between nodes
    pos = nx.spring_layout(G, k=0.5, seed= seed)

    node_sizes = [1 + 5 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    cmap = plt.get_cmap('viridis')
    
    nx.draw_networkx_nodes(G, pos, 
                           node_size=node_sizes,
                               node_color= color_state_map
                           ) 
                           
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
    )
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])
        
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    # Draw labels for nodes with the highest degrees
    nx.draw_networkx_labels(G, pos, labels=labels)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)

    plt.savefig(HERE/Path("assets","accounting_matrix",'sam_15.jpg'), 
                dpi=300, bbox_inches="tight")

social_accounting_matrix = accounting_matrix.get_both(HERE)


sam_15 = social_accounting_matrix['15']

edges = sam_15['matrix']
nodes_label = sam_15['labels']






cluster = sam_15['sam15_categories'].set_index("Column3")["Column5"].to_dict()

cluster_label = {k: str(cluster[v]) if v in cluster.keys() else str(v)
                 for k, v in nodes_label.items() }


#missing_nodes = [k for k in G.nodes() if k not in cluster_label.keys()]


strange_edges = ['mlnd', 'mcap', 'mliv','total']



edges = {k:v for k,v in edges.items() if k[0] not in strange_edges and 
                     k[1] not in strange_edges}

G = nx.DiGraph()
for (node1, node2), weight in edges.items():
    G.add_edge(node1, node2, weight=weight)




# Add node attributes
nx.set_node_attributes(G, nodes_label, 'name')

nx.set_node_attributes(G, cluster_label, 'cluster')


# Get the nodes with the ten highest degrees
# Get the nodes with the five highest in-degrees
top_nodes = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:2]

# Get the labels for these nodes
top_labels = {node: nodes_label[node] for node, _ in top_nodes}

# top_labels = {}

colors = (pd.DataFrame
          .from_dict(nx.get_node_attributes(G,'cluster'), 
                                orient = 'index', columns = ["cluster"])
          .fillna("")
          )




# color_mappings = {"labor": "blue",
#                   "Agricultura | animal| vegeta":"green",
#                   "Taxes": "red"}



color_mappings = {
                  "Transportation and storage| Business services": "red"}

for k,v in color_mappings.items():
    colors.loc[colors['cluster'].str.contains(k),"color_code"] = v

colors['color_code'] = colors['color_code'].replace('nan', np.nan).fillna("grey")
    
color_state_map = colors.set_index('cluster')['color_code'].to_list()
    


plot_static_graph(G, top_labels, color_state_map)















