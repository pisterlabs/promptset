import json
import os
import re
import pickle
import networkx as nx
from langchain import PromptTemplate, LLMChain

def check_nodes_in_descriptions(graph, descriptions):
    """Checks if all nodes in a graph have corresponding descriptions.

    Args:
        graph (networkx.classes.graph.Graph): The graph to be checked.
        descriptions (dict): A dictionary of descriptions.

    Returns:
        bool: True if all nodes have descriptions, else raises ValueError.

    Raises:
        ValueError: If a node does not have a description.
    """
    for node in graph.nodes():
        if node not in descriptions:
            print(f"The node '{node}' does not have a corresponding key in the descriptions dictionary.")
            raise ValueError("Missing description for node.")
    return True


def already_know(data, node_a, node_b):
    """Checks if a pair of nodes is already in the data.

    Args:
        data (dict): Dictionary with known data.
        node_a (str): First node.
        node_b (str): Second node.

    Returns:
        bool: True if nodes are already in the data, False otherwise.
    """
    key = tuple(sorted((node_a, node_b)))

    if key in data:
        return data[key]
    else:
        return False


def add_to_dict(data, key1, key2, value):
    """Adds a key-value pair to the dictionary.

    Args:
        data (dict): The dictionary to add to.
        key1 (str): First part of the key.
        key2 (str): Second part of the key.
        value (str): The value to be added.

    Returns:
        dict: The dictionary with the new key-value pair added.
    """
    key = tuple(sorted((key1, key2)))
    data[key] = value
    return data


def remove_from_dict(data, key1, key2):
    """Removes a key from the dictionary.

    Args:
        data (dict): The dictionary to remove from.
        key1 (str): First part of the key.
        key2 (str): Second part of the key.

    Returns:
        dict: The dictionary with the key removed.
    
    Side effects:
        Prints to console if the key is not found in the dictionary.
    """
    key = tuple(sorted((key1, key2)))
    if key in data:
        del data[key]
    else:
        print("Key not found in the dictionary")
    return data


def _process_edge(node_u, node_v, domain, descriptions, llm_chain, prompt_single, 
                  graph, file_path, data_path, immutable_features, classification_node):
    """Process an edge between two nodes to determine their connection.

    Args:
        node_u, node_v: The nodes to be connected.
        domain: Domain of the data.
        descriptions: Descriptions of the nodes.
        llm_chain: A pre-initialized LLMChain object.
        prompt_single: A PromptTemplate object.
        graph: The graph to which the edge is to be added.
        file_path: Path to the output file.
        data_path: Path to the data file.
        immutable_features: Immutable features of the graph.
        classification_node: The node used for classification.
    """
    result = None

    if os.path.exists(data_path) and os.path.getsize(data_path) > 2:
        with open(data_path, 'rb') as data_file:
            data = pickle.load(data_file)
    else:
        data = {}

    if not already_know(data, node_u, node_v):
        result = llm_chain.predict(domain=domain, a_node=node_u, b_node=node_v,
                                   a_description=descriptions[node_u], 
                                   b_description=descriptions[node_v])
        match = re.search(r"<Answer>([A-Z])</Answer>", result)
        if not match:
            return
        letter = match.group(1)
        data = add_to_dict(data, node_u, node_v, letter)

        with open(data_path, 'wb') as data_file:
            pickle.dump(data, data_file)
    else:
        print(f"No need to look forward for {node_u} and {node_v}, already checked")

    if letter in ["A", "B"]:
        src_node, dest_node = (node_u, node_v) if letter == "A" else (node_v, node_u)

        if dest_node in immutable_features or src_node == classification_node:

            with open(data_path, 'rb') as data_file:
                data = pickle.load(data_file)

            data = remove_from_dict(data, src_node, dest_node)

            with open(data_path, 'wb') as data_file:
                pickle.dump(data, data_file)

            error_node = dest_node if dest_node in immutable_features else src_node
            print(f"ERROR {error_node} is not valid!\n")
            print('-'*10 + "\n")
            return

        graph.add_edge(src_node, dest_node)
        print(f"{src_node} ---> {dest_node} added")

    if result is not None:
        _append_text_to_file(file_path, node_u, node_v, domain, descriptions, prompt_single, result)


def _append_text_to_file(file_path, source_node, destination_node, domain, 
                         descriptions, prompt_single, result):
    """Appends text to a file.

    Args:
        file_path (str): The path to the file.
        source_node (str): The source node of the edge.
        destination_node (str): The destination node of the edge.
        domain (str): Domain of the data.
        descriptions (dict): Dictionary with node descriptions.
        prompt_single (PromptTemplate): A PromptTemplate object.
        result (str): LLM response.

    Prints:
        The appended text to the console.

    Side effects:
        Appends the generated text to the file at the given path.
    """
    text_to_append = (
        f"U node: {source_node}\n"
        f"V node: {destination_node}\n"
        f"{prompt_single.format(domain=domain, a_node=source_node, b_node=destination_node, a_description=descriptions[source_node], b_description=descriptions[destination_node])}\n"
        "LLM RESPONSE:\n"
        f"{result}\n"
        f"{'-' * 10}\n"
    )

    print(text_to_append)  # Printing to console

    with open(file_path, 'a') as file:
        file.write(text_to_append)




def get_direct_graph(skeleton, descriptions, immutable_features, domain, file_path,
                     data_path, classification_node, model):
    """Gets a direct graph based on a skeleton graph.

    Args:
        skeleton: The skeleton graph.
        descriptions: Descriptions of the nodes in the graph.
        immutable_features: The immutable features of the graph.
        domain: Domain of the data.
        file_path: Path to the output file.
        data_path: Path to the data file.
        classification_node: The node used for classification.
        model: The model used to classify.

    Returns:
        A directed graph based on the skeleton graph.
    """
    if domain is None or domain.strip() == '':
        raise ValueError("Error: Domain is not provided or empty.")

    if not check_nodes_in_descriptions(skeleton, descriptions):
        raise ValueError("Error: Missing description for one or more nodes.")

    single_template = (
                            """You are a helpful assistant to a {domain} expert.\n"""
                            """Which of the following counterfactual scenarios is plausible?"""
                            """You must select one option, the most probable.\n"""
                            """A. If the value of {a_node} ({a_description}) was altered, """
                            """could it lead to a change in the value of {b_node} ({b_description})?\n"""
                            """B. If the value of {b_node} ({b_description}) was altered, """
                            """could it lead to a change in the value of {a_node} ({a_description})?\n"""
                            """C. None of the above. The scenarios presented are not plausible.\n\n"""
                            """Let's think step-by-step to make sure that we have the right answer and write the explanations.\n\n"""
                            """Then provide your final answer within the tags, <Answer>A/B/C</Answer>."""
                        )
    
    prompt_single = PromptTemplate(
        template=single_template,
        input_variables=["domain", "a_node", "a_description", "b_node", "b_description"]
    )

    llm_chain = LLMChain(prompt=prompt_single, llm=model)
    graph = nx.DiGraph()

    for node in skeleton.nodes():
        graph.add_node(node)

    for node_a, node_b in skeleton.edges():
        _process_edge(node_a, node_b, domain, descriptions, llm_chain, 
                      prompt_single, graph, file_path, data_path, 
                      immutable_features, classification_node)

    return graph


def get_all_direct_graphs(result, descriptions, immutable_features, domain, result_dir,
                          classification_node, model):
    """Gets all direct graphs based on the given result.

    Args:
        result: The causal skeleton result.
        descriptions: Descriptions of the nodes in the graph.
        immutable_features: The immutable features of the graph.
        domain: Domain of the data.
        result_dir: Directory to save the results.
        classification_node: The node used for classification.
        model: The model used to classify.

    Returns:
        A dictionary containing all the direct graphs.
    """
    direct_graphs = {}
    for method, graph in result.items():
        if isinstance(graph, nx.Graph):
            file_path = os.path.join(result_dir, f"{method}_explanations.txt")
            open(file_path, 'w').close()

            data = {}
            data_path = os.path.join(result_dir, f"{method}_data.pkl")
            with open(data_path, 'w') as file:
                json.dump(data, file)

            try:
                direct_graph = get_direct_graph(graph, descriptions, immutable_features, domain, 
                                                file_path, data_path, classification_node, model)
                direct_graphs[method] = direct_graph
            except ValueError as error:
                print(f"Error in get_direct_graph for method {method}: {str(error)}")
        else:
            print(f"Error: the value for method {method} is not a Graph: {graph}")

    return direct_graphs
