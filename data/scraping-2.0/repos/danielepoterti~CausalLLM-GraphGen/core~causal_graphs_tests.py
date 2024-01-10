import pickle
from core.llm_causal_inference import add_to_dict, already_know, remove_from_dict
from core.utils import draw_graph
from langchain import PromptTemplate, LLMChain
from dowhy.causal_refuters.graph_refuter import GraphRefuter
from itertools import combinations
from dowhy import CausalModel
import re
import pprint
import networkx as nx
import os

def check_causality_llm(node_a, node_b, descriptions, model, domain):
    """Checks the causality between two nodes using a model in a given domain.

    Args:
        node_a (str): First node to check.
        node_b (str): Second node to check.
        descriptions (dict): Descriptions for the nodes.
        model (Model): The model to use for checking.
        domain (str): The domain of the check.

    Returns:
        tuple: A tuple containing the match group if found, else None, and the response.
    """
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

    # Predicts the causal relationship between the nodes 'u' and 'v' in the domain "credit lending in Germany"
    response = llm_chain.predict(
        domain=domain, 
        a_node=node_a, 
        b_node=node_b, 
        a_description=descriptions[node_a], 
        b_description=descriptions[node_b]
    )

    # Prints a formatted message with the domain, node, and their description information
    print(
        prompt_single.format(
            domain=domain, 
            a_node=node_a, 
            b_node=node_b, 
            a_description=descriptions[node_a], 
            b_description=descriptions[node_b]
        )
    )

    # Prints the prediction result
    print(response)

    # Searches for a specific pattern "<Answer>([A-Z])</Answer>" in the result
    match = re.search(r"<Answer>([A-Z])</Answer>", response)

    # If the pattern is found, returns the letter found else returns None
    if match:
        return match.group(1), response
    else:
        return None, None


def get_violated_ci(model, k):
    """Find violated Conditional Independences (CI) in the given model.

    Args:
        model (Model): The model to check for violated CIs.
        k (int): The number of nodes to check for conditional independence.

    Returns:
        list: A list of violated conditional independences.
    """
    refuter = GraphRefuter(data=model._data)

    all_nodes = list(model._graph.get_all_nodes(include_unobserved=False))
    num_nodes = len(all_nodes)

    array_indices = list(range(num_nodes))
    all_possible_combinations = list(combinations(array_indices, 2))

    conditional_independences = []

    for combination in all_possible_combinations:
        i, j = combination
        node_a = all_nodes[i]
        node_b = all_nodes[j]

        if i < j:
            temp_arr = all_nodes[:i] + all_nodes[i+1:j] + all_nodes[j+1:]
        else:
            temp_arr = all_nodes[:j] + all_nodes[j+1:i] + all_nodes[i+1:]

        k_sized_lists = list(combinations(temp_arr, k))

        for k_list in k_sized_lists:
            if model._graph.check_dseparation([str(node_a)], [str(node_b)], k_list) == True:
                conditional_independences.append([node_a, node_b, k_list])

    independence_constraints = conditional_independences
    refuter.refute_model(independence_constraints=independence_constraints)

    res = [element for element in independence_constraints if element not in refuter._true_implications]

    return res

def get_descriptions(variable_list, description_dict):
    """Generates a string of descriptions for given variables.

    Args:
        variable_list (list): A list of variables to get descriptions for.
        description_dict (dict): A dictionary containing descriptions for the variables.

    Returns:
        str: A string of descriptions for the given variables.
    """
    output = ""
    for variable in variable_list:
        if variable in description_dict:
            output += f'"{variable}": "{description_dict[variable]}",\n'

    output = output.rstrip(',\n')

    return output


def get_cycle(lst):
    """Generates a string representing a cycle in a graph.

    Args:
        lst (list): A list of nodes in the cycle.

    Returns:
        str: A string representing the cycle.
    """
    joined_string = ' --> '.join(lst)
    cyclic_string = joined_string + ' --> ' + lst[0]

    return cyclic_string


def get_options(lst):
    """Generates a string of options for the given list.

    Args:
        lst (list): A list of options.

    Returns:
        str: A string of options.
    """
    output = ''
    for i, item in enumerate(lst):
        output += '{}. \'{}\' --> \'{}\'\n'.format(i+1, item, lst[(i+1)%len(lst)])

    return output


def get_answer(n):
    """Generates a string of possible answers for a given number of options.

    Args:
        n (int): The number of options.

    Returns:
        str: A string of possible answers.
    """
    numbers = '/'.join(str(i) for i in range(1, n+1))

    return f'<Answer>{numbers}</Answer>'


def break_cycle_llm(cycle, descriptions, model, file_path):
    """Breaks a cycle in a causal loop using LLM.

    Args:
        cycle (list): A list of nodes in the cycle.
        descriptions (dict): A dictionary of descriptions for the nodes.
        model (LLM): A LLM model.
        file_path (str): The path to the file to append the results.

    Returns:
        str or None: Returns the number of the cycle that breaks the loop or None if not found.
    """
    descriptions_text = get_descriptions(cycle, descriptions)
    cycle_text = get_cycle(cycle)
    options_text = get_options(cycle)
    answer_text = get_answer(len(cycle))

    single_template = (
        """I have the following variables:\n"""
        """{desc}\n"""
        """These variables create a causal loop:\n"""
        """{cycle}\n"""
        """How can I break the circle? Which of the following relationships are weak? You must select one option, the most probable.\n"""
        """Please select a relationship that breaks the circle:\n"""
        """{options}\n"""
        """Let's think step-by-step to make sure that we have the right answer and write the explanations.\n\n"""
        """Then provide your final answer within the tags (only the number), {answer}"""
    )

    prompt_single = PromptTemplate(template=single_template, 
                                   input_variables=["desc", "cycle", "options", "answer"])

    llm_chain = LLMChain(prompt=prompt_single, llm=model)

    res = llm_chain.predict(desc=descriptions_text, cycle=cycle_text, options=options_text, answer=answer_text)

    try:
        with open(file_path, 'a') as file:
            text_to_append = prompt_single.format(desc=descriptions_text,
                                                  cycle=cycle_text, 
                                                  options=options_text, 
                                                  answer=answer_text)

            file.write(text_to_append + '\n')
            print(text_to_append)

            file.write('LLM RESPONSE:\n')
            print('LLM RESPONSE:')

            file.write(res + '\n')
            print(res)

            file.write('-' * 10 + '\n')
            print('-' * 10)
    except IOError:
        print(f'An error occurred while appending text to the file: {file_path}')

    match = re.search(r"<Answer>([0-9]+)</Answer>", res)

    if match:
        return match.group(1)
    else:
        return None

def get_pair(index, string_list):
    """Gets a pair of strings at the given index and the next index.

    Args:
        index (int): The index of the first string.
        string_list (list): A list of strings.

    Returns:
        list: A list of two strings.
    """
    index = (int(index) - 1) % len(string_list)  # subtract 1 before modulus
    return [string_list[index], string_list[(index + 1) % len(string_list)]]

def remove_relations(graph, nodes):
    """Removes edges between the given nodes in the graph.

    Args:
        graph (Graph): A graph.
        nodes (list): A list of nodes in the graph.

    Returns:
        Graph: The updated graph with the edges removed.
    """
    for i in range(len(nodes)-1):
        if graph.has_edge(nodes[i], nodes[i+1]):  # Check if the edge exists
            graph.remove_edge(nodes[i], nodes[i+1])  # Remove the edge
        else:
            print(f"There is no edge between {nodes[i]} and {nodes[i+1]}")
    return graph

def break_cycles(graph, cycles, descriptions, model, file_path, data_path):
    """Breaks the cycles in the graph.

    Args:
        graph (Graph): The graph with cycles.
        cycles (list): The list of cycles in the graph.
        descriptions (dict): Descriptions of nodes.
        model (Model): The trained model.
        file_path (str): The path to the file to store the broken cycles.
        data_path (str): The path to the pickle file to store the data.
    """
    for cycle in cycles:
        choice = break_cycle_llm(cycle, descriptions, model, file_path)
        pair = get_pair(choice, cycle)
        graph = remove_relations(graph, pair)
        nx.drawing.nx_agraph.write_dot(graph, "temp/graph.dot")

        with open(data_path, 'rb') as file:
            data = pickle.load(file)

        data = remove_from_dict(data, pair[0], pair[1])

        with open(data_path, 'wb') as file:
            pickle.dump(data, file)

        cycles = list(nx.simple_cycles(graph))


def graph_independence_analysis(graph, df, descriptions, immutable_features, 
                                model_llm, domain, classification_variable, 
                                file_path, data_path):
    """Analyzes the independence of the graph.

    Args:
        graph (Graph): The graph to analyze.
        df (DataFrame): The DataFrame containing the data.
        descriptions (dict): Descriptions of nodes.
        immutable_features (list): The list of immutable features.
        model_llm (Model): The trained model.
        domain (str): The domain of the analysis.
        classification_variable (str): The classification variable.
        file_path (str): The path to the file to store the analysis.
        data_path (str): The path to the pickle file to store the data.
    """
    dir_name = "temp"
    full_dir_path = os.path.join(os.getcwd(), dir_name)

    # Checks if the directory already exists
    if not os.path.exists(full_dir_path):
        os.mkdir(full_dir_path)
    
    nx.drawing.nx_agraph.write_dot(graph, "temp/graph.dot")

    column_names = [col for col in df.columns if col != classification_variable]

    k = 1
    # As long as k is smaller than the number of nodes in the graph minus 1
    while k < len(graph.nodes) - 1:
        print('-'*10)
        print(f'k: {k}')

        cycles = list(nx.simple_cycles(graph))
        for cycle in cycles:
            print(cycle)
        
        if cycles:
            break_cycles(graph, cycles, descriptions, model_llm, file_path, data_path)

        model = CausalModel(
            data=df,
            treatment=column_names,
            outcome=classification_variable,
            graph="temp/graph.dot"
        )    

        res = model.refute_graph(k)

        if not res.refutation_result:
            print('Independencies not satisfied by data:')

            violations = get_violated_ci(model, k)
            pprint.pprint(violations)

            for inner_list in violations:
                del inner_list[-1]

            unique_ci = []
            seen = set()

            for ci in violations:
                ci_tuple = tuple(ci)
                if ci_tuple not in seen:
                    unique_ci.append(ci)
                    seen.add(ci_tuple)

            for ci in unique_ci:

                with open(data_path, 'rb') as file:
                    data = pickle.load(file)

                if already_know(data, ci[0], ci[1]) == False:
                    resllm, res = check_causality_llm(ci[0], ci[1], descriptions, model_llm, domain)

                    print(ci)
                    print(resllm)

                    data = add_to_dict(data, ci[0], ci[1], resllm)

                    # Salva i dati con pickle
                    with open(data_path, 'wb') as file:
                        pickle.dump(data, file)

                    if resllm in ['A', 'B']:
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


                        prompt_single = PromptTemplate(template=single_template, 
                                                    input_variables=["domain", "a_node", "a_description", "b_node", "b_description"]
                        )
                        
                        k = handle_scenario(k, ci, res, domain, prompt_single, descriptions, 
                           graph, file_path, data_path, classification_variable, 
                           immutable_features, resllm)
                else:
                    print(f"No need to look foward for {ci[0]} and {ci[1]}, already checked")
            k += 1
        else:
            k += 1
    
    # Ensure the classification_variable node doesn't have any outgoing edge.
    # If it has, delete them.
    for edge in list(graph.out_edges(classification_variable)):
        graph.remove_edge(*edge)

    # Check if G has one connected component.
    if nx.number_connected_components(graph.to_undirected()) > 1:
        # If not, for each connected component where classification_variable node is not present, 
        # make edges from each node to the classification_variable node.
        for component in nx.connected_components(graph.to_undirected()):
            if classification_variable not in component:
                for node in component:
                    graph.add_edge(node, classification_variable)
                    try:
                        with open(file_path, 'a') as file:
                            text_to_append = "U node: " + node
                            file.write(text_to_append + "\n")
                            text_to_append = "V node: " + classification_variable
                            file.write(text_to_append + "\n")
                            text_to_append = f"The relationship {node} ---> {classification_variable} has been forced since {node} was not in the same connected component of {classification_variable}"
                            file.write(text_to_append + "\n")
                        
                        
                    except IOError:
                        print(f"An error occurred while appending text to the file: {file_path}")

    return graph
 
def handle_scenario(k, ci, res, domain, prompt_single, descriptions, graph, file_path,
                    data_path, classification_variable, immutable_features, order):
    """Handles scenarios when 'resllm' is 'A' or 'B'.

    Args:
        k (int): Current count.
        ci (list): List of nodes.
        res (str): Result string.
        domain (str): The domain of the analysis.
        prompt_single (PromptTemplate): The prompt.
        descriptions (dict): Descriptions of nodes.
        graph (nx.Graph): The graph to analyze.
        file_path (str): The path to the file to store the analysis.
        data_path (str): The path to the pickle file to store the data.
        classification_variable (str): The classification variable.
        immutable_features (list): The list of immutable features.
        order (str): 'A' for ci[0]->ci[1] scenario and 'B' for ci[1]->ci[0] scenario.

    Returns:
        int: The updated value of k.
    """
    source_node, target_node = (ci[0], ci[1]) if order == 'A' else (ci[1], ci[0])

    if target_node not in immutable_features:
        if source_node != classification_variable:
            k = 0
            add_edge_and_write_file(ci, res, domain, prompt_single, descriptions,
                                    graph, file_path, source_node, target_node)
        else:
            with open(data_path, 'rb') as file:
                data = pickle.load(file)

            data = remove_from_dict(data, source_node, target_node)

            with open(data_path, 'wb') as file:
                pickle.dump(data, file)

            print(f'ERROR {source_node} ---> {target_node} is not valid!\n')
            print('-' * 10)
    else:
        print(f'ERROR {target_node} is immutable!')
        print('-' * 10)

    return k


def add_edge_and_write_file(ci, res, domain, prompt_single, descriptions, graph, file_path, 
                            u_node, v_node):
    """Adds an edge to the graph and writes details to the specified file.
    
    Args:
        ci (list): List of nodes.
        res (str): Result string.
        domain (str): The domain of the analysis.
        prompt_single (str): The prompt.
        descriptions (dict): Descriptions of nodes.
        graph (Graph): The graph to modify.
        file_path (str): The path to the file to store the analysis.
        u_node (str): Source node.
        v_node (str): Target node.
    """
    
    graph.add_edge(u_node, v_node)
    print(f'{u_node} ---> {v_node} added')

    try:
        with open(file_path, 'a') as file:
            text_to_append = f'U node: {u_node}\n'
            file.write(text_to_append)
            print(text_to_append)

            text_to_append = f'V node: {v_node}\n'
            file.write(text_to_append)
            print(text_to_append)

            text_to_append = prompt_single.format(
                domain=domain, 
                a_node=ci[0], 
                b_node=ci[1], 
                a_description=descriptions[ci[0]], 
                b_description=descriptions[ci[1]]
            )
            file.write(text_to_append + '\n')
            print(text_to_append)

            file.write('LLM RESPONSE:\n')
            print('LLM RESPONSE:')
            file.write(res + '\n')
            print(res)
            file.write('-'*10 + '\n')
            print('-'*10)
            
    except IOError:
        print(f'An error occurred while appending text to the file: {file_path}')

    nx.drawing.nx_agraph.write_dot(graph, 'temp/graph.dot')



def graphs_independence_analysis(result, df, descriptions, immutable_features, 
                                 domain, classification_variable, result_dir, 
                                 model_llm):
    """Analyzes graph independence and handles exceptions.

    Args:
        result (dict): The dictionary of results.
        df (DataFrame): The dataframe containing data.
        descriptions (dict): Descriptions of nodes.
        immutable_features (list): The list of immutable features.
        domain (str): The domain of the analysis.
        classification_variable (str): The classification variable.
        result_dir (str): The result directory path.
        model_llm (Model): The LLM model.
        
    Returns:
        dict: The dictionary of directed graphs.
    """
    
    directed_graphs = {}
    
    for method, graph in result.items():
        if isinstance(graph, nx.Graph):
            try:
                file_path = f"{result_dir}/{method}_explanations.txt"
                data_path = f"{result_dir}/{method}_data.pkl"
                
                directed_graph = graph_independence_analysis(graph, df, 
                                                             descriptions, 
                                                             immutable_features, 
                                                             model_llm, domain, 
                                                             classification_variable, 
                                                             file_path, data_path)
                directed_graphs[method] = directed_graph
                
            except Exception as e:
                print(f"Error in graph_independence_analysis for method {method}: {str(e)}")
                
        else:
            print(f"Error: the value for method {method} is not a Graph: {graph}")
            
    return directed_graphs


def refute_estimate_pipe(model, identified_estimand, estimate, file_path):
    """Refute estimates and append the result to a file.

    Args:
        model: The model for refuting estimates.
        identified_estimand: The identified estimand for refuting.
        estimate: The estimate to refute.
        file_path: The path to the file to store the refuting results.

    """
    refute_title = "*" * 10 + " REFUTE ESTIMATE " + "*" * 10
    with open(file_path, 'a') as file:
        file.write(refute_title + "\n")
    
    print(refute_title)
    
    refutation_methods = [
        {"method_name": "random_common_cause"},
        {"method_name": "placebo_treatment_refuter", "placebo_type": "permute"},
        {"method_name": "data_subset_refuter", "subset_fraction": 0.9},
    ]
    
    for method in refutation_methods:
        try:
            refute_result = model.refute_estimate(
                identified_estimand, 
                estimate, 
                **method
            )
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(str(refute_result) + "\n")
            print(refute_result)
        except Exception as e:
            print(str(e))

def estimate_and_refute(model, identified_estimand, method_name, file_path):
    """Estimate effects and refute them, then write the result to a file.

    Args:
        model: The model for estimating and refuting effects.
        identified_estimand: The identified estimand for estimation.
        method_name: The method name to use for the estimation.
        file_path: The path to the file to store the estimation and refutation results.
    """
    try:
        estimate = model.estimate_effect(
            identified_estimand, 
            method_name=method_name, 
            test_significance=True
        )
        
        try:
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write("*" * 15 + f" {method_name} " + "*" * 15 + "\n")
                file.write(str(estimate) + "\n")
        except IOError:
            print(f"An error occurred while appending text to the file: {file_path}")
        
        print("*" * 15 + f" {method_name} " + "*" * 15)
        print(estimate)

        refute_estimate_pipe(model, identified_estimand, estimate, file_path)

        sign_title = "*" * 5 + " TEST COVARIATE SIGNIFICANCE " + "*" * 5
        with open(file_path, 'a') as file:
            file.write(sign_title + "\n")
    
        print(sign_title)

        with open(file_path, 'a') as file:
            file.write("p-value: " + str(estimate.test_stat_significance()["p_value"][0]) + "\n")

        print("p-value: " + str(estimate.test_stat_significance()["p_value"][0]))
    
    except Exception as e:
        print(f"An error occurred while running the {method_name} estimation and refutation: {e}")


def explore_parents(node, graph, df, file_path, visited=None):
    """Explores parent nodes in a graph.

    Args:
        node: The node to start exploration.
        graph: The graph that includes the nodes.
        df: The data frame that contains the data.
        file_path: The path to the file to write exploration results.
        visited: A set to track visited nodes.
    """
    if visited is None:
        visited = set()

    if node in visited:
        return  # BASE CASE 1

    visited.add(node)

    # If the node has no parents, it's a root node
    if not list(graph.predecessors(node)):
        # print(f"{node} is a root node")
        return  # BASE CASE 2

    # Otherwise, explore the parents of the node
    for parent in graph.predecessors(node):
        with open(file_path, 'a') as file:
            divider = "*" * 90
            file.write(f"{divider}\n{parent} ---> {node}\n{divider}\n")

        print(f"{divider}\n{parent} ---> {node}\n{divider}")

        dir_name = "temp"
        # Full path of the directory
        directory_path = os.path.join(os.getcwd(), dir_name)

        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

        nx.drawing.nx_agraph.write_dot(graph, "temp/graph.dot")

        model = CausalModel(
            data=df,
            treatment=[parent],
            outcome=[node],
            graph="temp/graph.dot"
        )
        
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=False)

        if not identified_estimand.no_directed_path:
            if identified_estimand.estimands["backdoor"] is not None:
                estimate_and_refute(
                    model, 
                    identified_estimand, 
                    "backdoor.linear_regression", 
                    file_path
                )

            if identified_estimand.estimands["iv"] is not None:
                estimate_and_refute(
                    model, 
                    identified_estimand, 
                    "iv.linear_regression", 
                    file_path
                )

            if identified_estimand.estimands["frontdoor"] is not None:
                estimate_and_refute(
                    model, 
                    identified_estimand, 
                    "frontdoor.linear_regression", 
                    file_path
                )
        else:
            print("Causal relationship does not exist")

        explore_parents(parent, graph, df, file_path, visited)


def explore_parents_graphs(start, result, df, result_dir):
    """Explore parent nodes in each graph from the results.

    Args:
        start: The node to start exploration.
        result: A dictionary where each item is a method name and a Graph.
        df: The data frame that contains the data.
        result_dir: The directory where results will be written.
    """
    for method, graph in result.items():
        if isinstance(graph, nx.Graph):
            try:
                file_path = f"{result_dir}/{method}_causalreport.txt"
                explore_parents(start, graph, df, file_path)
            except Exception as e:
                print(f"Error in explore_parents for method {method}: {e}")
        else:
            print(f"Error: the value for method {method} is not a Graph: {graph}")
