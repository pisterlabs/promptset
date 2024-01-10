import json
import os
from openai_query import openai_chat
from topological_sorting import generate_graph, topological_sort
from model_nodes_edges import load_nodes_edges, get_genes
from file_io import read_system_json, get_root_path, get_model_directory_path, write_system_tsv, read_system_tsv
from gene_feature import summarize_gene_feature, summarized_gene_feature_to_tsv
from pages_io import write_system_page, read_system_page, create_music_2_system_analysis_page
from chatgpt_prompts import create_music_2_chatGPT_prompt_text, create_music_2_chatGPT_prompt_parent,  estimate_tokens
from ontology_modify import find_children

## load the config file 
####remember to change config if you want to use different map/model
with open('config.json') as config_file:
    data = json.load(config_file)

#`MODEL_ANNOTATION_ROOT` is the path to the root directory of the model annotation repository
os.environ['MODEL_ANNOTATION_ROOT'] = data["MODEL_ANNOTATION_ROOT"]

# load the API key
openai.api_key = data["OPENAI_API_KEY"]
temperature = data["TEMP"] # Set your temperature here 
# CH: I want it to be deterministic, so I set temperature to 0
max_tokens = data["MAX_TOKENS"] # Set your max tokens here
rate_per_token = data["RATE_PER_TOKEN"]# Set your rate per token here 
model = data["GPT_MODEL"]
DOLLAR_LIMIT = data["DOLLAR_LIMIT"]  # Set your dollar limit here
logfile_name = data["LOG_NAME"] # Set your log file name here
# set the context for music2
context = data["MUSIC2_CONTEXT"]

# set the model name and version, canbe found in the directory
model_name = data["MAP_NAME"]

version = data["MAP_V"]

file_name = data["MAP_FILE"]

## remember to change the name of the log file 
LOG_FILE = os.path.join(get_model_directory_path(model_name, version), f'{logfile_name}log.json')

## load the model 
nodes, edges = load_nodes_edges(model_name, version, file_name)
# print(nodes.head()), print(edges.head())

# get the root node info
root_node_info = read_system_json(model_name, version, 'root_node', 'my_gene', get_root_path())
## get the sorted nodes 
graph = generate_graph(edges)
sorted_nodes = topological_sort(graph)

huge_token_nodes = []
for system in sorted_nodes:
    # skip the root node
    if system == 'Cluster0-0':
        continue
    genes = get_genes(system, nodes)
    if len(genes) < 120:
        # generate the prompt with gene features
        # TODO: modify the gene feature to remove duplicates #check if the length can shorten 
        print(f"generating prompt for {system}")
        summarized_info = summarize_gene_feature(root_node_info, genes)  # Summarize the gene features
        summarized_tsv = summarized_gene_feature_to_tsv(summarized_info) 
        write_system_tsv(summarized_tsv, model_name, version, system, 'go_summary', get_root_path())
        prompt = create_music_2_chatGPT_prompt_text(system,nodes, summarized_tsv, n_genes=max(2, int(len(genes)/25)))

        est_tokens = estimate_tokens(context + '\n' +prompt)

        print (f"Estimated number of tokens: {est_tokens}")

        write_system_page(prompt, 'txt',model_name, version, system, "chatgpt_prompt", get_root_path()) # write the prompt to text file

        if est_tokens > 5500: # if the estimated tokens is too large, skip it
            huge_token_nodes.append(system)
            continue

        response_path = os.path.join(get_model_directory_path(model_name, version), system, f"{system}_chatgpt_response")

        # run chatgpt if response file does not exist, avoid duplicate runs
        if not os.path.exists(response_path + '.md'):
            print(f"running Chatgpt for {system}")
            # TODO: run chatgpt
            prompt = read_system_page('txt', model_name, version, system, "chatgpt_prompt", get_root_path())
            # print(prompt)
            response_text = openai_chat(context, prompt, model,temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT)
            if response_text:
                # save markdown file
                with open(response_path + '.md', 'w') as f:
                    f.write(response_text)
                #save a html file
                # write_response_page_html(response_path,response_text, system)
                # #save to the analysis page
                if not summarized_tsv:
                    summarized_tsv = read_system_tsv(model_name, version, system, "go_summary", get_root_path())
                    
                analysis_page = create_music_2_system_analysis_page(system, response_text, nodes, summarized_tsv, n_genes=max(2, int(len(genes)/25)))
                write_system_page(analysis_page,'md',model_name, version, system, "analysis_page", get_root_path()) # write an analysis page in markdown
        else:
            print(f"Chatgpt response file exists for {system}, skipping")

    else:
        # check if is parent 
        if system in edges['parent'].values:
            print(f"generating prompt for parent {system}")
            # TODO: generate the prompt with just gene names but in children groups
            prompt = create_music_2_chatGPT_prompt_parent(system, nodes, edges)
            est_tokens = estimate_tokens(context + '\n' +prompt)

            print (f"Estimated number of tokens: {est_tokens}")
            write_system_page(prompt,'txt', model_name, version, system, "chatgpt_prompt", get_root_path()) # write the prompt to text file

            if est_tokens > 5500:
                huge_token_nodes.append(system)
                continue
            
            response_path = os.path.join(get_model_directory_path(model_name, version), system, f"{system}_chatgpt_response")

            # run chatgpt if response file does not exist, avoid duplicate runs
            if not os.path.exists(response_path + '.md'):
                print(f"running Chatgpt for {system}")
                # TODO: run chatgpt
                prompt = read_system_page('txt', model_name, version, system, "chatgpt_prompt", get_root_path())
                # print(prompt)
                response_text = openai_chat(context, prompt, model,temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT)
                if response_text:
                    # save markdown file
                    with open(response_path + '.md', 'w') as f:
                        f.write(response_text)

                    #keep getting gene features 
                    summarized_info = summarize_gene_feature(root_node_info, genes)  # Summarize the gene features
                    summarized_tsv = summarized_gene_feature_to_tsv(summarized_info) 
                    write_system_tsv(summarized_tsv, model_name, version, system, 'go_summary', get_root_path())
                    analysis_page = create_music_2_system_analysis_page(system, response_text, nodes, summarized_tsv, n_genes=max(2, int(len(genes)/25)))
                    write_system_page(analysis_page,'md',model_name, version, system, "analysis_page", get_root_path()) # write an analysis page in markdown format

            else:
                print(f"Chatgpt response file exists for {system}, skipping")        

        #if are children just print the long list of genes 
        elif system not in edges['parent'].values:
            print(f"generating prompt for large children {system}")
            # TODO: generate the prompt with just gene names 
            prompt = create_music_2_chatGPT_prompt_text(system,nodes)
            est_tokens = estimate_tokens(context + '\n' +prompt)
            print (f"Estimated number of tokens: {est_tokens}")
            write_system_page(prompt, 'txt', model_name, version, system, "chatgpt_prompt", get_root_path())

            if est_tokens > 5500: # if the estimated tokens is too large, skip it
                huge_token_nodes.append(system)
                continue
            response_path = os.path.join(get_model_directory_path(model_name, version), system, f"{system}_chatgpt_response")

            # run chatgpt if response file does not exist, avoid duplicate runs
            if not os.path.exists(response_path + '.md'):
                print(f"running Chatgpt for {system}")
                # TODO: run chatgpt
                prompt = read_system_page('txt', model_name, version, system, "chatgpt_prompt", get_root_path())
                # print(prompt)
                response_text = openai_chat(context, prompt, model,temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT)
                if response_text:
                    # save markdown file
                    with open(response_path + '.md', 'w') as f:
                        f.write(response_text)

                    #keep getting gene features 
                    summarized_info = summarize_gene_feature(root_node_info, genes)  # Summarize the gene features
                    summarized_tsv = summarized_gene_feature_to_tsv(summarized_info) 
                    write_system_tsv(summarized_tsv, model_name, version, system, 'go_summary', get_root_path())
                    analysis_page = create_music_2_system_analysis_page(system, response_text, nodes, summarized_tsv, n_genes=max(2, int(len(genes)/25)))
                    write_system_page(analysis_page,'md',model_name, version, system, "analysis_page", get_root_path()) # write an analysis page in markdown format

            else:
                print(f"Chatgpt response file exists for {system}, skipping")   

                continue
            
if huge_token_nodes:
    with open(os.path.join(get_model_directory_path(model_name, version), 'huge_token_nodes_tocheck.txt'), 'w') as f:
        f.write('\n'.join(huge_token_nodes))