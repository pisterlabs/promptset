import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html, dcc
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import openai
import os
import json
import subprocess
import math
import dash_cytoscape as cyto

def clear_json_file(path):
    with open(path, "w") as f:
        json.dump({}, f)

def clear_text_file(path):
    gc_file = open(path, "w")
    n = gc_file.write('No rules to display.')
    gc_file.close()

def add_asterisks_to_capitalized(text):
    result = []
    prev_char = None
    
    for char, next_char in zip(text, text[1:] + ' '):
        if char.isupper() and (prev_char == ' ' or next_char == ' '):
            result.extend(['*', char, '*'])
        else:
            result.append(char)
        
        prev_char = char
    
    return ''.join(result)

def save_event_to_log(event, value):
    '''Store the new event in the user's event log. 
    Loads in event log, adds a new event id and stores the result in json'''
    log_dict = get_event_log()
    event_numbers = list(log_dict.keys())
    new_event_number = str(int(event_numbers[-1])+1)
    log_dict[new_event_number] = {"Time": "2021-07-01 12:00:00", 
                "User": "user1", "Event": event, 
                "Value": value}
    save_dict_to_json(log_dict, 'components/all_dynamic_files/event_log.json')

def change_aetable_class(sample_id, new_class = 'pos'):
    selected_dataset = get_selected_dataset()
    csv_path = f'data/{selected_dataset}/raw_data/ae_file.csv'

    # load in csv file
    df = pd.read_csv(csv_path)

    # change the class value of the sample_id row to 'pos'
    df.loc[df['id'] == sample_id, 'class'] = new_class

    # save df as csv to csv_path
    df.to_csv(csv_path, index = False)

    

    return None

# save the pos_pred_dict to a json file
def save_dict_to_json(dictObj, output_path):
    with open(output_path, "w") as outfile:
        json.dump(dictObj, outfile)
    return None

def get_event_log():
    '''Loads and return dictionary containing the current event log'''
    with open('components/all_dynamic_files/event_log.json') as json_file:
        log_dict = json.load(json_file)
    return log_dict
    

def get_selected_dataset():

    # determine which dataset we should be using by loading in relevant meta_data
    with open('data/meta_data/working_data.json') as f:
        meta_data = json.load(f)
    # Extract information required to run induction on the selected data
    return str(meta_data['trial_number'])


def assert_example_positive(sample_id):
    '''Adds the selected example to positive file'''

    # Check the current dataset
    selected_dataset = get_selected_dataset()

    # read the positive examples file
    pos_file_path = f'data/{selected_dataset}/pred_pos/{selected_dataset}.f'
    unknown_file_path = f'data/{selected_dataset}/pred_pos/{selected_dataset}_unknown.pl'
    neg_file_path = f'data/{selected_dataset}/pred_pos/{selected_dataset}.n'

    # add the sample_id to the positive file
    with open(pos_file_path, 'a') as f:
        f.write(f'eastbound({sample_id}).\n')
    
    # remove row from the negative file if it contains the sample_id
    with open(neg_file_path, 'r') as f:
        lines = f.readlines()
    with open(neg_file_path, 'w') as f:
        for line in lines:
            if sample_id not in line:
                f.write(line)
    
    # remove row from the unknown file if it contains the sample_id
    with open(unknown_file_path, 'r') as f:
        lines = f.readlines()
    with open(unknown_file_path, 'w') as f:
        for line in lines:
            if sample_id not in line:
                f.write(line)
    
    return None

def assert_example_negative(sample_id):
    '''Adds the selected example to positive file'''

    # Check the current dataset
    selected_dataset = get_selected_dataset()

    # read the positive examples file
    pos_file_path = f'data/{selected_dataset}/pred_pos/{selected_dataset}.f'
    unknown_file_path = f'data/{selected_dataset}/pred_pos/{selected_dataset}_unknown.pl'
    neg_file_path = f'data/{selected_dataset}/pred_pos/{selected_dataset}.n'

    # add the sample_id to the positive file
    with open(neg_file_path, 'a') as f:
        f.write(f'eastbound({sample_id}).\n')
    
    # remove row from the positive file if it contains the sample_id
    with open(pos_file_path, 'r') as f:
        lines = f.readlines()
    with open(pos_file_path, 'w') as f:
        for line in lines:
            if sample_id not in line:
                f.write(line)
    
    # remove row from the unknown file if it contains the sample_id
    with open(unknown_file_path, 'r') as f:
        lines = f.readlines()
    with open(unknown_file_path, 'w') as f:
        for line in lines:
            if sample_id not in line:
                f.write(line)
    
    return None

def remove_car_at_beginning(input_string):
    if input_string.startswith("car"):
        return input_string[3:]  # Skip the first 3 characters ("car")
    else:
        return input_string


def write_to_prolog_file(inference_file_path, lst, dataset):
    '''Writes a list of strings to a prolog file'''
    # kb_path = f"data/{dataset}/pred_pos/{dataset}.b"
    kb_path = f"data/train_trials/rule{dataset}/train{dataset}.b"
    with open(inference_file_path, 'w') as f:
        f.write(f":-consult('{kb_path}').\n")
        for item in lst:
            item = item.replace(".'", "") # remove all '. where is occurs
            item = item.replace("'", "") # remove all ' and " from the string item
            f.write("%s.\n" % item)
    return None

def gpt_translate(hypothesis):
    # openai.api_key = "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"
    openai.api_key = "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"

    # Set up the model and prompt
    model_engine = "text-davinci-003"
    nl_rules = []
    for rule in hypothesis:
        prompt = f"Translate this from first order logic into natural language: \
                {rule}"

        # Generate a response
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        response = completion.choices[0].text
        response = response.replace('\n\n','')
        nl_rules.append(response)

    return nl_rules

def parameter_dropdown(d_id, placeholder, options):
    submit_button = dbc.Button("+", id = 'fake_induce', color = "primary", className='me-2', n_clicks=0)

    return html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(
                        options,
                        placeholder = placeholder,
                        id = d_id,
                        style={
                            'height':'37px',
                            'width': '412px',
                            'borderWidth': '1px',
                            'borderRadius': '1px',
                            'textAlign': 'center',
                            'margin-bottom':'2px'
                        }   
                    )
                ], 
                style={'display': 'inline-block'},
            ),
            html.Div([submit_button], style={'display': 'inline-block', 'margin-left':'1rem'}),
            dcc.Store(id='predicate-store-1', data=[], storage_type='memory')
        ],
        style={"display": "flex", "margin-bottom":"1rem"}
    )

def fetch_mutag_arrays(exampleNum):

    molecule_df = pd.read_csv('data/mutag_plus/raw_data/molecule.csv')
    atom_df = pd.read_csv('data/mutag_plus/raw_data/atom.csv')
    bond_df = pd.read_csv('data/mutag_plus/raw_data/bond.csv')

    mol_array = molecule_df[molecule_df['molecule_id']==f'd{exampleNum}'].values.tolist()
    bond_df['molecule_id'] = [id.split('_')[0] for id in bond_df['atom1_id']]
    bond_array = bond_df[bond_df['molecule_id']==f'd{exampleNum}'].values.tolist()
    atom_array = atom_df[atom_df['molecule_id']==f'd{exampleNum}'].values.tolist()

    return mol_array, bond_array, atom_array

# Create placeholder rule coverage figure
def generate_coverageGraph(data_path="coverageData.csv"):
    '''Creates placeholder rule coverage figure'''
    df = pd.read_csv(data_path)
    fig = px.treemap(df, path=['label', 'rule_ID'], 
                    values='examples_covered', color='label')
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)')
    fig.update(layout_coloraxis_showscale=False)
    return fig

def parse_upload(contents, filename, date):
    '''
    Parses the files uploaded by the user and converts to a pandas dataframe. 
    Then Saves as csv
    '''
    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            # save to pandas dataframe
            df.to_csv('uploaded_data.csv')

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    
    return html.Div([
        html.H5(f'Successfully uploaded: {filename}')
    ])


# Function for cleaning hypothesis list
def str2lst(hypothesis_text):

    '''Cleans hypothesis list'''
    hyp_list = hypothesis_text[1:-2].split(".',")
    
    result = []
    for idx, rule in enumerate(hyp_list):
        result.append(f'R{idx}: {rule}')
        result.append(html.P(html.Br()))

    return result

# Function for cleaning hypothesis list
def gpt_str2lst(hypothesis_text):

    '''Cleans hypothesis list'''
    hyp_list = hypothesis_text[1:-2].split(".',")
    
    result = []
    for idx, rule in enumerate(hyp_list):
        # result.append(f'R{idx}: {rule}') # uncomment this line to add rule number
        # add full stop to end of rule if there is not one already
        if rule[-1] != '.':
            rule = rule + '.'
            
        print(rule)
        result.append(rule)

    return result

def generate_comparison_graph(plot_type = 1):

    if plot_type == 1:

        data1 = [0, 5, 10, 99]
        labels1 = ['None', 'R2: active(A) :- atm(A,B,c,29,C), bond(A,D,E,1),\n bond(A,F,D,2).',
            'R1: active(A) :- bond(A,B,C,2), bond(A,C,D,1),\n ring_size_5(A,E)', 'R0: active(A) :- ind1(A,1.0)']

        # Data for the second chart
        data2 = [10, 19, 29, 36]
        labels2 = ['R3: active(A) :- atm(A,B,c,14,C), bond(A,D,E,1),\n bond(A,F,D,2).', 'R2: active(A) :- atm(A,B,c,29,C), bond(A,D,E,1),\n bond(A,F,D,2).', 'R1: active(A) :- bond(A,B,C,2), bond(A,C,D,1),\n ring_size_5(A,E)', 
            'R0: active(A) :- atm(A,B,n,32,C), atm(A,D,o,40,C).']
        fig_name = 'v0v1'

    else:
        data1 = [0, 0, 0, 0, 0, 5, 10, 99]
        labels1 = ['None', 'None ', ' None', 'None.', '.None', 'R2: active(A) :- atm(A,B,c,29,C), bond(A,D,E,1),\n bond(A,F,D,2).',
            'R1: active(A) :- bond(A,B,C,2), bond(A,C,D,1),\n ring_size_5(A,E)', 'R0: active(A) :- ind1(A,1.0)']
        # Data for the second chart
        data2 = [7,9, 9,16, 19, 26, 29, 42]
        labels2 = ['R6: active(A) :- atm(A,B,c,29,C), bond(A,D,E,1), \nbond(A,F,D,2).', 'R5: active(A) :- ball3(A,B).', 'R5: active(A) :- carbon_5_aromatic_ring(A,B).', 'R4: active(A) :- atm(A,B,n,32,C), atm(A,D,o,40,C).', 
                'R3: active(A) :- atm(A,B,c,29,C), bond(A,D,E,1), \nbond(A,F,D,2).', 'R2: active(A) :- atm(A,B,c,29,C), bond(A,D,E,1),\n bond(A,F,D,2).', 'R1: active(A) :- bond(A,B,C,2), bond(A,C,D,1),\n ring_size_5(A,E)', 
                'R0: active(A) :- atm(A,B,n,32,C), atm(A,D,o,40,C).']
        fig_name = 'v0v2'


    # color_dict = {'overlap': '#1f77b4', 'disagree':'#FF4136'}
    color_dict = {'overlap': 'c', 'disagree':'m'}

    color = [] 
    for label1, label2 in list(zip(labels1, labels2)):
        if label1 == label2:
            color.append(color_dict['overlap'])
        else:
            color.append(color_dict['disagree'])
    
    col_labels = list(color_dict.keys())
    handles = [plt.Rectangle((0,0),1,1, color=color_dict[label]) for label in col_labels]
    

    # Plotting the first chart
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))#, gridspec_kw={'width_ratios': [1, 1]})
    ax[0].barh(labels1, data1, color= color)
    
    ax[0].set_title(fig_name[0:2])
    ax[0].invert_xaxis()
    ax[0].yaxis.tick_left()
    ax[0].set_xlabel("Rule Coverage", labelpad=20)
    ax[0].xaxis.set_label_coords(1.0, -0.1)

    #Reduce the distance between bars on the y axis
    ax[0].set_yticks(np.arange(len(labels1)))
    ax[0].set_yticklabels(labels1, fontdict={'fontsize': 8})

    # Plotting the second chart
    ax[1].barh(labels2, data2, color= color)#['#FF4136', '#1f77b4', '#1f77b4', '#FF4136'])
    ax[1].set_title(fig_name[2:])
    ax[1].yaxis.tick_right()

    # Reduce the distance between bars on the y axis
    ax[1].set_yticks(np.arange(len(labels2)))
    ax[1].set_yticklabels(labels2, fontdict={'fontsize': 8})

    # Remove the gap between the two charts
    plt.subplots_adjust(wspace=0)
    plt.xlim(0,110)
    plt.legend(handles, col_labels)

    # fig.savefig(f'model_comparison_{fig_name}.png', bbox_inches='tight')
    return fig


def get_att_node_positions(center_x, center_y, num_nodes, radius = 100):
    positions = []
    for i in range(num_nodes):
        angle = (2 * math.pi * i) / num_nodes

        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)

        positions.append({'x': x, 'y': y})
    
    return positions

def generate_train_cytoscape(train_data):
    """RETURNS: cytoscape with number of cars in train_data
    INPUT: dictionary with cars as keys and attributes,x and y positions as values"""

    # Identify cars in train_data
    cars = list(train_data.keys())

    # Generate nodes and their positions
    nodes = []
    for car_num, car in enumerate(cars):
        att_positions = get_att_node_positions(train_data[car]['x'], train_data[car]['y'], num_nodes = len(list(train_data[car]['attributes'].keys())))
        nodes.append({'data': {'id': car, 'label': car}, 'position': {'x': train_data[car]['x'], 'y':  train_data[car]['y']}, 'style': {'background-color': 'blue', 'label': 'data(label)'}})

        for att_num, att in enumerate(list(train_data[car]['attributes'].keys())):
            nodes.append({'data': {'id': att+f'_{str(car_num+1)}', 'label': train_data[car]['attributes'][att]}, 'position': att_positions[att_num], 'style': {'background-color': 'green', 'label': 'data(label)'}})

    # Define how the nodes connect to eachother
    edges = []
    att_edges = []

    # Define the edges connecting the individual cars
    main_edges = [{'data': {'id': f'edge_car_{i}', 'source': f'car_{i}', 'target': f'car_{i+1}', 'label': 'in_front'}, 'style': {'label': 'data(label)'}} for i in range(1,len(cars))]

    # For each car in the train, identify its attributes, and add each attribue as an edge connected to the relevant car
    for car_num, car in enumerate(cars):
        attributes = list(train_data[car]['attributes'].keys())
        for att_num, att in enumerate(attributes):
            att_edges.append({'data': {'id': f'edge_{att}_{str(car_num+1)}', 'source': car, 'target': att + f'_{str(car_num+1)}', 'label': att}, 'style': {'label': 'data(label)'}})

    elements = nodes + main_edges + att_edges

    return cyto.Cytoscape(
            id='cytoscape',
            elements=elements,
            layout={'name': 'preset'},
            style={'width': '600px', 'height': '200px'},
            stylesheet=[
                {
                    'selector': 'edge',
                    'style': {
                        'label': 'data(label)',
                        'line-color': 'gray',
                        'target-arrow-color': 'gray'
                    }
                },
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(label)',
                        'background-color': 'gray',
                        'width': '40px',
                        'height': '40px',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '12px'
                    }
                }
            ]
        )


def generate_train_img(direction, img_number, trial_number = '3', data_type = 'test'):
    
    img_path = f"/assets/images/train_trials/rule{trial_number}/{data_type}/{direction}/train{trial_number}_{img_number}.gif"

    return html.Div([
                    html.Img(src=img_path)                
                    ],
                    style={'width': '50%', 'margin-top':'2rem'},
                )

def generate_component_img(component):
    
    img_path = f"/assets/images/component_images/{component}"

    return html.Div([
                    html.Img(src=img_path)                
                    ],
                    style={'width': '50%', 'margin-top':'2rem'},
                )


def save_model_state():
    pass