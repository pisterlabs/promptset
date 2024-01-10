import json
import re
import openai
from graphviz import Digraph
import base64

import streamlit as st

import func_prompt
from globals import (
    OPENAI_MODELS_CHAT,
    DEFAULT_MODEL_CONFIG, LANG_MODEL_PRICING
)
from common import SafeFormatter

from app_state import (state, _set_state_cb, init_app_state, reset_app_state)
init_app_state() # ensure all state variables are initialized

# GRAPH GENERATOR -------------------------------------------------------------

def correct_json(response_data):
    """
    Corrects the JSON response from OpenAI to be valid JSON
    """
    # clean up the response data JSON
    response_data = response_data.replace('  ',' ').replace(',\n }','\n }')
    # For good measure
    response_data = re.sub(
        r',\s*}', '}', re.sub(
        r',\s*]', ']', re.sub(
        r'(\w+)\s*:', r'"\1":', 
        response_data
    )))
    return response_data

@st.cache_data(ttl=60*60, show_spinner=False)
def get_llm_graph_data_response(user_input, model_name=DEFAULT_MODEL_CONFIG['chat_model']):
    if not user_input:
        return None
    print(f"OpenAI call ({model_name})")
    try:
        model_config = {
            'model': model_name,
            'temperature': state.temperature,
            'top_p': state.top_p,
            'max_tokens': state.max_tokens,
        }
        completion = openai.chat.completions.create(
            messages=json.loads(SafeFormatter().format(json.dumps(func_prompt.MESSAGES), user_input=user_input)),
            functions=func_prompt.FUNCTIONS,
            function_call=func_prompt.FUNCTION_CALL,
            **model_config
        )
    except openai.RateLimitError as e:
        # request limit exceeded or something.
        return str(e)
    except Exception as e:
        # general exception handling
        return str(e)
    
    response_data = completion.choices[0].message.function_call.arguments
    # clean up the response data JSON
    response_data = correct_json(response_data)
    # print(response_data)

    estimated_cost = ((completion.usage.prompt_tokens / 1000.0) * LANG_MODEL_PRICING[state.chat_model]['input']) + \
        ((completion.usage.completion_tokens / 1000.0) * LANG_MODEL_PRICING[state.chat_model]['output'])
    print('Knowledge Graph Generation Estimated Cost: $', estimated_cost)
    state.estimated_cost_graph = estimated_cost
    state.cumulative_cost += estimated_cost
    
    return response_data

# Function to generate a graph image using Graphviz
def generate_knowledge_graph(response_data):
    dot = Digraph(comment="Knowledge Graph")
    response_dict = json.loads(response_data)

    # Add nodes to the graph
    for node in response_dict.get("nodes", []):
        dot.node(node["id"], f"{node['label']} ({node['type']})")

    # Add edges to the graph
    for edge in response_dict.get("edges", []):
        dot.edge(edge["from"], edge["to"], label=edge["relationship"])

    # Requires GraphViz executable, so we can't use it in Streamlit Cloud
    if json.loads(st.secrets['IS_CLOUD_DEPLOYMENT']): 
        return {'dot': dot, 'png': None, 'gv': None}
    else:
        # Render and visualize
        dot.render("./static/knowledge_graph.gv", view=False)
        # Render to PNG format and save it
        dot.render("./static/knowledge_graph", format = "png", view=False)
        return {'dot': dot, 'png': "./static/knowledge_graph.png", 'gv': "./static/knowledge_graph.gv"}

def get_graph_data(response_data):
    try:
        response_dict = json.loads(response_data)
        # Assume response_data is global or passed appropriately
        nodes = [
            {
                "data": {
                    "id": node["id"],
                    "label": node["label"],
                    "color": node.get("color", "defaultColor"),
                }
            }
            for node in response_dict["nodes"]
        ]
        edges = [
            {
                "data": {
                    "source": edge["from"],
                    "target": edge["to"],
                    "label": edge["relationship"],
                    "color": edge.get("color", "defaultColor"),
                }
            }
            for edge in response_dict["edges"]
        ]
        return {"elements": {"nodes": nodes, "edges": edges}}
    except:
        return {"elements": {"nodes": [], "edges": []}}

# UTILITY ---------------------------------------------------------------------

def image_html_fragments(image, text, image_style=None, text_style=None):
    with open(image, 'rb') as img_f:
        img_b64 = base64.b64encode(img_f.read()).decode('utf-8')

    img_style = image_style if image_style else "height: 200px; margin: 3px;"
    image_tag_html = f'<img src="data:image/png;base64,{img_b64}" style="{img_style} vertical-align:middle;">'
    image_download_link = f'<a download="knowledge_graph.png" href="data:image/png;base64,{img_b64}">Download</a>'
    
    # style copied from dev tools
    span_style = text_style if text_style else "font-weight: 600; font-size: 1.75rem;"
    span_style = ( f'font-family: Source Sans Pro, sans-serif; {span_style}'
                   'color: rgb(49, 51, 63); letter-spacing: -0.005em;'
                   'padding: 0.5rem 0px 1rem; margin: 0px; line-height: 1.2;'
                   'text-size-adjust: 100%; -webkit-font-smoothing: auto;'
                   'position: relative; vertical-align:middle;' )
    text_html = f'<span style="{span_style}">{text}</span>'

    image_html = f'{text_html}&nbsp;&nbsp;{image_tag_html}'

    return {'image_html': image_html, 'image_tag_html': image_tag_html, 'image_download_link': image_download_link}

# MAIN ------------------------------------------------------------------------

def main(title, user_input_confirmed=False, response=None):
    # Sidebar
    with st.sidebar:
        st.markdown(f'#### {title} Settings')
        st.selectbox(
            'OpenAI model', options=OPENAI_MODELS_CHAT,
            on_change=_set_state_cb, kwargs={'chat_model': 'selectbox_graph_chat_model_name'},
            index=OPENAI_MODELS_CHAT.index(state.chat_model),
            help='Allowed models. Accuracy, speed, token consumption and costs will vary.',
            key='selectbox_graph_chat_model_name'
        )

    # GPT chat models can handle web sites, so we can keep URLs in the user input
    user_input = state.user_input if state.user_input.strip().startswith('http') else response
    user_input = user_input.replace('\n', ' ').replace('\r', '') if user_input else user_input

    if user_input_confirmed and user_input:
        with st.spinner("Generating knowledge graph (this takes a while)..."):
            response_data = get_llm_graph_data_response(user_input, model_name=state.chat_model)

    if user_input:
        st.subheader('💡 Answer Knowledge Graph')
        # This will use cached response!
        with st.spinner("Generating knowledge graph (this takes a while)..."):
            response_data = get_llm_graph_data_response(user_input, model_name=state.chat_model)

        c1, c2, _ = st.columns([2, 1, 3])
        with c1:
            radio_options = ["Interactive", "Static", "Data"]
            radio_option = st.radio('Knowledge graph options', options=radio_options, horizontal=True)
        with c2:
            height = st.slider("Adjust image height", 100, 1000, 750, 50)
        
        if radio_option == radio_options[0]:
            from graph_frontend import graph_component

            # NOTE: This component doesn't actually return any data, so handle_event is a no-op
            def run_component(props):
                value = graph_component(key='graph', **props)
                return value
            def handle_event(value):
                if value is not None:
                    st.write('Received from graph component: ', value)

            props = {
                'data': { 'graph': get_graph_data(response_data) },
                'graph_height': height,
                'show_graph_data': False,
            }
            handle_event(run_component(props))

        if radio_option == radio_options[1]:
            graph_data = generate_knowledge_graph(response_data)
            # If graphviz executable is available, then we'll have a PNG to download or display
            if graph_data['png']:
                image_html_frags = image_html_fragments(
                    graph_data['png'], '',
                    image_style=f"height: {height}px; margin: 5px;",
                    text_style="font-weight: 600; font-size: 1.75rem;"
                )
                st.markdown(f"{image_html_frags['image_download_link']}", unsafe_allow_html=True)
                # st.markdown(f"{image_html_frags['image_tag_html']}", unsafe_allow_html=True)
                # st.markdown(f"{image_html_frags['image_html']}", unsafe_allow_html=True)
            
            # Display using Streamlit's D3.js graphviz renderer
            st.graphviz_chart(graph_data['dot'])
            
        if radio_option == radio_options[2]:
            st.json(get_graph_data(response_data), expanded=True)
