import os
import openai
from PIL import Image
import streamlit as st
openai.api_key = st.secrets["OPENAI_API_KEY"]
from llama_index.composability import ComposableGraph
from llama_index import LLMPredictor, PromptHelper, ServiceContext
from llama_index.logger import LlamaLogger
from llama_index import GPTSimpleVectorIndex
from langchain import OpenAI
import openai
from langchain.chat_models import ChatOpenAI
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from langchain.chat_models import ChatOpenAI

# set page config
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# title
def write_h1(text:str):
    return st.markdown(f"<h1 style='text-align: center; color: grey;'>{text}</h1>", unsafe_allow_html=True)
write_h1("RegGPT Demo")

# Initialization of a bunnch of session state variables
if 'token_tracker' not in st.session_state:
    st.session_state['token_tracker'] = 0
    # Initialization
if 'query_embedding_tracker' not in st.session_state:
    st.session_state['query_embedding_tracker'] = 0
    # Initialization
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0
    # Initialization
if 'conversation_log' not in st.session_state:
    st.session_state['conversation_log'] = """
                                            <h1>Chatbot log</h1>
                                            <p><strong>Date:</strong> {date}</p>
                                            <p><strong>Total tokens used (approx.):</strong> {tokens}</p>
                                            <p><strong>Total session cost (USD) (approx.):</strong> {cost}</p>

                                           """
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'sources' not in st.session_state:
    st.session_state['sources'] = []




## Set up the app UI
image = Image.open('deloitte-logo-white.png')
st.sidebar.image(image)
st.sidebar.title("RegGPT V0.0.2")
st.sidebar.divider()
st.sidebar.subheader("Search configuration")

# Add first slider
n_doc = st.sidebar.slider("Number of documents to search", min_value=1, max_value=13, value=1, step=1)
# Add second slider
n_chunk = st.sidebar.slider("Number of text chunks to search per document",min_value=1, max_value=10, value=1, step=1)
st.sidebar.divider()
#addtional prompt eng features
st.sidebar.subheader("Prompt Engineering methods")
if st.sidebar.checkbox('Query Augmentation - Rephrases your query to better convey the intention'):
    input_aug = True
else:
    input_aug = False
    
if st.sidebar.checkbox('Query Decompose - Breaks down complex queries into individual sub-queries. Particularly useful for "compare/contrast" type queries'):
    query_decompose = True
else:
    query_decompose = False
    
st.sidebar.divider()

# define the endsession function to call when the you are done and want to download the log. It doesnt actually end the session though
def end_session():
    from datetime import datetime   
    date = datetime.today().strftime("%d %b %Y")
    # the log is essentially an html string
    final_log = st.session_state['conversation_log'].format(date = date,tokens = st.session_state['token_tracker'], cost = round(st.session_state['total_cost'], 5))
    # save the log as .html. Cannot display on streamlit due to some bug.
    with open("/app/nlp-vis/logs/Log_Output.html", "w") as file:
        file.write(final_log)
        st.success("Logs downloaded to the 'logs' folder in the app directory")

    
# Add button to end session
if st.sidebar.button("End session and download️ log"):
    end_session()




## the main ask AI function (apologies in advance!)
# takes the slider values as inputs as well as the checkbox flags
def ask_ai(query, n_doc, n_chunk, input_aug, query_decompose):
    with st.spinner('Searching database...'):

        # we will be using chatGPT as the base llm because of cost considerations

        # max input size - max number of input prompt token (includes the query + context chunk + output)
        max_in = 4096

        # max output token - max number of tokens the chatbot will return as response
        num_out = 150

        # temp - crrativity vs reproducability. Setting as 0 to reduce hallucination
        temp = 0
 

        # note max_in + max_out = context length of the model. FOr davincii it is 4097
        # note chunk_size + our wuery  =  max_in
        # model name - using gpt-3.5-turbo as it is optimised for dialogue
        model_name = "text-davinci-003"

        # llm predictor
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=temp, model_name=model_name))

        # prompt helper
        prompt_helper = PromptHelper(max_input_size=max_in, num_output=num_out, max_chunk_overlap=20)

        # helps with the conversation log
        llama_logger = LlamaLogger()

        # service context is the main interface for all the above objects
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper,
                                                       llama_logger=llama_logger)

        # load our graph structure
        graph = ComposableGraph.load_from_disk('regGPT_vectorstore.json', service_context=service_context)

        # query configuration block
        # setting an id for the grpah structure. Otherwise breaks the query config.
        # get root index
        root_index = graph.get_index(graph.index_struct.root_id, GPTSimpleVectorIndex)
        # set id of root index
        root_index.index_struct.index_id = "graph"

        #################### PLAYGROUND ##########################
        # The query config is at the heart of the chatbot back end. This is where users can define how many chunks to
        # refer to (in a way defining the search space), how to combine these chunks and use them effectively to get
        # the best response. THis is made possible by differern "response synthesis" techniques offered by llama-index.
        # This is also where we can incorporate more advanced response refining techniques like "step_decompose" and
        # feed in custom prompt templates (eg. few short learning refinement)
        #
        # useful info - https://gpt-index.readthedocs.io/en/latest/guides/primer/usage_pattern.html#setting-response-mode
        # step_decompose - https://gpt-index.readthedocs.io/en/latest/how_to/query/query_transformations.html

        from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
        from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
        decompose_transform = DecomposeQueryTransform(
            llm_predictor, verbose=True)

        # number of graph nodes to retrieve during query. USer defined. Take into account the cost, performance and
        # speed tradeoff.
        num_graph_nodes_to_retreive = n_doc

        # number of sub indices within a graph node to retrive during query
        num_sub_indices_to_retreive = n_chunk

        if query_decompose == True:
            # the query config allows us to define the strategy for querying
            query_configs = [
                # this config is for the graph structure
                {"index_struct_id": "graph",
                 "index_struct_type": "simple_dict",
                 "query_mode": "default",
                 "query_kwargs": {
                     "response_mode": "compact",
                     "similarity_top_k": num_graph_nodes_to_retreive,

                 },

                 },

                # this config is for the sub indices
                {

                    "index_struct_type": "simple_dict",
                    "query_mode": "default",
                    "query_kwargs": {
                        "similarity_top_k": num_sub_indices_to_retreive,

                    },
                    # NOTE: set query transform for subindices
                    "query_transform": decompose_transform

                },
            ]

        else:
            # the query config allows us to define the strategy for querying
            query_configs = [
                # this config is for the graph structure
                {"index_struct_id": "graph",
                 "index_struct_type": "simple_dict",
                 "query_mode": "default",
                 "query_kwargs": {
                     "response_mode": "compact",
                     "similarity_top_k": num_graph_nodes_to_retreive,

                 },

                 },

                # this config is for the sub indices
                {

                    "index_struct_type": "simple_dict",
                    "query_mode": "default",
                    "query_kwargs": {
                        "similarity_top_k": num_sub_indices_to_retreive,

                    }

                },
            ]

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #################### PLAYGROUND ##########################
        if input_aug == True:
            prompt_template = """
    
                Question: {user_query} 
                \n
                We have the opportunity to rephrase the above question so that the resulting refined question is more 
                easily understandable by a large language model. 
                Based on this could you try to rephrase and refine the above question and return the new question? 
                If this is not possible return the original question.
                \n
                Return the new question in the following format:-
                "<the new question>"
    
    
                """
            formatted_query = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{'role': 'user', 'content': prompt_template.format(user_query=query)}],
                temperature=0
            )

            st.write("Your query has been formated as: " + formatted_query.choices[-1].message.content)
            final_query = formatted_query.choices[-1].message.content
            # obtain token usage and cost incurred in this call via dropdown
            # cost for formatting
            tokens_used_formatting = formatted_query.usage.total_tokens
            cost_for_formatting = tokens_used_formatting * (0.002 / 1000)  # price of chatgpt
        else:
            final_query = query
            tokens_used_formatting = 0
            cost_for_formatting = 0

        response = graph.query(final_query, query_configs=query_configs)
        chatbot_response = response.response

        # obtain source chunks in a dropdown (As explanations) in streamlit
        # response object contains the source nodes + refining nodes that the chatbot uses to store intermediate
        # answers. We only need the source nodes
        node_objs = response.source_nodes[::-1]
        sources = []
        for i in range((num_graph_nodes_to_retreive * num_sub_indices_to_retreive)):
            # store the docuemnt source name, similarity of the chunk/node to the query, and the text content
            sources.append([node_objs[i].node.extra_info, node_objs[i].score, node_objs[i].node.text])
    #     print(sources)

        # cost for embedding the final query
        import tiktoken
        encoding = tiktoken.get_encoding("p50k_base")
        query_tokens_used = len(encoding.encode(final_query))
        
        st.session_state['query_embedding_tracker'] += query_tokens_used
        cost_for_embedding_query = query_tokens_used * (0.0004 / 1000)  # price of text ada model

        # total tokens used in this conversation. This is stored in the llm_predictor class
        # finding the exact tokens per calls in a granualar fashion is a limitation in llama-index
        total_tokens_used = llm_predictor.total_tokens_used
       
        st.session_state['token_tracker'] += total_tokens_used+tokens_used_formatting
        cost_for_conv = total_tokens_used * (0.02 / 1000)

        # total cost for conv
        total_cost_for_conv = cost_for_embedding_query + cost_for_conv + cost_for_formatting
       
        st.session_state['total_cost'] += total_cost_for_conv

        # conversation Log function
        def generate_html():

            # the service_context.llama_logger.get_logs() function gives all the log details in an unstructured way.
            # The current implementation is only accurate for the "default response mode" in the query config and gives
            # inaccurate logs for other response synthesis modes.
            # there is a way to debug on jupyter NB see: https://gpt-index.readthedocs.io/en/latest/getting_started/starter_example.html
            # but has not been explored in this NB as of yet.
            logs = service_context.llama_logger.get_logs()
            # you can print logs for debugging
           
            conversation = []
            for i in logs:
                keys = list(i.keys())
                if 'formatted_prompt_template' in keys:
                    conversation.append(i['formatted_prompt_template'])
                elif 'initial_response' in keys:
                    conversation.append(i['initial_response'])

                elif 'refined_response' in keys:
                    conversation.append(i['refined_response'])

            i = 0
            sect_html = """"""
            while i < len(conversation):
                sect_html += """
                    <p style="text-align: left;">&nbsp;</p>
                    <p style="text-align: left;"><strong>LLM input:&nbsp;</strong>{LLM_input}</p>
                    <p style="text-align: left;"><strong>LLM output:&nbsp;</strong>{LLM_output}</p>
    
                    """.format(LLM_input=conversation[i], LLM_output=conversation[i + 1])
                i += 2
                
            if input_aug == True:



                html_template = """
                    <p style="text-align: left;"><strong>---------------------------------------------------------------------------------------------</strong></p>
                    <h2 style="text-align: left;"><strong>Conversation ID - ###</strong></h2>
                    <p style="text-align: left;"><strong>Original query:&nbsp;</strong>{user_input}</p>
                    <p style="text-align: left;"><strong>Formatted query powered by LLM:&nbsp;</strong>{formatted_input}</p>
                    <p style="text-align: left;">&nbsp;</p>
                    <p style="text-align: left;"><strong>User input:</strong> {final_input}</p>
                    <p style="text-align: left;"><strong>Chatbot response:&nbsp;</strong>{chatbot_response}</p>
                    <p style="text-align: left;"><strong>Total query and completion tokens used:&nbsp;</strong>{total_tokens}</p>
                    <p style="text-align: left;"><strong>Total embedding tokens used:&nbsp;</strong>{embed_tokens}</p>
                    <p style="text-align: left;"><strong>Total cost:&nbsp;</strong>{total_cost}</p>
                    <p style="text-align: left;">&nbsp;</p>
                    <p style="text-align: left;"><strong>Internal conversation:</strong></p>
                    <p style="text-align: left;"><strong>LLM input:&nbsp;</strong>{LLM_input}</p>
                    <p style="text-align: left;"><strong>LLM output:&nbsp;</strong>{LLM_output}</p>""" + sect_html + """
                    <p style="text-align: left;">&nbsp;</p>
                    <p style="text-align: left;"><strong>---------------------------------------------------------------------------------------------</strong></p>
                    """
                html_final = html_template.format(user_input=query, formatted_input=formatted_query.choices[-1].message.content,
                                                  final_input=final_query, chatbot_response=chatbot_response,
                                                  total_tokens=total_tokens_used + tokens_used_formatting,
                                                  embed_tokens=query_tokens_used, total_cost=total_cost_for_conv,
                                                  LLM_input=prompt_template.format(user_query=query),
                                                  LLM_output=formatted_query.choices[-1].message.content)



            else:



                html_template = """
                    <p style="text-align: left;"><strong>---------------------------------------------------------------------------------------------</strong></p>
                    <h2 style="text-align: left;"><strong>Conversation ID - ###</strong></h2>
                    <p style="text-align: left;"><strong>User input:</strong> {final_input}</p>
                    <p style="text-align: left;"><strong>Chatbot response:&nbsp;</strong>{chatbot_response}</p>
                    <p style="text-align: left;"><strong>Total query and completion tokens used:&nbsp;</strong>{total_tokens}</p>
                    <p style="text-align: left;"><strong>Total embedding tokens used:&nbsp;</strong>{embed_tokens}</p>
                    <p style="text-align: left;"><strong>Total cost:&nbsp;</strong>{total_cost}</p>
                    <p style="text-align: left;">&nbsp;</p>
                    <p style="text-align: left;"><strong>Internal conversation:</strong></p>
                    """ + sect_html + """
                    <p style="text-align: left;">&nbsp;</p>
                    <p style="text-align: left;"><strong>---------------------------------------------------------------------------------------------</strong></p>
                    """
                html_final = html_template.format(final_input=final_query, chatbot_response=chatbot_response,
                                                  total_tokens=total_tokens_used + tokens_used_formatting,
                                                  embed_tokens=query_tokens_used, total_cost=total_cost_for_conv,
                                                  )
            st.session_state['conversation_log'] += html_final

        generate_html()

        # returns the chatbot response, sources and the total tokens used in this given interaction
        return chatbot_response, sources, total_tokens_used+tokens_used_formatting





# user input and the main action
user_input = st.text_input("Ask a question: ", key="input")

# needs a submit button or the app randomly runs the query which leads to unnecessary API calls
submit_button = st.button("Submit", key="submit_button")
if submit_button:
    output, sources, tokens = ask_ai(user_input, n_doc, n_chunk, input_aug, query_decompose)

    # initailise the source string
    sources_string = "<h2><strong>Total tokens used:&nbsp;{tokens}</strong></h2><p>&nbsp;</p>"
    # for each source in the source list, extract valuable data
    for i in sources:
        string = """<p><strong>Document: </strong>{doc}</p>
                    <p><strong>Cosine Similarity: </strong>{sim}</p>
                    <p><strong>Text: </strong>{text}</p>
                    <p>&nbsp;</p>
                    <p>&nbsp;</p>"""
        sources_string+=string.format(doc = i[0]['Document Name'], sim = i[1], text = i[2])

    # save user input, out put and sources
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    source_expander = st.expander("Source")
    with source_expander:
       st.session_state.sources.append(sources_string.format(tokens = tokens))


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.write(f"User: {st.session_state['past'][i]}")
        st.write(f"RegGPT response: {st.session_state['generated'][i]}")
        source_expander = st.expander("Source")
        with source_expander:
    
            st.markdown(st.session_state['sources'][i], unsafe_allow_html=True)
            
      
        

        
        
