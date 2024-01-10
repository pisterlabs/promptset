import calendar
import os
import time
from datetime import datetime
from dateutil import tz
from dotenv import find_dotenv, load_dotenv
from typing import List, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nicegui import Tailwind, Client, app, ui
from nicegui.events import MouseEventArguments

# import functions from API folder for estimator tab
import sys

""" LOAD API KEYS FROM ENV """
#_ = load_dotenv(find_dotenv(filename='tab2_apikeys.txt'))
_ = load_dotenv(find_dotenv())
PVWATTS_API_KEY = os.environ['PVWATTS_API_KEY']
OPENUV_API_KEY = os.environ['OPENUV_API_KEY']
TOMTOM_API_KEY = os.environ['TOMTOM_API_KEY']

#sys.path.insert(0, r'C:/Users/Zhong Xuean/Documents/dsaid-hackathon23-illuminati/api/')
sys.path.insert(0, r'../api/')
from conversions import to_bearing
from demand import get_demand_estimate
from geocode import geocode
from pvwatts import get_solar_estimate
from solarposition import get_optimal_angles, get_suninfo, utc_to_sgt, time_readable

# TODO: modularize hot loading LLM to make this script more readable and lightweight
#########################
# START HOT LOADING LLM #
#########################

from pathlib import Path

import openai
from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_index import (ListIndex, LLMPredictor, ServiceContext,
                         VectorStoreIndex, download_loader)
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import \
    DecomposeQueryTransform
from llama_index.langchain_helpers.agents import (IndexToolConfig,
                                                  LlamaToolkit,
                                                  create_llama_chat_agent)
from llama_index.query_engine.transform_query_engine import \
    TransformQueryEngine

# adding Xuean's node post processor
sys.path.insert(0, '../chatbot/')
sys.path.insert(1,'../gui/assets/')
#sys.path.insert(0, r'.../dsaid-hackathon23-illuminati/chatbot/') # Xuean's edit - original line didn't work on my laptop
from custom_node_processor import CustomSolarPostprocessor

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

# list ema docs
ema = [1,2,3,4,5,6,7,8,9]

UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()
doc_set = {}
all_docs = []

for ema_num in ema:
    ema_docs = loader.load_data(file=Path(f'../chatbot/data/EMA/EMA_{ema_num}.csv'), split_documents=False)
    # insert year metadata into each year
    for d in ema_docs:
        d.extra_info = {"ema_num": ema_num}
    doc_set[ema_num] = ema_docs
    all_docs.extend(ema_docs)

"""
### Setup a Vector Index for each EMA doc in the data file
We setup a separate vector index for each file
We also optionally initialize a "global" index by dumping all files into the vector store.
"""

# initialize simple vector indices + global vector index
# NOTE: don't run this cell if the indices are already loaded!
index_set = {}
service_context = ServiceContext.from_defaults(chunk_size=512)
for ema_num in ema:
    cur_index = VectorStoreIndex.from_documents(doc_set[ema_num], service_context=service_context)
    index_set[ema_num] = cur_index

# Load indices from disk
index_set = {}
for ema_num in ema:
    index_set[ema_num] = cur_index

"""
### Composing a Graph to synthesize answers across all the existing EMA docs.
We want our queries to aggregate/synthesize information across *all* docs. To do this, we define a List index
on top of the 4 vector indices.
"""


index_summaries = [f"These are the official documents from EMA. This is document index {ema_num}." for ema_num in ema]

index_summary_new = []

index_summary_new.append('These are official documents Q&A documents from EMA. Structured as a CSV with the following columns: (Title -- Questions -- Answers) This is document index 1.')

index_summary_new.append('These are official documents Q&A documents from EMA. Structured as a CSV with the following columns: (Title -- Questions -- Answers) This is document index 2.')

index_summary_new.append('This official document from EMA contains all minister speeches on Singapore\s energy policy. Structured as a CSV with the following columns: (Title -- Date -- Content) This is document index 3.')

index_summary_new.append('This official document from EMA summaries of videos in document index 3. Structured as a CSV with the following columns: (Title -- Summary) This is document index 4.')

index_summary_new.append('This official document from EMA contains the official video transcript of policy explainers from EMA. Structured as a CSV with the following columns: (URL Source -- Title -- Content) This is document index 5.')

index_summary_new.append('This official document from EMA contains the contents of EMA\'s Official Solar Handbook. Structured as a CSV with the following columns: (URL Source -- Title -- Content) This is document index 6.')

index_summary_new.append('This official document from EMA contains the Question and Answer component of EMA\'s Official Solar Handbook. Structured as a CSV with the following columns: (URL Source -- Title -- Content) This is document index 7.')

index_summary_new.append('This official document from EMA contains the summaries of EMA\'s Official Solar Handbook. Structured as a CSV with the following columns: (Title -- Summary) This is document index 8.')

index_summary_new.append('This official document from EMA contains the raw content of EMA\'s Official Solar Handbook. Structured as a CSV with the following columns: (Chapter -- Subheader1 -- Subheader2 -- Text) This is document index 9.')


# set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    ListIndex,
    [index_set[ema_num] for ema_num in ema],
    index_summaries=index_summary_new,
    service_context=service_context
)

"""
## Setting up the Chatbot Agent
We use Langchain to define the outer chatbot abstraction. We use LlamaIndex as a core Tool within this abstraction.
"""

# define a decompose transform
decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)

# define custom query engines
custom_query_engines = {}
for index in index_set.values():
    query_engine = index.as_query_engine()
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={'index_summary': index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine
custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    response_mode='tree_summarize',
    verbose=True,
)

# construct query engine
graph_query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
node_postprocessor = CustomSolarPostprocessor(service_context=service_context, top_k_recency = 1, top_k_min = 3)

query_engine_node_postproc = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[node_postprocessor]
)

index_configs = []

for y in range(1, 9):
    query_engine_node_postproc = index.as_query_engine(
        similarity_top_k=3,
        node_postprocessors=[node_postprocessor]
    )
    tool_config = IndexToolConfig(
        query_engine=query_engine,
        name=f"Vector Index {y}",
        description=f"Useful for answering queries regarding Solar Energy, Singapore's National Energy Policies or the installation of Solar PV. {index_summary_new[y]} ",  
        tool_kwargs={"return_direct": True, "return_sources": True},
    )
    index_configs.append(tool_config)

graph_config = IndexToolConfig(
    query_engine=graph_query_engine,
    name=f"Graph Index",
    description="Necessary for when you want to answer queries regarding EMAs energy policy.",
    tool_kwargs={"return_direct": True, "return_sources": True},
    return_sources=True
)

toolkit = LlamaToolkit(
    index_configs=index_configs,
    graph_configs=[graph_config]
)


# initialize agent
memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
)

inj = """
        
        Please respond to the statement above.
        Your name is Jamie Neo. Your pronouns are they/them.
        You are an AI chatbot created to support EMA by answering questions about solar energy in Singapore.
        You will answer only with reference to official documents from EMA.
        Refer to the context FAQs and the EMA documents in composing your answers. Define terms if needed.
        If the user asks for a specific document, you can provide the URL link to the document.
        If the user is unclear, you can ask the user to clarify the question.
        When in doubt and/or the answer is not in the EMA documents, you can say "I am sorry but do not know the answer."
        Keep your answers short and terse. Be polite at all times.
    """


def get_chatbot_respone(text_input):
    return agent_chain.run(input = text_input)

#######################
# END HOT LOADING LLM #
#######################

#############
# START GUI #
#############

messages: List[Tuple[str, str, str, str]] = []
thinking: bool = False

# bot id and avatar
bot_id = str('b15731ba-d28c-4a77-8076-b5750f5296d3')
#bot_avatar = f'https://robohash.org/{bot_id}?bgset=bg2'
bot_avatar = 'https://raw.githubusercontent.com/wonkishtofu/dsaid-hackathon23-illuminati/main/gui/assets/bot.png' # this worked before and now it doesn't?
disclaimer = "Hello there! I am Jamie Neo, an AI chatbot with a sunny disposition 😎 On behalf of the Energy Market Authority (EMA), I'm here to answer your questions about solar energy in Singapore."

# TODO: convert all time stamps on chat messages to SGT
stamp = datetime.utcnow().strftime('%H:%M:%S')#.replace(tzinfo = tz.gettz('UTC'))
#sgt_stamp = stamp.astimezone(tz.gettz('Asia/Singapore'))
messages.append(('Bot', bot_avatar, disclaimer, stamp))

# refresh function for CHATBOT
@ui.refreshable
async def chat_messages(own_id: str) -> None:
    for user_id, avatar, text, stamp in messages:
        ui.chat_message(text = text, stamp = stamp, avatar = avatar, sent = own_id == user_id)
    if thinking:
        # TODO: SHOW UI.SPINNER BEFORE RESPONSE GENERATION
        ui.spinner(size = 'lg', color = 'yellow')
        ui.classes('self-center')
        #ui.spinner(size='3rem').classes('self-center')
    await ui.run_javascript("window.scrollTo(0,document.body.scrollHeight)", respond = False) # autoscroll

@ui.page('/')
async def main(client: Client):
    async def send() -> None:
        stamp = datetime.utcnow().strftime('%H:%M:%S')#.replace(tzinfo = tz.gettz('UTC'))
        #sgt_stamp = stamp.astimezone(tz.gettz('Asia/Singapore'))
        user_input = text.value
        messages.append((user_id, avatar, user_input, stamp))
        chat_messages.refresh()
        thinking = True
        text.value = ''
        # TODO: CLEAR TEXT UPON SEND
        # TODO: DECREASE LAG BETWEEN SEND AND APPEAR IN CONVERSATION
        text.label = ''

        response = get_chatbot_respone(user_input)
        stamp = datetime.utcnow().strftime('%H:%M:%S')#.replace(tzinfo = tz.gettz('UTC'))
        #sgt_stamp = stamp.astimezone(tz.gettz('Asia/Singapore'))
        messages.append(('Bot', bot_avatar, response, stamp))
        thinking = False
        text.set_value(None)

    def on_tab_change(event):
        print(event.value)
        # remove the text and avatar when move to different tab
        # have to do this cos they are in footer
        avatar_ui.set_visibility(event.value == 'CHATBOT')
        text.set_visibility(event.value == 'CHATBOT')
        
    # define the tabs
    with ui.header().classes(replace = 'row items-center') as header:
        with ui.tabs().classes('w-full') as tabs:
            chatbot = ui.tab('CHATBOT')
            estimator = ui.tab('ESTIMATOR')
            realtime = ui.tab('REALTIME')
    
    # set tabs in a tab panel
    with ui.tab_panels(tabs,
                       value=chatbot,
                       on_change=on_tab_change).classes('w-full'):
        
        # what appears in chatbot tab
        with ui.tab_panel(chatbot):
            user_id = str('6af9dba1-022a-41ba-8f5b-34c21d1cc89a')
            avatar = f'https://robohash.org/{user_id}?bgset=bg2'
            
            # TODO: change color of speech bubble for something less jarring
            with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
                with ui.row().classes('w-full no-wrap items-center'):
                    avatar_ui = ui.avatar().on('click', lambda: ui.open(main))
                    with avatar_ui:
                        ui.image(avatar)
                    text = ui.input(placeholder = 'type your query here').on('keydown.enter', send) \
                        .props('rounded outlined input-class=mx-6').classes('flex-grow')
                    text.props('clearable') # button to clear type text
            
            await client.connected()  # chat_messages(...) uses run_javascript which is only possible after connecting

            with ui.column().classes('w-full max-w-2xl mx-auto items-stretch'):
                await chat_messages(user_id)
            
        # what appears in estimator tab
        with ui.tab_panel(estimator):
            # with ui.column().classes('w-full items-center'):
            await ui.run_javascript("window.scrollTo(0,document.body.scrollHeight)", respond = False) # autoscroll
            with ui.stepper().props('vertical').classes('w-full') as stepper:
                with ui.step('Generation'):
                    # 1. enter address
                    ADDRESS = ui.input(label = 'Enter an address or postal code',
                       validation = {'Input too short': lambda value: len(value) >= 5})\
                        .on('keydown.enter', lambda: trigger_generation.refresh())\
                        .props('clearable')\
                        .classes('w-80')
                    
                    # generation function for ESTIMATOR, triggered upon entering an address
                    @ui.refreshable
                    def trigger_generation():
                        """
                        FUNCTION to trigger address-related api calls including:
                        1. geocode to get LAT, LON coordinates, and SYSTEM_MSG
                        2. get_suninfo to get exposure_times dictionary with time of dawn, sunrise, sunriseEnd, solarNoon, sunsetStart, sunset, dusk
                        3. pick icon to display alongside current datetime (DT) based on time of day
                        4. get_optimal_angles to get optimal azimuth and altitude angles for PVWatts query
                        """
                        if ADDRESS.value != "":
                            try:
                                # sequentially run all the functions to return outputs
                                LAT, LON, SYSTEM_MSG = geocode(ADDRESS.value)
                                DT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) # UTC
                                exposure_times = get_suninfo(LAT, LON, DT)
                                azimuth, tilt = get_optimal_angles(LAT, LON, exposure_times)
                                AC_output = get_solar_estimate(LAT, LON, azimuth, tilt)
                                
                                with ui.column().classes('w-100 items-left'):
                                    ui.label(f"{SYSTEM_MSG}")
                                    ui.label(f"The coordinates are ({LAT}, {LON})")
                                
                                    with ui.row():
                                        if pd.to_datetime(DT) < pd.to_datetime(exposure_times['dawn']) or pd.to_datetime(DT) > pd.to_datetime(exposure_times['dusk']):
                                            ui.image('./assets/nosun.svg').classes('w-8')
                                            ui.label(f"\nCurrent time is {time_readable(utc_to_sgt(DT))}\n")
                                            ui.image('./assets/nosun.svg').classes('w-8')
                                        elif pd.to_datetime(DT) <= pd.to_datetime(exposure_times['sunriseEnd']) or pd.to_datetime(DT) >= pd.to_datetime(exposure_times['sunsetStart']):
                                            ui.image('./assets/halfsun.svg').classes('w-8')
                                            ui.label(f"\nCurrent time is {time_readable(utc_to_sgt(DT))}\n")
                                            ui.image('./assets/halfsun.svg').classes('w-8')
                                        else:
                                            ui.image('./assets/fullsun.svg').classes('w-8')
                                            ui.label(f"\nCurrent time is {time_readable(utc_to_sgt(DT))}\n")
                                            ui.image('./assets/fullsun.svg').classes('w-8')
                                            
                                    ui.label("Today's Expected Solar Exposure:\n")
                                    with ui.grid(columns = 2):
                                        ui.label(f"{time_readable(utc_to_sgt(exposure_times['dawn']))}")
                                        ui.label("DAWN")
                                        ui.label(f"{time_readable(utc_to_sgt(exposure_times['sunrise']))}")
                                        ui.label("SUNRISE")
                                        ui.label(f"{time_readable(utc_to_sgt(exposure_times['solarNoon']))}")
                                        ui.label("SOLAR NOON")
                                        ui.label(f"{time_readable(utc_to_sgt(exposure_times['sunset']))}")
                                        ui.label("SUNSET")
                                        ui.label(f"{time_readable(utc_to_sgt(exposure_times['dusk']))}")
                                        ui.label("DUSK")
                                    
                                    ui.label("Optimal Solar Panel Orientation:\n")
                                    with ui.grid(columns = 2):
                                        ui.label("Azimuth")
                                        ui.label(f"{np.round(azimuth, 2)}° ({to_bearing(azimuth)})")
                                        ui.label("Tilt")
                                        ui.label(f"{np.round(tilt,2)}°")
                                
                            except AssertionError:
                                ui.label("Oops! The address you have queried was not found in Singapore")
                                ui.label("Please input a Singapore address or postal code or simply type 'SUNNY' and hit enter for an island-averaged estimate.")
                    trigger_generation()
                    
                    # TODO: grey-out NEXT button unless there is a valid output for Generation step
                    with ui.stepper_navigation():
                        ui.button('Next', on_click = stepper.next)
                        
                with ui.step('Consumption'):
                    # 2. enter dwelling type
                    DWELLING = ui.select(label = 'Select dwelling type',
                        options = ['1-room / 2-room', '3-room', '4-room', '5-room and Executive', 'Landed Property'],
                        with_input = True)\
                        .classes('w-80')\
                        .on('update:model-value', lambda: trigger_roofarea.refresh())
                    
                    # refresh roof area function for ESTIMATOR triggered upon entering dwelling type
                    @ui.refreshable
                    def trigger_roofarea():
                        # FUNCTION to trigger roof area user input if dwelling type is Landed Property
                        DT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        annual_demand, ytd_demand, hours_elapsed = get_demand_estimate(DT, DWELLING.value)
                        ui.label(f"{annual_demand}")
                        ui.label(f"{ytd_demand}")
                        ui.label(f"{hours_elapsed}")
                        
                        num_panels = 1
                        
                        if DWELLING.value == "Landed Property":
                            ui.label('Estimate your roof area in m²')
                            with ui.row():
                                ROOF_AREA = ui.slider(min = 10, max = 200, value = 10).classes('w-50')
                                ui.label().bind_text_from(ROOF_AREA, 'value')
                            
                            num_panels = int(np.floor(float(ROOF_AREA.value)/1.6)) # TODO: needs to refresh
                            ui.label(f"You can fit {num_panels} Standard 250 W (1.6 m²) Solar Panels on your roof.")
                            
                    trigger_roofarea()
                    
                    with ui.stepper_navigation():
                        ui.button('Next', on_click = stepper.next)
                        ui.button('Back', on_click = stepper.previous).props('flat')
                
        # what appears in realtime tab
        # TODO: what are the needed params?
        with ui.tab_panel(realtime):
            with ui.column().classes('w-full items-center'):
                ui.label('Realtime table')

                # create plot using matplotlib.pyplot
                plt.figure(figsize=(8, 6), dpi=80)
                x = np.linspace(0.0, 5.0)
                y = np.cos(2 * np.pi * x) * np.exp(-x)
                plt.plot(x, y, '-')
                os.makedirs('./assets/', exist_ok=True)
                plt.savefig('./assets/sparkline_table.png')
                app.add_static_files('/assets/', './assets/')
                
                def mouse_handler(e: MouseEventArguments):
                    print(e)
                    ii.tooltip(f'{e.image_x}, {e.image_y}')
                    ii.update()
                    # ii.content += f'<circle cx="{e.image_x}" cy="{e.image_y}" r="15" fill="none" stroke="{color}" stroke-width="4" />'
                    # ui.notify(f'{e.type} at ({e.image_x:.1f}, {e.image_y:.1f})')

                
                ii = ui.interactive_image('./assets/sparkline_table.png',
                                            on_mouse=mouse_handler,
                                            events=['click'],
                                            cross=True) # show cross on hover
                ii.tooltip('After')

ui.run(title = "Jamie Neo [EMA]")
