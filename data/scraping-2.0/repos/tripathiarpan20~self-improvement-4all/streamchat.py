import datetime
import asyncio
import json
import sys
import streamlit as st
import random
import time
import os
from langchain_utils import EMBEDDING_MODEL, initialize_vector_memory, load_vector_memory, save_vector_memory, prepare_memory_object, retrieve
from langchain.schema import Document
from langchain import PromptTemplate

old_stdout = sys.stdout
log_file = open("message.log","w")
sys.stdout = log_file

if os.path.exists('logs/memory_base.txt'):
  os.remove('logs/memory_base.txt')
if os.path.exists('logs/memory_empathetic.txt'):
  os.remove('logs/memory_empathetic.txt')
if os.path.exists('logs/memory_therapeutic.txt'):
  os.remove('logs/memory_therapeutic.txt')

#Assuming 0 indexing, the even entries refer to 'Patient' messages, while odd entries refer to 'Joi' entries
if 'dialogue_history' not in st.session_state:
    st.session_state['dialogue_history'] = []

try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.")

# For local streaming, the websockets are hosted without ssl - ws://
HOST = 'localhost:5005'
URI = f'ws://{HOST}/api/v1/chat-stream'
global_stream = ''
joi_thoughts = ''
joi_message = ''
memory_importance_reasoning = ''
importance_score = None


dialogue_memory = None
empathetic_memory = None
therapeutic_memory = None

prompt_template = PromptTemplate.from_template(
"""The following are the memories about the patient that Joi recalls:
'''
{memories}
'''

The following are the therapeutical insights that Joi recalls:
'''
{therapeuits}
'''

The following are the empathetic thoughts about the patient that Joi recalls:
'''
{empathatics}
'''

Given the conversation history between the patient and Joi below, predict Joi's next dialogue in the format mentioned before:
'''
{convo_history}
'''

"""
)

reflection_template_pt1 = PromptTemplate.from_template(
"""The following contains insights by a therapist in a session with a patient:
{retrieved_info}
Output a numbered list of 5 questions (structured in third-person format) that the Therapist can answer with the given information

"""
)

reflection_template_pt2 = PromptTemplate.from_template(
"""CONTEXT:
'''
{retrieved_info}
'''

Answer the following questions given above information briefly (no more than couple lines), do not include the original questions in the response:
{questions}
"""
)


latest_prompt = 'logs/prompt.txt'

st.session_state.base_txt = 'logs/memory_base.txt'
st.session_state.therapeutic_txt = 'logs/memory_therapeutic.txt'
st.session_state.empathetic_txt = 'logs/memory_empathetic.txt'


VECTOR_STORE_ROOT = '/content/vector_mems'
if 'dialogue_memory' not in st.session_state:
  if os.path.exists(os.path.join(VECTOR_STORE_ROOT, 'dialogue') + '.pbz2'):
    st.session_state.dialogue_memory = load_vector_memory(VECTOR_STORE_ROOT, 'dialogue')
  else:
    st.session_state.dialogue_memory = initialize_vector_memory()

if 'empathetic_memory' not in st.session_state:
  if os.path.exists(os.path.join(VECTOR_STORE_ROOT, 'empathetic') + '.pbz2'):
    st.session_state.empathetic_memory = load_vector_memory(VECTOR_STORE_ROOT, 'empathetic')
  else:
    st.session_state.empathetic_memory =  initialize_vector_memory()

if 'therapeutic_memory' not in st.session_state:
  if os.path.exists(os.path.join(VECTOR_STORE_ROOT, 'therapeutic') + '.pbz2'):
    st.session_state.therapeutic_memory = load_vector_memory(VECTOR_STORE_ROOT, 'therapeutic')
  else:
    st.session_state.therapeutic_memory = initialize_vector_memory()

REFLECTION_EVERY_N_MESSAGES = 3
if 'msg_count' not in st.session_state:
  st.session_state.msg_count = 0

# For reverse-proxied streaming, the remote will likely host with ssl - wss://
# URI = 'wss://your-uri-here.trycloudflare.com/api/v1/stream'

async def run(user_input, history, instruction_template = 'Vicuna-v1.1'):
    # Note: the selected defaults change from time to time.
    request = {
        'user_input': user_input,
        'max_new_tokens': 250,
        'history': history,
        'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
        'character': 'Example',
        'instruction_template': instruction_template,  # Will get autodetected if unset
        # 'context_instruct': '',  # Optional
        'your_name': 'You',

        'regenerate': False,
        '_continue': False,
        'stop_at_newline': False,
        'chat_generation_attempts': 1,
        'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 1.1,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    async with websockets.connect(URI, ping_interval=None) as websocket:
        await websocket.send(json.dumps(request))

        while True:
            incoming_data = await websocket.recv()
            incoming_data = json.loads(incoming_data)

            match incoming_data['event']:
                case 'text_stream':
                    yield incoming_data['history']
                case 'stream_end':
                    return


async def print_response_stream(user_input, history):
    global global_stream
    global_stream = ""
    joi_message = ""
    cur_len = 0
    message_placeholder = st.empty()
    message_placeholder_sidebar = st.sidebar.empty()

    sidebar_msg = None
    #main_msg = None
    async for new_history in run(user_input, history, instruction_template = 'Template_recalled_dialogue_2'):
        cur_message = new_history['visible'][-1][1][cur_len:]
        cur_len += len(cur_message)
        global_stream += cur_message

        if 'Joi:' in global_stream:
          joi_message = global_stream[global_stream.find('Joi:') + 4:] + "▌"
          message_placeholder.markdown(joi_message)
        else:
          sidebar_msg = global_stream + "▌"
          message_placeholder_sidebar.markdown(sidebar_msg)

    #joi_message = main_msg
    del message_placeholder, message_placeholder_sidebar


#Generates the questions
async def run_reflection_pt1(memory, last_n = 3):
    global_stream = f"----------Reflecting on last {last_n} observations----------\n"
    global_stream = global_stream + "Reflection questions:\n"
    init_len_stream = len(global_stream)

    cur_len = 0
    message_placeholder_sidebar = st.sidebar.empty()

    sidebar_msg = None

    retrieved_info = ""
    for x in memory.memory_stream[-last_n:]:
        retrieved_info = retrieved_info + x.page_content + '\n'

    reflection_prompt_1 = reflection_template_pt1.format(retrieved_info = retrieved_info)

    llm_history = [reflection_prompt_1, ""]
    history = {'internal': [llm_history], 'visible': [llm_history]}

    async for new_history in run(reflection_prompt_1, history, instruction_template = 'Vicuna-v1.1'):
        cur_message = new_history['visible'][-1][1][cur_len:]
        cur_len += len(cur_message)
        global_stream += cur_message

        sidebar_msg = global_stream + "▌"
        message_placeholder_sidebar.markdown(sidebar_msg)

    del message_placeholder_sidebar

    #Updating history with new message
    llm_history[-1] = global_stream[init_len_stream:]
    return global_stream, llm_history

#Generates the answers to the questions
async def run_reflection_pt2(memory, llm_history, last_n = 3):
    global_stream = ""
    cur_len = 0
    message_placeholder_sidebar = st.sidebar.empty()

    retrieved_info = ""
    for x in memory.memory_stream[-last_n:]:
        retrieved_info = retrieved_info + x.page_content + '\n'

    reflection_prompt_2 = reflection_template_pt2.format(retrieved_info = retrieved_info, questions = llm_history[-1])

    llm_history.append(reflection_prompt_2)
    llm_history.append("")
    history = {'internal': [llm_history], 'visible': [llm_history]}

    sidebar_msg = None
    #main_msg = None
    async for new_history in run(reflection_prompt_2, history, instruction_template = 'Vicuna-v1.1'):
        cur_message = new_history['visible'][-1][1][cur_len:]
        cur_len += len(cur_message)
        global_stream += cur_message

        sidebar_msg = global_stream + "▌"
        message_placeholder_sidebar.markdown(sidebar_msg)

    del message_placeholder_sidebar
    return global_stream.split('\n')



#async def assign_memory_importance(user_input, history):
#     global importance_score
#     global memory_importance_reasoning

#     with st.sidebar:
#       message_placeholder_sidebar = st.empty()

#     sidebar_msg = None
#     cur_len = 0
#     async for new_history in run(user_input, history, instruction_template = 'Template2'):
#         cur_message = new_history['visible'][-1][1][cur_len:]
#         cur_len += len(cur_message)
#         memory_importance_reasoning += cur_message
#         message_placeholder_sidebar.markdown(memory_importance_reasoning)

#     #TODO: Parse and assign `importance_score`


#     del message_placeholder_sidebar

# Creating a lock for the function below
reflection_lock = asyncio.Lock()
async def reflection_task(memory, reflection_vector_function, timestamp):
    async with reflection_lock:
        ques, history = await run_reflection_pt1(memory, REFLECTION_EVERY_N_MESSAGES)
        ans = await run_reflection_pt2(memory, history, REFLECTION_EVERY_N_MESSAGES)
        for reflection in ans:
            reflection_vector = reflection_vector_function(reflection)
            memory.add_documents([Document(page_content=reflection, metadata=prepare_memory_object(timestamp, timestamp, reflection_vector, 10))])


async def main():
    st.title("Self-improvement WizardLM v1.1")

    st.sidebar.markdown("Bot observation stream")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.sidebar:
      if "messages" not in st.session_state:
          st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #with st.sidebar:
    #  for message in st.sidebar.session_state.messages:
    #      with st.sidebar.chat_message(message["role"]):
    #          st.markdown(message["content"])

    memory_prompt_history = []

    user_input = st.chat_input("What is up?")
    if user_input:

        #Fetching top 3 relevant memories from corresponding retreivers
        memories = retrieve(user_input, st.session_state.dialogue_memory)
        therapeuits = retrieve(user_input, st.session_state.therapeutic_memory)
        empathatics = retrieve(user_input, st.session_state.empathetic_memory)

        if memories is not "":
          with st.sidebar:
            with st.chat_message("assistant"):
              st.write("Recalled memory:\n" + memories)

        if therapeuits is not "":
          with st.sidebar:
            with st.chat_message("assistant"):
              st.write("Recalled therapeuits:\n" + therapeuits)

        if empathatics is not "":
          with st.sidebar:
            with st.chat_message("assistant"):
              st.write("Recalled empathatics:\n" + empathatics)

        #Extracting just the last 6 dialogues in the conversation
        if len(st.session_state.dialogue_history)>6:
          last_n_dialogues = st.session_state.dialogue_history[-6:]
        else:
         last_n_dialogues = st.session_state.dialogue_history

        #Formatting the conversations for prompts
        formatted_dialogues = [('Patient: "' + dialogue + '"') if (idx%2 is 0) else ('Joi: "' + dialogue + '"') for idx, dialogue in enumerate(last_n_dialogues + [user_input])]
        formatted_dialogues = '\n'.join(formatted_dialogues)

        prompt = prompt_template.format(memories = memories, therapeuits = therapeuits, empathatics = empathatics, convo_history = formatted_dialogues)

        llm_history = [prompt, ""]
        history = {'internal': [llm_history], 'visible': [llm_history]}

        with st.chat_message("user"):
          st.write(user_input)
        with st.sidebar:
          with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            await print_response_stream(user_input, history)

        #TODO: Add memory assignment prompt for all the 3 memories (message, therapeuits, empathatics)
        #memory_prompt_history.append("Patient: " + user_input + "\nTherapist's thoughts:" + joi_thoughts)
        #memory_prompt_history.append("")
        #mem_history = {'internal': [memory_prompt_history], 'visible': [memory_prompt_history]}
        #with st.sidebar.chat_message("assistant"):
        #    await assign_memory_importance(user_input, mem_history)

        #with st.sidebar:
        #    st.session_state.messages.append({"role": "user", "content": user_input})
        #    st.session_state.messages.append({"role": "assistant", "content": joi_thoughts})
        #    st.session_state.messages.append({"role": "assistant", "content": memory_importance_reasoning})

        st.session_state.messages.append({"role": "user", "content": user_input})


        ### Adding the entities to vector memories ###
        #Parsing memory entries from response
        joi_message = global_stream[global_stream.find('Joi:') + len('Joi:'):]
        joi_empathy = global_stream[global_stream.find('Inner Empathetic Voice:') + len('Inner Empathetic Voice:'): global_stream.find('Inner Theoretical Therapist Voice:')]
        joi_therapeutic = global_stream[global_stream.find('Inner Theoretical Therapist Voice:') + len('Inner Theoretical Therapist Voice:'): global_stream.find('Joi:')]

        timestamp = datetime.datetime.now()
        lastAccess = timestamp

        dialogue_vector = EMBEDDING_MODEL.embed_query(joi_message)
        empathetic_vector = EMBEDDING_MODEL.embed_query(joi_empathy)
        therapeutic_vector = EMBEDDING_MODEL.embed_query(joi_therapeutic)

        st.session_state.dialogue_memory.add_documents([Document(page_content=joi_message, metadata=prepare_memory_object(timestamp, lastAccess, dialogue_vector, 10))])
        st.session_state.empathetic_memory.add_documents([Document(page_content=joi_empathy, metadata=prepare_memory_object(timestamp, lastAccess, empathetic_vector, 10))])
        st.session_state.therapeutic_memory.add_documents([Document(page_content=joi_therapeutic, metadata=prepare_memory_object(timestamp, lastAccess, therapeutic_vector, 10))])

        #Saving backup of vector memories to disk
        #TODO: Find a way to speedup memory, maybe via asynchronity
        #save_vector_memory(dialogue_memory, mem_name = 'dialogue')
        #save_vector_memory(empathetic_memory, mem_name = 'empathetic')
        #save_vector_memory(therapeutic_memory, mem_name = 'therapeutic')
        ######

        st.session_state.messages.append({"role": "assistant", "content": joi_message})
        #st.session_state.messages.append({"role": "assistant", "content": global_stream})

        st.session_state.dialogue_history = [x["content"] for x in st.session_state.messages]


        ### Logging
        with open(latest_prompt, "w") as file:
          file.write(prompt)
          #file.write('\n'.join(st.session_state.dialogue_history))
          #file.write(formatted_dialogues)
        with open(st.session_state.base_txt, "a") as file:
          file.write(joi_message + '\n')
        with open(st.session_state.therapeutic_txt, "a") as file:
          file.write(joi_empathy + '\n')
        with open(st.session_state.empathetic_txt, "a") as file:
          file.write(joi_therapeutic + '\n')


          #Reflection on dialogues
          #with st.chat_message("assistant"):
          #  dialogue_ques, history = await run_reflection_pt1(st.session_state.dialogue_memory, REFLECTION_EVERY_N_MESSAGES)
          #  dialogue_ans = await run_reflection_pt2(st.session_state.dialogue_memory, history, REFLECTION_EVERY_N_MESSAGES)
          #  for reflection in dialogue_ans:
          #    dialogue_vector = EMBEDDING_MODEL.embed_query(reflection)
          #    st.session_state.dialogue_memory.add_documents([Document(page_content=reflection, metadata=prepare_memory_object(timestamp, lastAccess, dialogue_vector, 10))])


        ###Reflection
        st.session_state.msg_count = st.session_state.msg_count + 1
        if st.session_state.msg_count%REFLECTION_EVERY_N_MESSAGES is 0:
          # Reflection on empathetic memory
          await reflection_task(
              st.session_state.empathetic_memory,
              EMBEDDING_MODEL.embed_query,
              timestamp
          )

          # Reflection on therapeutic memory
          await reflection_task(
              st.session_state.therapeutic_memory,
              EMBEDDING_MODEL.embed_query,
              timestamp
          )

        #DONE: Add sliding window context handling, i.e, when user messages get too many, delete older messages.
        #DONE: Add recalled memories part of the prompt for response by therapist, on top of reasoning with inner thoughts
        #DONE: Add vector memory system
        #DONE: Add queueing of messages
        #DONE: Add reflection of memories
        #DONE: Evaluation
if __name__ == '__main__':
    asyncio.run(main())