import os
import pkgutil
import re
from ast import literal_eval
from replicate.exceptions import ReplicateError
from langchain.llms.replicate import Replicate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

PATH = "virtmulib.applogic.ai_playlister"
SETUP_SECRET = "llm-setup-secret"
SETUP_REG = "llm-setup"
LLAMA_2_7B = "meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0"
LLAMA_2_13B = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
LLAMA_2_70B = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"

TEMP = 0.80
TOP_P = 0.9
MAX_LENGTH = 128
#TOP_K = 50
MAX_NEW_TOKENS = 800
MIN_NEW_TOKENS = -1
REPETITION_PENALTY = 1
DEBUG = False

#REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]

# If a question does not make any sense, or is not factually coherent, 
# explain why instead of answering something not correct. 
# If you don't know the answer to a question, please don't share false information.

def read_setup():
    setup = None
    try :
        setup = pkgutil.get_data(PATH, SETUP_SECRET)
    except FileNotFoundError:
        setup = pkgutil.get_data(PATH, SETUP_REG)
    return setup.decode('utf-8')


def setup_llm_conversation():
    llm = Replicate(
        model=LLAMA_2_13B,
        #replicate_api_token=REPLICATE_API_TOKEN,
        streaming=False,
        model_kwargs={
            "temperature": TEMP, "max_length": MAX_LENGTH, "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS, "min_new_tokens": MIN_NEW_TOKENS, 
            "repetition_penalty": REPETITION_PENALTY, "debug": DEBUG,
            #"top_k": top_k
        },
    )

    setup = read_setup()

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                setup
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True, k=2,key="question"
    )

    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    return conversation

def format_output(out):
    out = out.replace('\\n', '\n')
    out = out.replace('\\', "'")
    title_p = re.compile(r'\# (.+)')
    track_p = re.compile(r'\d+\. (.+)')
    title = None
    tracks = []
    for line in out.split(os.linesep.format()):
        tr = track_p.findall(line)
        if tr:
            tracks.append(tr[0])
        elif title_p.findall(line):
            title = line.split('#')[1].strip()
    return title, tracks

def inference(user_query):
    conversation = setup_llm_conversation()
    #chat_history = []
    #user_query = st.text_input('You:', '')
    #user_query = 'How many stars are in the solar system?'

    # user_query = """
    # Top artists: Mos Def, Laika, Miles Davis, 
    # Top songs: Herbie Hancock - Maiden Voyage, Laika - Praire Dog
    # What music means to me: Music is what I listen to when I need to discover new feelings and new imagination
    # Liked Music: I like many kinds of music, as long as its creative, emotional, and midtempo. I like world music, I like african american music in general.
    # Music not liked: I typically don't like rock music, but there are exceptions, typically open-minded artists that are self-conscious of the role of the west in colonalism.
    # Playlist request: It is a nice sunny sunday in december and I would like to listen to some creative and relaxed world music
    # """

    #if user_query:
    #user_response = {"role": "user", "content": user_query}
    #chat_history.append(user_response)
    response = None
    try:
        response = conversation({"question": user_query})
    except ReplicateError as e:
        #print(str(e))
        return None

    bot_response = response["text"]
    bot_response = {"role": "bot", "content": bot_response}
    #chat_history.append(bot_response)

    response = bot_response['content']

    return format_output(repr(response))
    # return literal_eval(response)
