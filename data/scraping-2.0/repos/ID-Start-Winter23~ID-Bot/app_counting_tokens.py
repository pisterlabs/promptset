import os
import openai
import gradio as gr
from llama_index import ServiceContext, StorageContext, load_index_from_storage, set_global_service_context
#from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
import asyncio
import time

from llama_index.memory import ChatMemoryBuffer

from llama_index.llms import MockLLM
from llama_index import MockEmbedding
import tiktoken
from llama_index.callbacks import CallbackManager, TokenCountingHandler

from theme import CustomTheme

# use MockLLM and MockEmbedding for testing (it's free); use OpenAI for production
llm = MockLLM(max_tokens=56)
embed_model = MockEmbedding(embed_dim=1536)

#llm = OpenAI(temperature=0.1, model="gpt-4-1106-preview")
splitter =TokenTextSplitter(
     chunk_size=1024,
     chunk_overlap=128,
     separator=" "
)

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4-1106-preview").encode
)

callback_manager = CallbackManager([token_counter])

set_global_service_context(
    ServiceContext.from_defaults(
        llm=llm, text_splitter=splitter, embed_model=embed_model, callback_manager=callback_manager
    )
)

#set_global_service_context(
#    ServiceContext.from_defaults(
#        llm=llm, text_splitter=splitter
#    )
#)

# create storage context
storage_context = StorageContext.from_defaults(persist_dir="modulhandbuch")
# load index
index = load_index_from_storage(storage_context)


system_prompt =(
    "You are a study program manager."
)

context = (
    "Context information is below.\n"
    "--------------\n"
    "{context_str}\n"
    "--------------\n"
    "Greet the user in a friendly way.\n"
    "Always keep the user on a first-name basis.\n"
    "Answer always in German and in a friendly, humorous matter.\n"
    "Keep the answers short and simple.\n"
    "Tell the user in a friendly way that you can only answer questions about the modules and courses in the study program Informatics and Design if they have questions about other topics.\n"
    "If the user asks a question that you cannot answer, tell them that you cannot answer the question and that they should contact the study program manager.\n"
    "Don't be afraid to ask the user to rephrase the question if you don't understand it.\n"
    "Don't repeat yourself.\n"
)

memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

chat_engine = index.as_chat_engine(
    similarity_top_k = 2,
    chat_mode = "context",
    system_prompt = system_prompt,
    context_template = context,
    memory = memory,
)

default_text="Ich beantworte Fragen zum Modulhandbuch des Studiengangs Informatik und Design. Wie kann ich Dir helfen?"

bot_examples = [
    "Wer lehrt Mobile Anwendungen?",
    "Welche Prüfungsform hat das Modul Software Engineering?",
    "Wie viele Semesterwochenstunden hat das Modul Computational Thinking?",
]

submit_button = gr.Button(
        value="Frag MUC.DAI",
        elem_classes=["ask-button"],
)

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)
token_counter.reset_counts()

response_counter = 0

def response(message, history):
    global response_counter
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if message == "":
        answer = default_text
    else:
        answer = chat_engine.stream_chat(message)
        #answer = chat_engine.stream_chat(message, chat_history=chat_engine.chat_history)

    print("message", message)
    print("answer", answer)
    print("history", history)

    output_text = ""
    for token in answer.response_gen:
        output_text += token
        time.sleep(0.05)
        yield output_text

    print(
        "Embedding Tokens: ",
        token_counter.total_embedding_token_count,
        "\n",
        "LLM Prompt Tokens: ",
        token_counter.prompt_llm_token_count,
        "\n",
        "LLM Completion Tokens: ",
        token_counter.completion_llm_token_count,
        "\n",
        "Total LLM Token Count: ",
        token_counter.total_llm_token_count,
        "\n",
    )

    print(response_counter)
    response_counter += 1
    if (response_counter % 5) == 0:
        print("Resetting chat engine ..." + str(response_counter))
        chat_engine.reset()

def main():
    #openai.api_key = os.environ["OPENAI_API_KEY"]

    custom_theme = CustomTheme()

    chatbot = gr.Chatbot(
        avatar_images=["assets/smile.png", "assets/mucdai.png"],
        layout='bubble',
        height=600,
        value=[[None, default_text]]
    )

    chat_interface = gr.ChatInterface(
        fn=response,
        retry_btn=None,
        undo_btn=None,
        title="MUC.DAI Informatik und Design - frag alles, was Du wissen willst!",
        submit_btn=submit_button,
        theme=custom_theme,
        chatbot=chatbot,
        #textbox=gr.Textbox(placeholder="Frage mich ..."),
        css="style.css",
        examples=bot_examples
    )

    chat_interface.launch(inbrowser=True, debug=True, favicon_path="assets/favicon.ico")


if __name__ == "__main__":
    main()
