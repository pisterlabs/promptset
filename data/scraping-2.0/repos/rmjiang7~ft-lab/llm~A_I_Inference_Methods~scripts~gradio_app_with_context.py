import argparse

import gradio as gr
from peft import AutoPeftModelForCausalLM, PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path_or_id",
    type=str,
    default="mistralai/Mistral-7B-v0.1",
    required=False,
    help="Model ID or path to saved model",
)

parser.add_argument(
    "--lora_path",
    type=str,
    default=None,
    required=False,
    help="Path to the saved lora adapter",
)

args = parser.parse_args()

if args.lora_path:
    # load base LLM model with PEFT Adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.lora_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        use_flash_attention_2=True,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path_or_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        use_flash_attention_2=True,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_id)

# The connection to the database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver= "psycopg2",
    host = "localhost",
    port = "5432",
    database = "postgres",
    user= "username",
    password="password"
)

# The embedding function that will be used to store into the database
embedding_function = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True}
)

# Creates the database connection to our existing DB
db = PGVector(
    connection_string = CONNECTION_STRING,
    collection_name = "embeddings",
    embedding_function = embedding_function
)

with gr.Blocks() as demo:
    gr.HTML(
        f"""
        <h2> Instruction Chat Bot Demo </h2>
        <h3> Model ID : {args.model_path_or_id} </h3>
        <h3> Peft Adapter : {args.lora_path} </h3>
    """
    )

    chat_history = gr.Chatbot(label="QA Bot")
    msg = gr.Textbox(label="Question")
    with gr.Accordion(label="Generation Parameters", open=False):
        prompt_format = gr.Textbox(
            label="Formatting prompt", value="{question}", lines=8
        )
        with gr.Row():
            max_new_tokens = gr.Number(
                minimum=25, maximum=500, value=100, label="Max New Tokens"
            )
            temperature = gr.Slider(
                minimum=0, maximum=1.0, value=0.7, label="Temperature"
            )

    clear = gr.ClearButton([msg, chat_history])

    def user(user_message, history):
        return "", [[user_message, None]]

    def bot(chat_history, prompt_format, max_new_tokens, temperature):
        # Format the instruction using the format string with key
        # {instruction}

        docs_with_scores = db.similarity_search_with_score(chat_history[-1][0], k = 1)
        formatted_inst = prompt_format.format(
            context = docs_with_scores[0][0].page_content,
            question = chat_history[-1][0]
        )

        # Tokenize the input
        input_ids = tokenizer(
            formatted_inst, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        # Support for streaming of tokens within generate requires
        # generation to run in a separate thread
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=temperature,
            use_cache=True,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        chat_history[-1][1] = ""
        for new_text in streamer:
            chat_history[-1][1] += new_text
            yield chat_history

    msg.submit(user, [msg, chat_history], [msg, chat_history], queue=False).then(
        bot, [chat_history, prompt_format, max_new_tokens, temperature], chat_history
    )

demo.queue()
demo.launch()
