# This code prepares and runs Fine Tuning tor Turbo OpenAi Model
from myfunc.mojafunkcija import st_style
import json
import os
import tiktoken
import numpy as np
from collections import defaultdict
import openai
import streamlit as st
import time
import io


# Example of fine tuning data
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}

ft_model = None
st_style()

# Next, we specify the data path and open the JSONL file


        


def verify_data():

    data_path = st.file_uploader(
        "Izaberite JSONL fajl za verifikaciju", key="upload_verifikacije", type='JSONL', help="JSONL file sa pitanjima i odgovorima")

    if data_path is not None:

        # promeniti ovo u string
       
        # Load dataset
        with io.open(data_path.name, "wb") as file:
                file.write(data_path.getbuffer())
        
        dataset = []
        with io.open(data_path.name, "rb") as file:
            for line in file:
                data = json.loads(line)
                dataset.append(data)

        # We can inspect the data quickly by checking the number of examples and the first item

        # Initial dataset stats
        st.info(f"Br. primera: {len(dataset)}")
        st.info("1. primer:")
        for message in dataset[0]["messages"]:
            st.success(message)

        # Now that we have a sense of the data, we need to go through all the different examples
        # and check to make sure the formatting is correct and matches the Chat completions message structure

        # Format error checks


        format_errors = defaultdict(int)

        for ex in dataset:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue

            messages = ex.get("messages", None)
            if messages is None:
                format_errors["missing_messages_list"] += 1
                continue

            for message in messages:
                if not isinstance(message, dict):
                    format_errors["message_missing_key"] += 1
                    continue

                if any(k not in ("role", "content", "name") for k in message):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role", None) not in ("system", "user", "assistant"):
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                if not content or not isinstance(content, str):
                    format_errors["missing_content"] += 1

            if not any(message.get("role", None) == "assistant" for message in messages):
                    format_errors["example_missing_assistant_message"] += 1

# Now, format_errors is a defaultdict containing error counts.


    

        if format_errors:
            st.error("Nađeni error-i:")
            for k, v in format_errors.items():
                st.info(f"{k}: {v}")
        else:
            st.success("Nisu pronađeni error-i")

        # Beyond the structure of the message, we also need to ensure that the length does not exceed the 4096 token limit.

        # Token counting functions
        encoding = tiktoken.get_encoding("cl100k_base")

        def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            num_tokens += 3
            return num_tokens

        def num_assistant_tokens_from_messages(messages):
            num_tokens = 0
            for message in messages:
                if message["role"] == "assistant":
                    num_tokens += len(encoding.encode(message["content"]))
            return num_tokens

        def print_distribution(values, name):
            st.info(f"\n#### Distribucija {name}:")
            st.info(f"min / max: {min(values)}, {max(values)}")
            st.info(f"mean / median: {np.mean(values)}, {np.median(values)}")
            st.info(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

        # Last, we can look at the results of the different formatting operations before proceeding with creating a fine-tuning job:

        # Warnings and tokens counts
        n_missing_system = 0
        n_missing_user = 0
        n_messages = []
        convo_lens = []
        assistant_message_lens = []

        for ex in dataset:
            messages = ex["messages"]
            if not any(message["role"] == "system" for message in messages):
                n_missing_system += 1
            if not any(message["role"] == "user" for message in messages):
                n_missing_user += 1
            n_messages.append(len(messages))
            convo_lens.append(num_tokens_from_messages(messages))
            assistant_message_lens.append(
                num_assistant_tokens_from_messages(messages))

        st.info(f"Br. primera kod kojih fali sistemska poruka: {n_missing_system}")
        st.info(f"Br. primera kod kojih fali korisnička poruka: {n_missing_user}")
        print_distribution(n_messages, "num_messages_per_example")
        print_distribution(convo_lens, "num_total_tokens_per_example")
        print_distribution(assistant_message_lens,
                            "num_assistant_tokens_per_example")
        n_too_long = sum(l > 4096 for l in convo_lens)
        st.info(
            f"\n{n_too_long} primera je potencijalno iznad tokenske granice od 4096; biće skraćeni tokom FT.")

        # Pricing and default n_epochs estimate
        MAX_TOKENS_PER_EXAMPLE = 4096

        MIN_TARGET_EXAMPLES = 100
        MAX_TARGET_EXAMPLES = 25000
        TARGET_EPOCHS = 3
        MIN_EPOCHS = 1
        MAX_EPOCHS = 25

        n_epochs = TARGET_EPOCHS
        n_train_examples = len(dataset)
        if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
            n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
        elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
            n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

        n_billing_tokens_in_dataset = sum(
            min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
        st.info(f"""
                Dataset ima ~{n_billing_tokens_in_dataset}, tokena koji će biti naplaćeni tokom treninga.\n
                Po default-u, trening nad ovim dataset-om će imati {n_epochs} iteraija.\n
                Po default-u, biće vam naplaćeno ~{n_epochs * n_billing_tokens_in_dataset} tokena [br. iteracija x br. tokena u dataset-u]\n
                Pogledajte stranicu za naplaćivanje da estimirate ukupne troškove.
                """)

##############################################################################################################

def create_ft_model():
# Upload fine tuning data
    data_path = st.file_uploader(
        "Izaberite JSONL fajl za kreiranje FT modela", key="upload_modela", type='JSONL', help="JSONL file sa pitanjima i odgovorima za trening")

    if data_path is not None:
        # promeniti ovo u string
        with io.open(data_path.name, "wb") as file:
                file.write(data_path.getbuffer())
        izvor = data_path.name

    # training_file validation name
        ver_path = st.file_uploader(
            "Izaberite JSONL fajl za validaciju FT modela", key="upload_ver", type='JSONL', help="JSONL file sa pitanjima i odgovorima za verifikaciju")

        if ver_path is not None:
            with io.open(ver_path.name, "wb") as file:
                file.write(ver_path.getbuffer())
            izvor = data_path.name
            ft_model_validation = ver_path.name

            suffix = st.text_input(
                "Unesite sufiks npr. ime_stila: ", help="Ime modela po kojem ćete ga prepoznati")  # suffix name
            if suffix:
                training_resp = openai.File.create(
                    file=open(izvor, "r", encoding="utf-8"),
                    purpose='fine-tune'
                )
                treining_file_id = training_resp.id
                validation_resp = openai.File.create(
                    file=open(ft_model_validation, "r", encoding="utf-8"),
                    purpose='fine-tune'
                )
                validation_file_id = validation_resp.id
                st.success(f"Trening fajl id: {treining_file_id}")
                st.success(f"Validacija fajl id: {validation_file_id}")
                with st.spinner("Please wait... "):
                    time.sleep(30)
                
                    # openai.organization = "org-77SVjL6mRtS5U57fDU1w1T2z"

                    # Creating a model

                    ft_job = openai.FineTuningJob.create(
                        training_file=treining_file_id,
                        validation_file=validation_file_id,
                        model="gpt-3.5-turbo",
                        suffix=suffix
                    )

                    st.info("Fine tuning job id: ", ft_job.id)
                    with open(f"log{suffix}.txt", "w", encoding='utf-8') as output_file:
                        output_file.write(ft_job.id)
                    # additional tasks


def ft_utils():

    ft_model = st.text_input("Unesi ime FT modela ili Job-a: ",
                            help="U zavisnosti od opcije unesite ime FT modela ili FT Job-a")
    if ft_model is None or ft_model == " " or ft_model == "":
        st.info("Unesite ime FT modela")
        return None
    else:
        return ft_model


def ft_jobs():
    # List 10 fine-tuning jobs
    st.code(openai.FineTuningJob.list(limit=10))


def ft_state():
    ft_model = ft_utils()
    # Retrieve the state of a fine-tune
    if ft_model is not None:
        try:
            st.code(openai.FineTuningJob.retrieve(ft_model))
        except:
            st.error("FT model ne postoji")


def cancel_job():
    ft_model = ft_utils()
    # Cancel a job
    if ft_model is not None:
        try:
            st.info(openai.FineTuningJob.cancel(ft_model))
        except:
            st.error("FT model ne postoji")
    # List up to 10 events from a fine-tuning job

def intro():
    st.write("Fine Tuning Turbo Modela - odaberite opciju")

def list_events():
    ft_model = ft_utils()

    if ft_model is not None:
        try:
            st.info(openai.FineTuningJob.list_events(id=ft_model, limit=50))
        except:
            st.error("FT model ne postoji")
    # Delete a fine-tuned model (must be an owner of the org the model was created in)


def delete_ft_model():
    ft_model = ft_utils()
    if ft_model is not None:
        try:
            st.info(openai.Model.delete(ft_model))
        except:
            st.error("FT model ne postoji")


def list_models():

    st.code(openai.Model.list())
    
def main():
   
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    with st.sidebar:
        st.success("Select an Action from a Drop Box.")
    page_names_to_funcs = {
            "Intro": intro,
            "Verify Data": verify_data,
            "Create FT Model": create_ft_model,
            "Model list": list_models,
            "List 10 FT jobs": list_events,
            "Retrieve the state of a FT": ft_state,
            "List up to 20 events from a FT job": ft_jobs,
            "Cancel a job": cancel_job,
            "Delete FT model": delete_ft_model,
        }

    demo_name = st.sidebar.selectbox("Odaberite aplikaciju", page_names_to_funcs.keys(
    ), help="Odaberite operaciju u vezi FT Modela")
    page_names_to_funcs[demo_name]()


