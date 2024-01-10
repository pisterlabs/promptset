import logging
import time

import tiktoken
import torch
from google.api_core.exceptions import ResourceExhausted, InternalServerError
from langchain.chains import LLMChain
from langchain.chat_models import ChatVertexAI, ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import config

tqdm.pandas()


def select_titles(topic_df, topic, limiter, base_prompt):
    """
    Create the base prompt that will be used to ask the LLM on the theme summarization of the topics from the
    title of the articles in the topics together with the published dates. The goal is to
    fit as many titles into the prompt as possible without exceeding the context window. The maximum
    fit is determined by the limiter parameter passed in, which is a function that takes the prompt and
    determine if it is too big.
    """

    topic_segment = topic_df.loc[topic_df.topic == topic][["title", "published", "topic_prob"]].sort_values(
        by="topic_prob", ascending=False)
    if len(topic_segment.index) == 0:
        raise KeyError("Invalid Topic: " + str(topic))

    current_idx = 0
    end_idx = len(topic_segment.index)
    prompt = base_prompt
    agg_text = ""
    row_format = "Title: %s\n"
    while limiter(prompt) and current_idx < end_idx:
        agg_text = agg_text + row_format % (topic_segment.iloc[current_idx]["title"])
        prompt = base_prompt.format(text=agg_text)
        current_idx = current_idx + 1
    logging.info("Summarizing themes using the prompt:\n%s" % prompt)
    return prompt.strip()


def create_huggingface_topic_summarizer(model, prompt=config.TOPIC_SUM_HF_PROMPT, temperature=0,
                                        max_new_tokens=256, max_prompt_tokens=2048):
    """
    Creates a topic summarizer function that uses a local HF model and the summarization pipeline.
    """

    tokenizer = AutoTokenizer.from_pretrained(model, device_map="auto", max_input_size=max_prompt_tokens)
    model = AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True, device_map="auto")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)

    def token_count_limiter(final_prompt):
        tokens = tokenizer(final_prompt, return_tensors="pt")
        token_count = tokens["input_ids"].shape[1]
        return token_count <= max_prompt_tokens

    def summarize_topics(topic_df, topic_number):
        final_prompt = select_titles(topic_df, topic_number, token_count_limiter, base_prompt=prompt)
        result = summarizer(final_prompt, temperature=temperature)["summary_text"].strip()
        return result

    return summarize_topics


def create_lc_summarizer(chain, adapter=lambda x: x, max_prompt_tokens=4096):
    engine = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(engine)

    def token_count_limiter(prompt):
        tokens = encoding.encode(prompt)
        return len(tokens) <= max_prompt_tokens

    def summarize_topic(topic_df, topic_number):
        final_prompt = select_titles(topic_df, topic_number, token_count_limiter, base_prompt="{text}")
        try:
            raw = chain(final_prompt)
        except ValueError:
            return config.TOPIC_SUM_LC_DEFAULT
        return adapter(raw)

    return summarize_topic


def create_passthrough_summarizer(max_prompt_tokens=512):
    engine = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(engine)

    def token_count_limiter(prompt):
        tokens = encoding.encode(prompt)
        return len(tokens) <= max_prompt_tokens

    def summarize_topic(topic_df, topic_number):
        final_prompt = select_titles(topic_df, topic_number, token_count_limiter, base_prompt="{text}")
        return final_prompt

    return summarize_topic


def create_topic_summarizer(kind="pass", **kwargs):
    """
    This creates a summarizer function that will summarize the theme of the articles in a given topic.
    The signature of the summarizer function is summarizer: (DataFrame, Integer) -> String. It will select
    the appropriate rows from the dataframe and create a suitable summary for later querying.
    """

    if kind == "hf":
        return create_huggingface_topic_summarizer(**kwargs)
    if kind == "lc":
        return create_lc_summarizer(**kwargs)
    if kind == "pass":
        return create_passthrough_summarizer(**kwargs)
    raise ValueError("Invalid kind: " + kind)


def create_chat_chain(llm):
    output_parser = RegexParser(regex=config.TOPIC_SUM_LC_REGEX, output_keys=["summary"])
    system = SystemMessagePromptTemplate.from_template(config.TOPIC_SUM_LC_SYSTEM_PROMPT)
    user = HumanMessagePromptTemplate.from_template(config.TOPIC_SUM_LC_USER_PROMPT)
    prompt = ChatPromptTemplate.from_messages([system, user])
    return LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)


def create_palm2_chain(credentials, max_output_tokens=1024):
    plan_llm = ChatVertexAI(temperature=0,
                            model_name="chat-bison",
                            credentials=credentials,
                            project=config.VERTEX_AI_PROJECT,
                            max_output_tokens=max_output_tokens)
    return create_chat_chain(plan_llm)


def create_openai_chain(key, max_token=1024):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key, temperature=0, max_tokens=max_token)
    return create_chat_chain(llm)


def summarization_wrapper(summarizer, work, topic_df):
    while True:
        try:
            summary = summarizer(topic_df, work)
            return work, summary
        except (ResourceExhausted, InternalServerError):
            time.sleep(1)
