# Modified from https://gist.github.com/rht/c5fd97d9171a5e71863d85e47af7a7e3
import time
import traceback
from typing import Any, Dict

from langchain import OpenAI, PromptTemplate
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.output_parsers.regex import RegexParser

import zulip

# Supply the API key by setting OPENAI_API_KEY

# Set the temperature to 0 for a more deterministic output
llm = OpenAI(temperature=0)


def create_topic_generator():
    # Creates a topic phrase for the conversation.
    topic_generator_prompt_template = """
    Write a topic phrase of at most 4 words of the following:

    {text}

    TOPIC PHRASE OF THE TEXT:"""
    TOPIC_GENERATOR_PROMPT = PromptTemplate(
        template=topic_generator_prompt_template, input_variables=["text"]
    )
    return load_summarize_chain(llm, chain_type="stuff", prompt=TOPIC_GENERATOR_PROMPT)


def create_stuff_summarizer():
    # Creates a summary of the conversation using the
    # stuff summarizer.
    return load_summarize_chain(llm, chain_type="stuff")


def create_refine_summarizer():
    refine_template = (
        "Your job is to produce a final summary of a conversation\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary"
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    return load_summarize_chain(llm, chain_type="refine", refine_prompt=refine_prompt)


def create_map_reduce_summarizer():
    return load_summarize_chain(llm, chain_type="map_reduce")


def create_map_rerank_summarizer():
    map_template_string = """Given a part of a conversation below, your job is try to produce a possible summary
    for the entire conversation and give an integer score from 0 to 100 on how confident
    you are with that the summary reflects the entire conversation including the parts that are not seen,
    in the following format where SUMMARY must be only followed by a newline before SCORE:
    SUMMARY: 
    SCORE: 
    
    ENDOFFORMAT
    {text}
    """

    output_parser = RegexParser(
        regex=r"SUMMARY: (.+)\nSCORE: (\d+)", output_keys=["topic", "score"]
    )

    MAP_PROMPT = PromptTemplate(
        input_variables=["text"],
        template=map_template_string,
        output_parser=output_parser,
    )

    map_rerank = MapRerankDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=MAP_PROMPT),
        rank_key="score",
        answer_key="topic",
        document_variable_name="text",
        return_intermediate_steps=True,
    )
    return map_rerank


chain_topic_generator_stuff = create_topic_generator()
chain_topic_generator_map_rerank = create_map_rerank_summarizer()

chain_topic_summarizer_stuff = create_stuff_summarizer()
chain_topic_summarizer_refine = create_refine_summarizer()
chain_topic_summarizer_map_reduce = create_map_reduce_summarizer()


def topic_from_string(text):
    return chain_topic_generator_stuff.run([Document(page_content=text)]).strip()


def exit_immediately(s):
    print("\nERROR\n", s)
    exit(1)


# Retrieves all messages matching request from Zulip, starting at post id anchor.
# As recommended in the Zulip API docs, requests 1000 messages at a time.
# Returns a list of messages.
def request_all(client, request, anchor=0):
    request["anchor"] = anchor
    request["num_before"] = 0
    request["num_after"] = 1000
    response = safe_request(client.get_messages, request)
    msgs = response["messages"]
    while not response["found_newest"]:
        request["anchor"] = response["messages"][-1]["id"] + 1
        response = safe_request(client.get_messages, request)
        msgs = msgs + response["messages"]
    return msgs


# runs client.cmd(args). If the response is a rate limit error, waits
# the requested time and then retries the request.
def safe_request(cmd, *args, **kwargs):
    rsp = cmd(*args, **kwargs)
    while rsp["result"] == "error":
        if "retry-after" in rsp:
            print("Timeout hit: {}".format(rsp["retry-after"]))
            time.sleep(float(rsp["retry-after"]) + 1)
            rsp = cmd(*args, **kwargs)
        else:
            exit_immediately(rsp["msg"])
    return rsp


zulip_client = zulip.Client(config_file="./zuliprc")


def generate_topic(chain, docs):
    try:
        topic = chain.run(docs).strip()
    except Exception:
        traceback.print_exc()
        return "Error in generation"
    return f"**topic:** {topic}"


def summarize_and_generate_topic(chain, docs):
    try:
        response = chain(docs)
        if "intermediate_steps" in response:
            print("intermediate_steps:", response["intermediate_steps"])
        summary = response["output_text"].strip()
        topic = topic_from_string(summary)
    except Exception:
        traceback.print_exc()
        return "Error in generation"
    return f"**topic:** {topic} \n\n **summary:** {summary}"


def generate_topic_from_intro_message(chain, docs):
    return generate_topic(chain, docs[:1])


def get_answer(message):
    stream_topic = message.split("#**")[1][:-2]
    stream_name, topic_name = stream_topic.split(">")
    request = {
        "narrow": [
            {"operator": "stream", "operand": stream_name},
            {"operator": "topic", "operand": topic_name},
        ],
        "client_gravatar": True,
        "apply_markdown": False,
    }

    thread_content = request_all(zulip_client, request)
    thread_formatted = []
    for msg in thread_content:
        thread_formatted.append(f"{msg['sender_full_name']} said: {msg['content']}")

    print("Conversation text input from Zulip:\n{}".format("\n".join(thread_formatted)))
    # texts = text_splitter.split_text(thread_txt)
    docs = [Document(page_content=t) for t in thread_formatted]

    topic_chain = {
        "stuff": chain_topic_generator_stuff,
    }
    summary_chain = {
        "refine": chain_topic_summarizer_refine,
        "stuff": chain_topic_summarizer_stuff,
        "map_reduce": chain_topic_summarizer_map_reduce,
        "map_rerank": chain_topic_generator_map_rerank,
    }
    chain_tests = {
        "topic": (topic_chain, generate_topic),
        "summary": (summary_chain, summarize_and_generate_topic),
        "summary from first message only": (
            topic_chain,
            generate_topic_from_intro_message,
        ),
    }

    output = []
    output.append(f"# Summarize #**{stream_topic}**")
    for chain_type in chain_tests:
        chain_map, generate_content = chain_tests[chain_type]
        for chain_name in chain_map:
            before = time.perf_counter()
            with get_openai_callback() as cb:
                print(f'Running {chain_name} ({chain_type}) for "{stream_topic}"...')
                output.append(
                    f"## {chain_name} ({chain_type}):\n{generate_content(chain_map[chain_name], docs).strip()}"
                )
            after = time.perf_counter()
            output.append(
                f"**Tokens Used:** *{cb.total_tokens}*; **API Cost:** *{cb.total_cost}*; **Total Time:** *{after - before} seconds*"
            )

    return "\n\n".join(output)


# The code after this line could be simplified by https://github.com/zulip/python-zulip-api/pull/786
def handle_message(msg: Dict[str, Any]) -> None:
    print(f"Processing\n{msg}")
    if msg["type"] != "stream":
        return

    message = msg["content"]
    try:
        content = get_answer(message)
    except Exception:
        traceback.print_exc()
        content = "Failed to process message {}".format(msg["content"])
    request = {
        "type": "stream",
        "to": msg["display_recipient"],
        "topic": msg["subject"],
        "content": content,
    }
    print(f"Sending\n{content}")
    zulip_client.send_message(request)


def watch_messages() -> None:
    print("Watching for messages...")

    def handle_event(event: Dict[str, Any]) -> None:
        if "message" not in event:
            # ignore heartbeat events
            return
        handle_message(event["message"])

    # https://zulip.com/api/real-time-events
    narrow = [["is", "mentioned"]]
    zulip_client.call_on_each_event(
        handle_event, event_types=["message"], all_public_streams=True, narrow=narrow
    )


def generate_answers():
    questions = [
        # Short conversations
        "#**api documentation>user activity**",
        "#**api documentation>security scheme validation/testing**",
        '#**issues>visual notification on despite setting "Do not disturb"**',
        "#**design>@topic mention**",
        # Long conversations
        "#**design>Mark all messages as read.**",
        "#**feedback>issues link in description**",
        "#**design>Profile button**",
        # Extra long conversations
        "#**api documentation>prev_stream in message history**",
        "#**api design>Previewable URL Api**",
    ]
    for question in questions:
        content = get_answer(question)
        with open("output.log", "+a") as f:
            f.write(content + "\n")

# If you want to test directly:
# generate_answers()
# Run the summarizer as an interactive bot
watch_messages()
