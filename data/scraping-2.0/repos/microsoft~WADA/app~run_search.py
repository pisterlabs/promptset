# Copyright © Microsoft Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr
import openai
import openai.error
import tenacity
from utils import save_markdown_file, split_markdown_code

from wada.agents import TopicAgent
from wada.debate_simulator import DebateSimulator
from wada.topic import Topic
from wada.typing import ModelType, TopicType

ChatBotHistory = List[Tuple[Optional[str], Optional[str]]]

DEFAULT_TOPIC = "As a new graduate, should I start my career in my hometown Chengdu or the capital city Beijing?"


@dataclass
class State:
    topic_agent: Optional[TopicAgent]
    debate: Optional[DebateSimulator]
    chat: ChatBotHistory
    debate_history: ChatBotHistory
    ready_for_debate: bool

    @classmethod
    def empty(cls) -> 'State':
        return cls(None, None, [], [], False)

    @staticmethod
    def construct_inplace(state: 'State', topic_agent: Optional[TopicAgent],
                          session: Optional[DebateSimulator],
                          chat: ChatBotHistory, debate_history: ChatBotHistory,
                          ready_for_debate: bool = False):
        state.topic_agent = topic_agent
        state.debate = session
        state.chat = chat
        state.debate_history = debate_history
        state.ready_for_debate = ready_for_debate

    @staticmethod
    def export(state: 'State', catagory: str):
        topic_abbr = state.topic_agent.abbreviate_topic()
        result = {
            'model': state.topic_agent.model.value,
            'topic': state.topic_agent.topic.content,
            'topic_pro': state.topic_agent.topic.pro,
            'topic_con': state.topic_agent.topic.con,
            'topic_abbr':
            topic_abbr if topic_abbr else state.topic_agent.topic.content,
            'catagory': catagory,
            'specified_aspects': state.topic_agent.topic.specified_aspects,
            'background': state.topic_agent.topic.background,
            'chat_history': state.chat,
            'preference': state.topic_agent.topic.preference,
            'debate_history': state.debate.history,
            'judgement': state.debate.host.judgement,
        }
        return result


def parse_arguments():
    parser = argparse.ArgumentParser("WADA data explorer")
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key')
    parser.add_argument('--share', type=bool, default=False,
                        help='Expose the web UI to Gradio')
    parser.add_argument('--server-port', type=int, default=8080,
                        help='Port ot run the web page on')
    parser.add_argument('--inbrowser', type=bool, default=False,
                        help='Open the web UI in the default browser on lunch')
    parser.add_argument(
        '--concurrency-count', type=int, default=1,
        help='Number if concurrent threads at Gradio websocket queue. ' +
        'Increase to serve more requests but keep an eye on RAM usage.')
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print("Unknown args: ", unknown)
    return args


def cleanup_on_launch(state) -> Tuple[State, Dict, Dict, Dict, Dict]:
    State.construct_inplace(state, None, None, [], [], False)
    return state, gr.update(interactive=False), gr.update(
        interactive=False), gr.update(visible=True), gr.update(
            interactive=False)


def specify_topic(
    state: State,
    topic: str,
    model_type: str,
) -> Union[Dict, Tuple[State, str, Dict]]:

    try:
        model = ModelType.GPT_3_5_TURBO if model_type == "GPT 3.5" else ModelType.GPT_4
        topic_agent = TopicAgent(topic=Topic(content=topic), model=model)
        (pos, neg) = topic_agent.break_down_topic()
        print(pos, neg)
        specified_aspects = topic_agent.specify_topic()
        state.topic_agent = topic_agent

    except (openai.error.RateLimitError, tenacity.RetryError,
            RuntimeError) as ex:
        print("OpenAI API exception 0 " + str(ex))
        return (state, "", str(ex))

    return (state, specified_aspects, gr.update(visible=True))


def collect_bg(
    state: State
) -> Union[Dict, Tuple[State, str, ChatBotHistory, Dict, Dict]]:

    try:
        background = state.topic_agent.collect_bg()
        (is_summary, content) = state.topic_agent.collect_pref()
        if is_summary:
            raise RuntimeError("Not a question")
        state.chat.append((None, split_markdown_code(content)))
    except (openai.error.RateLimitError, tenacity.RetryError,
            RuntimeError) as ex:
        print("OpenAI API exception 0 " + str(ex))
        return state, str(ex), [], gr.update(), gr.update(visible=True)

    return state, background, gr.update(
        value=state.chat,
        visible=True), gr.update(visible=True), gr.update(interactive=True)


def submit_new_msg(
        state: State,
        reply: str) -> Union[Dict, Tuple[State, ChatBotHistory, Dict]]:

    state.chat.append((split_markdown_code(reply), None))
    return state, state.chat, gr.update(interactive=False)


def send_new_msg(
    state: State, reply: str
) -> Union[Dict, Tuple[State, Union[Dict, str], Union[Dict, str],
                       ChatBotHistory]]:

    try:
        (is_summary, content) = state.topic_agent.collect_pref(reply)
        if not is_summary:
            state.chat.append((None, split_markdown_code(content)))
            return [state, "", "", state.chat]
        else:
            state.ready_for_debate = True
            return (state, gr.update(visible=False),
                    gr.update(value=content, visible=True), state.chat)

    except (openai.error.RateLimitError, tenacity.RetryError,
            RuntimeError) as ex:
        print("OpenAI API exception 0 " + str(ex))
        return (state, str(ex), "", [])


def start_debate(
    state: State
) -> Union[Dict, Tuple[State, ChatBotHistory, str, Dict, Dict, Dict]]:
    try:
        if not state.ready_for_debate:
            return state, [], "", gr.update(
                interactive=True), gr.update(), gr.update()

        debate_session = DebateSimulator(
            topic=state.topic_agent.topic,
            verbose=True,
        )

        intro_a = f"I am {debate_session.debater_a_name}, I will be arguing for: {state.topic_agent.topic.pro}"
        intro_b = f"I am {debate_session.debater_b_name}, I will be arguing for: {state.topic_agent.topic.con}"

        state.debate_history.append((split_markdown_code(intro_a), None))
        state.debate_history.append((None, split_markdown_code(intro_b)))

        state.debate = debate_session
        debate_session.reset()

        judge_result = ""

        while debate_session.terminated is False:
            debater_a_reply, debater_b_reply, judge_result = debate_session.step(
            )
            state.debate_history.append(
                (split_markdown_code(debater_a_reply), None))
            state.debate_history.append(
                (None, split_markdown_code(debater_b_reply)))

        return state, state.debate_history, gr.update(
            value=judge_result,
            visible=True), gr.update(interactive=True), gr.update(
                visible=True), gr.update(visible=True)

    except (openai.error.RateLimitError, tenacity.RetryError,
            RuntimeError) as ex:
        print("OpenAI API exception 0 " + str(ex))
        return state, [], "", gr.update(
            interactive=True), gr.update(), gr.update()


def update_state(state: State) -> Union[Dict, Tuple[State, ChatBotHistory]]:
    debate_cb_dict = dict()
    if state.ready_for_debate:
        debate_cb_dict['visible'] = True

    if state.debate_history is not None:
        debate_cb_dict['value'] = state.debate_history

    return (state, gr.update(**debate_cb_dict))


def change_topic(topic: str) -> Dict:
    if topic.strip() == "":
        return gr.update(interactive=False)
    else:
        return gr.update(interactive=True)


def reset_page(state):
    State.construct_inplace(state, None, None, [], [], False)
    return (
        state,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value=DEFAULT_TOPIC),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value=[], visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value=[], visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def save(state: State, catagory: str) -> Tuple[State, Dict]:
    result = State.export(state, catagory)
    save_markdown_file(result)
    return state, gr.update(visible=True)


def main():
    args = parse_arguments()

    print("Getting Agents web server online...")

    demo.queue(args.concurrency_count) \
        .launch(inbrowser=args.inbrowser,
                server_name="127.0.0.1", server_port=args.server_port,
                debug=True)

    print("Exiting.")


css_str = "#start_button {border: 3px solid #4CAF50; font-size: 20px;}"

with gr.Blocks(css=css_str, title="WADA") as demo:

    gr.Markdown(
        "# WADA \n Wise AI Debate Assistant. Click here to view the [Case Book](https://wada-1.gitbook.io/wada-casebook/)."
    )

    with gr.Row():
        with gr.Column(scale=4):
            model_type_dd = gr.Dropdown(["GPT 3.5", "GPT 4"], value="GPT 4",
                                        label="Model Type", interactive=True)

        with gr.Column(scale=4):
            start_bn = gr.Button("Start", elem_id="start_button")

            reset_bn = gr.Button("Reset", elem_id="reset_button")

    topic_ta = gr.TextArea(label="Give me a topic here", value=DEFAULT_TOPIC,
                           lines=1, interactive=True)

    topic_ta.change(change_topic, topic_ta, start_bn, queue=False)

    specified_aspects_ta = gr.TextArea(
        label="Specified aspects of information of the topic", lines=1,
        interactive=False, visible=False)

    topic_background_ta = gr.TextArea(label="Generated background information",
                                      lines=1, interactive=False,
                                      visible=False)

    interact_cb = gr.Chatbot(label="Chat between you and the agent",
                             visible=False)

    reply_tb = gr.Textbox(show_label=False,
                          placeholder="Enter text and press enter",
                          visible=False, container=False)

    topic_preference_ta = gr.TextArea(label="Preference summary", lines=1,
                                      interactive=False, visible=False)

    debate_cb = gr.Chatbot(label="Chat between autonomous debaters",
                           visible=False)

    decision_ta = gr.TextArea(label="Suggestion", lines=1, interactive=False,
                              visible=False)

    with gr.Row():
        with gr.Column(scale=4):
            topic_catagory_dd = gr.Dropdown(
                [topic.value for topic in TopicType],
                value=TopicType.CAREER_EDUCATION, label="Topic Catagory",
                visible=False)

        with gr.Column(scale=4):
            save_bn = gr.Button("Save Result", elem_id="save_button",
                                visible=False)

    saved_hint = gr.Markdown("File Saved.", visible=False)

    state = gr.State(State.empty())

    save_bn.click(save, [state, topic_catagory_dd], [state, saved_hint],
                  queue=False)

    start_bn.click(cleanup_on_launch, state,
                   [state, model_type_dd, start_bn, specified_aspects_ta, reset_bn],
                   queue=False) \
        .then(specify_topic,
              [state, topic_ta, topic_catagory_dd],
              [state, specified_aspects_ta, topic_background_ta],
              queue=False) \
        .then(collect_bg, state,
              [state, topic_background_ta, interact_cb, reply_tb, reset_bn],
              queue=False)

    reply_tb.submit(submit_new_msg,
                    [state, reply_tb],
                    [state, interact_cb, reset_bn]) \
        .then(send_new_msg,
              [state, reply_tb],
              [state, reply_tb, topic_preference_ta, interact_cb],
              queue=False) \
        .then(start_debate,
              state,
              [state, debate_cb, decision_ta, reset_bn, save_bn, topic_catagory_dd],
              queue=False)

    demo.load(update_state, state, [state, debate_cb], every=0.5)

    reset_bn.click(
        reset_page, state, outputs=[
            state, model_type_dd, start_bn, topic_ta, specified_aspects_ta,
            topic_background_ta, interact_cb, reply_tb, topic_preference_ta,
            debate_cb, decision_ta, save_bn, topic_catagory_dd, saved_hint
        ], queue=False)

if __name__ == "__main__":

    main()
