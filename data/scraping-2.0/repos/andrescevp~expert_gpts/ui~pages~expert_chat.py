import datetime
import json
import uuid
from dataclasses import asdict

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, dcc, html
from langchain.schema.messages import AIMessage, HumanMessage

from expert_gpts.llms.chat_managers import get_history
from expert_gpts.main import LLMConfigBuilder
from ui.components.chat import CHAT_COMPONENT_TEMPLATE
from ui.components.chat_history import create_chat_list
from ui.components.chat_messages import get_system_chat_item, get_user_chat_item
from ui.config import get_configs
from ui.pages.states.web_chat_states import WebChatPageState
from ui.utils.chats import get_chats_list

dash.register_page(__name__, path_template="/expert_chat/<config_key>")

configurations = get_configs()


def layout(config_key=None):
    config = configurations[config_key].config
    SESSION_ID = str(uuid.uuid4())
    chats_uuids = get_chats_list(config.chain.chain_key, SESSION_ID)
    experts_list = list(config.experts.__root__.keys())
    return html.Div(
        children=[
            html.H1(
                f"{configurations[config_key].config.chain.chain_key} Chain Chat",
                id="chat_title",
                className="p-2 mx-auto text-center",
            ),
            dcc.Store(id="session_chat_to_remove", data=None),
            dcc.Store(id="chat_history", data=chats_uuids),
            dcc.Store(id="config_key", data=config_key),
            dcc.Store(id="web-chat-page-memory"),
            dcc.Store(id="session", data=dict(uid=SESSION_ID)),
            dcc.Location(id="url"),
            html.Span("chain", id="current_expert", className="d-none"),
            dbc.Row(
                children=[
                    dbc.Col(
                        children=[
                            html.H4(
                                "Available chats", className="p-2 mx-auto text-center"
                            ),
                            dbc.ListGroup(
                                [
                                    dbc.ListGroupItem(
                                        dbc.Button(
                                            f"{configurations[config_key].config.chain.chain_key} Chain Chat",
                                            id={
                                                "type": "btn-expert-init",
                                                "index": "chain",
                                            },
                                            className="chain text-break",
                                            value="chain",
                                            color="success",
                                            style={"width": "100%"},
                                        )
                                    ),
                                    dbc.ListGroupItem(
                                        dbc.Button(
                                            f"{configurations[config_key].config.planner.chain_key} Plan And Execute",
                                            id={
                                                "type": "btn-expert-init",
                                                "index": "planner",
                                            },
                                            className="planner text-break",
                                            value="planner",
                                            color="primary",
                                            style={"width": "100%"},
                                        )
                                    ),
                                ]
                                + [
                                    dbc.ListGroupItem(
                                        dbc.Button(
                                            f"{expert} Chat",
                                            id={
                                                "type": "btn-expert-init",
                                                "index": expert,
                                            },
                                            className=f"{expert} text-break",
                                            value=expert,
                                            color="secondary",
                                        )
                                    )
                                    for expert in experts_list
                                ]
                            ),
                        ],
                        id="agents-container",
                        className="col-2",
                    ),
                    dbc.Col(
                        children=CHAT_COMPONENT_TEMPLATE,
                        className="col-8",
                    ),
                    dbc.Col(
                        children=[
                            html.H4(
                                "Chats History", className="p-2 mx-auto text-center"
                            ),
                            dbc.ListGroup(
                                create_chat_list(chats_uuids),
                                id="chats-message-history",
                            ),
                        ],
                        className="col-2",
                    ),
                ],
                id="chat-container",
                className="w-100 h-100 mx-0",
            ),
        ],
        className="h-100 mx-0 w-100",
    )


@dash.callback(
    Output("chat-history", "children", allow_duplicate=True),
    Output("last-message-time", "children", allow_duplicate=True),
    Output("web-chat-page-memory", "data", allow_duplicate=True),
    Output("user-prompt", "value", allow_duplicate=True),
    Output("user-prompt", "readonly", allow_duplicate=True),
    Output("send-button", "disabled", allow_duplicate=True),
    Output("send-button", "children", allow_duplicate=True),
    Input("current_expert", "children"),
    Input("last-message-time", "children"),
    Input("chat-history", "children"),
    Input("send-button", "n_clicks"),
    Input("send-button", "n_clicks_timestamp"),
    Input("user-prompt", "value"),
    prevent_initial_call=True,
)
def add_chat_item(
    current_expert, last_send, children, n_clicks, n_clicks_timestamp, user_prompt
):
    if not n_clicks or not current_expert:
        return dash.no_update
    # check if user actually hit the button
    if n_clicks_timestamp and last_send and int(n_clicks_timestamp) <= int(last_send):
        return dash.no_update

    message_sent_at = datetime.datetime.fromtimestamp(
        n_clicks_timestamp / 1000
    ).strftime("%Y-%m-%d %H:%M:%S")
    if user_prompt:
        chats = [get_user_chat_item(user_prompt, message_sent_at)]
        default_memory_state = asdict(
            WebChatPageState(
                current_expert=current_expert, current_user_prompt=user_prompt
            )
        )
        if not children:
            return [
                chats,
                n_clicks_timestamp,
                default_memory_state,
                "",
                "readonly",
                "disabled",
                [dbc.Spinner()],
            ]
        return [
            children + chats,
            n_clicks_timestamp,
            default_memory_state,
            "",
            "readonly",
            "disabled",
            [dbc.Spinner()],
        ]
    else:
        return dash.no_update


@dash.callback(
    Output("chat-history", "children", allow_duplicate=True),
    Output("web-chat-page-memory", "data", allow_duplicate=True),
    Output("user-prompt", "readonly", allow_duplicate=True),
    Output("send-button", "disabled", allow_duplicate=True),
    Output("send-button", "children", allow_duplicate=True),
    Output("terminal", "children", allow_duplicate=True),
    Input("web-chat-page-memory", "data"),
    State("session", "data"),
    Input("config_key", "data"),
    Input("chat-history", "children"),
    prevent_initial_call=True,
)
def get_answer(data, session, config_key, chats_prevs):
    if not data:
        return dash.no_update
    config = configurations[config_key].config
    builder = LLMConfigBuilder(config)
    data = WebChatPageState(**data)
    if data.answer:
        return dash.no_update
    if data.current_expert not in ["chain", "planner"]:
        expert_chat = builder.get_expert_chat(data.current_expert, session["uid"])
        answer = expert_chat.ask(data.current_user_prompt)
        log = expert_chat.get_log()
    elif data.current_expert == "chain":
        chain_chat = builder.get_chain_chat(session["uid"])
        answer = chain_chat.ask(data.current_user_prompt)
        log = chain_chat.get_log()
    elif data.current_expert == "planner":
        chain_chat = builder.get_planner(session["uid"])
        answer = chain_chat.ask(data.current_user_prompt)
        log = chain_chat.get_log()
    else:
        raise ValueError(f"Unknown expert {data.current_expert}")
    chats = [get_system_chat_item(answer)]
    return [
        chats_prevs + chats,
        asdict(
            WebChatPageState(
                current_expert=data.current_expert,
                current_user_prompt=data.current_user_prompt,
                answer=answer,
            )
        ),
        False,
        False,
        "Send",
        json.dumps(log, indent=4, sort_keys=True, default=str),
    ]


@dash.callback(
    Output("chat-history", "children", allow_duplicate=True),
    Output("current_expert", "children"),
    Output("chat_title", "children"),
    Output("chats-message-history", "children"),
    State("session", "data"),
    Input("config_key", "data"),
    Input({"type": "btn-expert-init", "index": ALL}, "n_clicks_timestamp"),
    Input({"type": "btn-expert-init", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def get_expert_chat(session_id, config_key, n_clicks, value):
    config = configurations[config_key].config
    valued_n_clicks = [
        (n_click, val) for n_click, val in zip(n_clicks, value) if n_click
    ]
    last_clicked = sorted(valued_n_clicks, key=lambda x: x[0])[-1][1]
    if last_clicked == "chain":
        chats_uuids = get_chats_list(config.chain.chain_key, session_id["uid"])
        return [
            [
                get_system_chat_item(
                    "Hello I am a super assistant! I am able to connect to others experts and tools in "
                    "this app to "
                    "help you."
                )
            ],
            last_clicked,
            f"{configurations[config_key].config.chain.chain_key} Chain",
            create_chat_list(chats_uuids),
        ]
    if last_clicked == "planner":
        chats_uuids = get_chats_list(config.planner.chain_key, session_id["uid"])
        return [
            [
                get_system_chat_item(
                    """
                    Hello! I can use all the tools I have to help you to trace a plan to achieve your goal.
                    """
                )
            ],
            last_clicked,
            f"{configurations[config_key].config.planner.chain_key} Planner",
            create_chat_list(chats_uuids),
        ]
    chats_uuids = get_chats_list(last_clicked, session_id["uid"])
    return [
        [get_system_chat_item(f"Hello I am {last_clicked} assistant!")],
        last_clicked,
        config.experts.__root__[last_clicked].name or last_clicked,
        create_chat_list(chats_uuids),
    ]


# on click download-log download the log from terminal
@dash.callback(
    Output("download-terminal-log", "data"),
    Output("last-log-download", "data"),
    Input("last-log-download", "data"),
    Input("download-log", "n_clicks_timestamp"),
    Input("download-log", "n_clicks"),
    Input("terminal", "children"),
    prevent_initial_call=True,
)
def download_log(last_download_at, n_clicks, n_clicks_timestamp, log):
    if not log or not n_clicks:
        return dash.no_update
    if (
        n_clicks_timestamp
        and last_download_at
        and int(n_clicks_timestamp) <= int(last_download_at)
    ):
        return dash.no_update
    return [dict(content=log, filename="log.json"), n_clicks_timestamp]


@dash.callback(
    Output("chats-message-history", "children", allow_duplicate=True),
    Output("session_chat_to_remove", "data", allow_duplicate=True),
    State("session", "data"),
    Input("config_key", "data"),
    Input("current_expert", "children"),
    Input("session_chat_to_remove", "data"),
    Input("btn-chat-session-remove-confirm", "submit_n_clicks"),
    prevent_initial_call=True,
)
def remove_chat_session(
    session_id, config_key, current_expert, session_chat_to_remove, submit_n_clicks
):
    if not submit_n_clicks:
        return dash.no_update
    config = configurations[config_key].config
    ai_key = current_expert if current_expert != "chain" else config.chain.chain_key

    chat_history = get_history(session_id["uid"], ai_key)
    chat_history.delete_chat_session(session_chat_to_remove)
    chats_uuids = get_chats_list(ai_key, session_id["uid"])
    return [create_chat_list(chats_uuids), None]


@dash.callback(
    Output("btn-chat-session-remove-confirm", "displayed"),
    Output("session_chat_to_remove", "data", allow_duplicate=True),
    Input({"type": "btn-chat-session-remove", "index": ALL}, "n_clicks_timestamp"),
    Input({"type": "btn-chat-session-remove", "index": ALL}, "value"),
    config_prevent_initial_callbacks=True,
)
def display_confirm_remove_chat(n_clicks, value):
    all_none = [x is None for x in n_clicks]
    if all(all_none):
        return dash.no_update, dash.no_update
    valued_n_clicks = [
        (n_click, val) for n_click, val in zip(n_clicks, value) if n_click
    ]
    last_clicked = sorted(valued_n_clicks, key=lambda x: x[0])[-1][1]
    return [True, last_clicked]


@dash.callback(
    Output("chat-history", "children", allow_duplicate=True),
    Output("session", "data"),
    Input("config_key", "data"),
    Input("current_expert", "children"),
    Input({"type": "btn-chat-session-load", "index": ALL}, "n_clicks_timestamp"),
    Input({"type": "btn-chat-session-load", "index": ALL}, "value"),
    config_prevent_initial_callbacks=True,
)
def load_chat(config_key, current_expert, n_clicks, value):
    all_none = [x is None for x in n_clicks]
    if all(all_none):
        return dash.no_update, dash.no_update
    config = configurations[config_key].config
    ai_key = current_expert if current_expert != "chain" else config.chain.chain_key
    valued_n_clicks = [
        (n_click, val) for n_click, val in zip(n_clicks, value) if n_click
    ]
    last_clicked = sorted(valued_n_clicks, key=lambda x: x[0])[-1][1]

    chat_history = get_history(last_clicked, ai_key)
    messages = []
    for message in chat_history.raw_messages:
        if type(message["message"]) == HumanMessage:
            messages.append(
                get_user_chat_item(message["message"].content, message["created_at"])
            )
        elif type(message["message"]) == AIMessage:
            messages.append(get_system_chat_item(message["message"].content))

    return [messages, {"uid": last_clicked}]
