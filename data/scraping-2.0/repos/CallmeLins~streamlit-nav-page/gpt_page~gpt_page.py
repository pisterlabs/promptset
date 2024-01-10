from gpt_page.libs.helper import *

def chatgpt():
    import os.path
    import streamlit as st
    import uuid
    import pandas as pd
    import openai
    from requests.models import ChunkedEncodingError
    from streamlit.components import v1
    from gpt_page.libs.voice_toolkit import voice_toolkit

    if "apibase" in st.secrets:
        openai.api_base = st.secrets["apibase"]
    else:
        openai.api_base = "https://api.openai.com/v1"

    # st.set_page_config(page_title="ChatGPT Assistant", layout="wide", page_icon="🤖")
    # custom css style
    st.markdown(css_code, unsafe_allow_html=True)

    if "initial_settings" not in st.session_state:
        # historical chat
        st.session_state["path"] = "history_chats_file"
        st.session_state["history_chats"] = get_history_chats(st.session_state["path"])
        # ss para init
        st.session_state["delete_dict"] = {}
        st.session_state["delete_count"] = 0
        st.session_state["voice_flag"] = ""
        st.session_state["user_voice_value"] = ""
        st.session_state["error_info"] = ""
        st.session_state["current_chat_index"] = 0
        st.session_state["user_input_content"] = ""
        # read global setting
        if os.path.exists("./set.json"):
            with open("./set.json", "r", encoding="utf-8") as f:
                data_set = json.load(f)
            for key, value in data_set.items():
                st.session_state[key] = value
        # init finish
        st.session_state["initial_settings"] = True

    with st.sidebar:
        st.markdown("__Chat Box__")
        # purpose of creating a container is to cooperate with the listening operation of custom components
        chat_container = st.container()
        with chat_container:
            current_chat = st.radio(
                label="History chat",
                format_func=lambda x: x.split("_")[0] if "_" in x else x,
                options=st.session_state["history_chats"],
                label_visibility="collapsed",
                index=st.session_state["current_chat_index"],
                key="current_chat"
                + st.session_state["history_chats"][st.session_state["current_chat_index"]],
                # on_change=current_chat_callback  # not suite for callback, can't recognize cwindows change
            )
        st.write("---")


    # write data to file
    def write_data(new_chat_name=current_chat):
        if "apikey" in st.secrets:
            st.session_state["paras"] = {
                "temperature": st.session_state["temperature" + current_chat],
                "top_p": st.session_state["top_p" + current_chat],
                "presence_penalty": st.session_state["presence_penalty" + current_chat],
                "frequency_penalty": st.session_state["frequency_penalty" + current_chat],
            }
            st.session_state["contexts"] = {
                "context_select": st.session_state["context_select" + current_chat],
                "context_input": st.session_state["context_input" + current_chat],
                "context_level": st.session_state["context_level" + current_chat],
            }
            save_data(
                st.session_state["path"],
                new_chat_name,
                st.session_state["history" + current_chat],
                st.session_state["paras"],
                st.session_state["contexts"],
            )


    def reset_chat_name_fun(chat_name):
        chat_name = chat_name + "_" + str(uuid.uuid4())
        new_name = filename_correction(chat_name)
        current_chat_index = st.session_state["history_chats"].index(current_chat)
        st.session_state["history_chats"][current_chat_index] = new_name
        st.session_state["current_chat_index"] = current_chat_index
        # write data
        write_data(new_name)
        # transfer data
        st.session_state["history" + new_name] = st.session_state["history" + current_chat]
        for item in [
            "context_select",
            "context_input",
            "context_level",
            *initial_content_all["paras"],
        ]:
            st.session_state[item + new_name + "value"] = st.session_state[
                item + current_chat + "value"
            ]
        remove_data(st.session_state["path"], current_chat)


    def create_chat_fun():
        st.session_state["history_chats"] = [
            "New Chat_" + str(uuid.uuid4())
        ] + st.session_state["history_chats"]
        st.session_state["current_chat_index"] = 0


    def delete_chat_fun():
        if len(st.session_state["history_chats"]) == 1:
            chat_init = "New Chat_" + str(uuid.uuid4())
            st.session_state["history_chats"].append(chat_init)
        pre_chat_index = st.session_state["history_chats"].index(current_chat)
        if pre_chat_index > 0:
            st.session_state["current_chat_index"] = (
                st.session_state["history_chats"].index(current_chat) - 1
            )
        else:
            st.session_state["current_chat_index"] = 0
        st.session_state["history_chats"].remove(current_chat)
        remove_data(st.session_state["path"], current_chat)


    with st.sidebar:
        c1, c2 = st.columns(2)
        create_chat_button = c1.button(
            "Create", use_container_width=True, key="create_chat_button"
        )
        if create_chat_button:
            create_chat_fun()
            st.rerun

        delete_chat_button = c2.button(
            "Delete", use_container_width=True, key="delete_chat_button"
        )
        if delete_chat_button:
            delete_chat_fun()
            st.rerun

    with st.sidebar:
        if ("set_chat_name" in st.session_state) and st.session_state[
            "set_chat_name"
        ] != "":
            reset_chat_name_fun(st.session_state["set_chat_name"])
            st.session_state["set_chat_name"] = ""
            st.rerun

        st.text_input("Set chat name:", key="set_chat_name", placeholder="Click input")
        st.selectbox(
            "Select module", index=0, options=["gpt-3.5-turbo", "gpt-4"], key="select_model"
        )
        st.caption(
            """
        - Double click page can locate to input box
        - Ctrl + Enter can quick submit input
        """
        )
        st.sidebar.markdown('''<small>[Simple Nav Page v1.0](https://github.com/CallmeLins/streamlit-nav-page)  | Aug 2023 | [CallmeLins](https://CallmeLins.github.io/)</small>''', unsafe_allow_html=True)

    # load history data
    if "history" + current_chat not in st.session_state:
        for key, value in load_data(st.session_state["path"], current_chat).items():
            if key == "history":
                st.session_state[key + current_chat] = value
            else:
                for k, v in value.items():
                    st.session_state[k + current_chat + "value"] = v

    # keep different chat in same layer, avoid rendering again
    container_show_messages = st.container()
    container_show_messages.write("")
    # show chat
    with container_show_messages:
        if st.session_state["history" + current_chat]:
            show_messages(current_chat, st.session_state["history" + current_chat])

    # monitor is it need del chat
    if any(st.session_state["delete_dict"].values()):
        for key, value in st.session_state["delete_dict"].items():
            try:
                deleteCount = value.get("deleteCount")
            except AttributeError:
                deleteCount = None
            if deleteCount == st.session_state["delete_count"]:
                delete_keys = key
                st.session_state["delete_count"] = deleteCount + 1
                delete_current_chat, idr = delete_keys.split(">")
                df_history_tem = pd.DataFrame(
                    st.session_state["history" + delete_current_chat]
                )
                df_history_tem.drop(
                    index=df_history_tem.query("role=='user'").iloc[[int(idr)], :].index,
                    inplace=True,
                )
                df_history_tem.drop(
                    index=df_history_tem.query("role=='assistant'")
                    .iloc[[int(idr)], :]
                    .index,
                    inplace=True,
                )
                st.session_state["history" + delete_current_chat] = df_history_tem.to_dict(
                    "records"
                )
                write_data()
                st.rerun


    def callback_fun(arg):
        # quick click create and delete button will call error callback, add judjement in here
        if ("history" + current_chat in st.session_state) and (
            "frequency_penalty" + current_chat in st.session_state
        ):
            write_data()
            st.session_state[arg + current_chat + "value"] = st.session_state[
                arg + current_chat
            ]


    def clear_button_callback():
        st.session_state["history" + current_chat] = []
        write_data()


    def delete_all_chat_button_callback():
        if "apikey" in st.secrets:
            folder_path = st.session_state["path"]
            file_list = os.listdir(folder_path)
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith(".json") and os.path.isfile(file_path):
                    os.remove(file_path)
        st.session_state["current_chat_index"] = 0
        st.session_state["history_chats"] = ["New Chat_" + str(uuid.uuid4())]


    def save_set(arg):
        st.session_state[arg + "_value"] = st.session_state[arg]
        if "apikey" in st.secrets:
            with open("./set.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "open_text_toolkit_value": st.session_state["open_text_toolkit"],
                        "open_voice_toolkit_value": st.session_state["open_voice_toolkit"],
                    },
                    f,
                )


    # show input content
    area_user_svg = st.empty()
    area_user_content = st.empty()
    # show replay
    area_gpt_svg = st.empty()
    area_gpt_content = st.empty()
    # show error info
    area_error = st.empty()

    st.write("\n")
    st.header("ChatGPT Assistant")
    tap_input, tap_context, tap_model, tab_func = st.tabs(
        ["💬 Chat", "🗒️ Prompt", "⚙️ Module", "🛠️ Function"]
    )

    with tap_context:
        set_context_list = list(set_context_all.keys())
        context_select_index = set_context_list.index(
            st.session_state["context_select" + current_chat + "value"]
        )
        st.selectbox(
            label="Select context",
            options=set_context_list,
            key="context_select" + current_chat,
            index=context_select_index,
            on_change=callback_fun,
            args=("context_select",),
        )
        st.caption(set_context_all[st.session_state["context_select" + current_chat]])

        st.text_area(
            label="Add or define context",
            key="context_input" + current_chat,
            value=st.session_state["context_input" + current_chat + "value"],
            on_change=callback_fun,
            args=("context_input",),
        )

    with tap_model:
        st.markdown("OpenAI API Key (option)")
        st.text_input(
            "OpenAI API Key (option)",
            type="password",
            key="apikey_input",
            label_visibility="collapsed",
        )
        st.caption(
            "This key is only valid on the current webpage, its prority higher than in config file. [Get from offical](https://platform.openai.com/account/api-keys)"
        )

        st.markdown("Including conversations count:")
        st.slider(
            "Context Level",
            0,
            10,
            st.session_state["context_level" + current_chat + "value"],
            1,
            on_change=callback_fun,
            key="context_level" + current_chat,
            args=("context_level",),
            help="The number of historical conversations included in each conversation, excluding preset content.",
        )

        st.markdown("Module parameter:")
        st.slider(
            "Temperature",
            0.0,
            2.0,
            st.session_state["temperature" + current_chat + "value"],
            0.1,
            help="""Higher value (0.8) will make output more random, lower value (0.2) will make more concentrated and deterministic
            Recommend changing only one between this parameter and top_p para, Do not change both of the p parameters at the same time.""",
            on_change=callback_fun,
            key="temperature" + current_chat,
            args=("temperature",),
        )
        st.slider(
            "Top P",
            0.1,
            1.0,
            st.session_state["top_p" + current_chat + "value"],
            0.1,
            help="""A method that replaces temperature sampling is called "core probability based sampling". 
            In this method, the model considers the predicted result with the highest probability for top_p markers.
            Therefore, when the parameter is 0.1, only markers including the top 10% probability mass will be considered.
            Recommend changing only one between this parameter and top_p para, Do not change both of the p parameters at the same time.""",
            on_change=callback_fun,
            key="top_p" + current_chat,
            args=("top_p",),
        )
        st.slider(
            "Presence Penalty",
            -2.0,
            2.0,
            st.session_state["presence_penalty" + current_chat + "value"],
            0.1,
            help="""Positive values will penalize new tags based on whether they appear in the currently generated text, thereby increasing the likelihood of the model discussing new topics.""",
            on_change=callback_fun,
            key="presence_penalty" + current_chat,
            args=("presence_penalty",),
        )
        st.slider(
            "Frequency Penalty",
            -2.0,
            2.0,
            st.session_state["frequency_penalty" + current_chat + "value"],
            0.1,
            help="""Positive values will penalize new tags based on whether they appear in the currently generated text, thereby decreasing the model generate same topics.""",
            on_change=callback_fun,
            key="frequency_penalty" + current_chat,
            args=("frequency_penalty",),
        )
        st.caption(
            "[Offical parameter induction](https://platform.openai.com/docs/api-reference/completions/create)"
        )

    with tab_func:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("Clear chat history", use_container_width=True, on_click=clear_button_callback)
        with c2:
            btn = st.download_button(
                label="Export chat records",
                data=download_history(st.session_state["history" + current_chat]),
                file_name=f'{current_chat.split("_")[0]}.md',
                mime="text/markdown",
                use_container_width=True,
            )
        with c3:
            st.button(
                "Delete all chat", use_container_width=True, on_click=delete_all_chat_button_callback
            )

        st.write("\n")
        st.markdown("Custom function:")
        c1, c2 = st.columns(2)
        with c1:
            if "open_text_toolkit_value" in st.session_state:
                default = st.session_state["open_text_toolkit_value"]
            else:
                default = True
            st.checkbox(
                "Enable text toolkit",
                value=default,
                key="open_text_toolkit",
                on_change=save_set,
                args=("open_text_toolkit",),
            )
        with c2:
            if "open_voice_toolkit_value" in st.session_state:
                default = st.session_state["open_voice_toolkit_value"]
            else:
                default = True
            st.checkbox(
                "Enable voice toolkit",
                value=default,
                key="open_voice_toolkit",
                on_change=save_set,
                args=("open_voice_toolkit",),
            )

    with tap_input:

        def input_callback():
            if st.session_state["user_input_area"] != "":
                # rename chat name
                user_input_content = st.session_state["user_input_area"]
                df_history = pd.DataFrame(st.session_state["history" + current_chat])
                if df_history.empty or len(df_history.query('role!="system"')) == 0:
                    new_name = extract_chars(user_input_content, 18)
                    reset_chat_name_fun(new_name)

        with st.form("input_form", clear_on_submit=True):
            user_input = st.text_area(
                "**Input:**",
                key="user_input_area",
                help="Content format as below can help GPT identification:"
                "\n- Code block use three backquotes and annotate the language type"
                "\n- Special character or regular expressions use quotation marks",
                value=st.session_state["user_voice_value"],
            )
            submitted = st.form_submit_button(
                "Confirm Submit", use_container_width=True, on_click=input_callback
            )
        if submitted:
            st.session_state["user_input_content"] = user_input
            st.session_state["user_voice_value"] = ""
            st.rerun

        if (
            "open_voice_toolkit_value" not in st.session_state
            or st.session_state["open_voice_toolkit_value"]
        ):
            # voice input toolkit
            vocie_result = voice_toolkit()
            # vocie_result will save latest result
            if (
                vocie_result and vocie_result["voice_result"]["flag"] == "interim"
            ) or st.session_state["voice_flag"] == "interim":
                st.session_state["voice_flag"] = "interim"
                st.session_state["user_voice_value"] = vocie_result["voice_result"]["value"]
                if vocie_result["voice_result"]["flag"] == "final":
                    st.session_state["voice_flag"] = "final"
                    st.rerun


    def get_model_input():
        # History to be inputted
        context_level = st.session_state["context_level" + current_chat]
        history = get_history_input(
            st.session_state["history" + current_chat], context_level
        ) + [{"role": "user", "content": st.session_state["pre_user_input_content"]}]
        for ctx in [
            st.session_state["context_input" + current_chat],
            set_context_all[st.session_state["context_select" + current_chat]],
        ]:
            if ctx != "":
                history = [{"role": "system", "content": ctx}] + history
        # module para to be set
        paras = {
            "temperature": st.session_state["temperature" + current_chat],
            "top_p": st.session_state["top_p" + current_chat],
            "presence_penalty": st.session_state["presence_penalty" + current_chat],
            "frequency_penalty": st.session_state["frequency_penalty" + current_chat],
        }
        return history, paras


    if st.session_state["user_input_content"] != "":
        if "r" in st.session_state:
            st.session_state.pop("r")
            st.session_state[current_chat + "report"] = ""
        st.session_state["pre_user_input_content"] = st.session_state["user_input_content"]
        st.session_state["user_input_content"] = ""
        # temporary display
        show_each_message(
            st.session_state["pre_user_input_content"],
            "user",
            "tem",
            [area_user_svg.markdown, area_user_content.markdown],
        )
        # module input
        history_need_input, paras_need_input = get_model_input()
        # call Interface
        with st.spinner("🤔"):
            try:
                if apikey := st.session_state["apikey_input"]:
                    openai.api_key = apikey
                # configure temporary apikey, which will not retain chat records and is suitable for public use
                elif "apikey_tem" in st.secrets:
                    openai.api_key = st.secrets["apikey_tem"]
                # note: When apikey is configured in st.secrets, chat records will be retained even if this apikey is not used
                else:
                    openai.api_key = st.secrets["apikey"]
                r = openai.ChatCompletion.create(
                    model=st.session_state["select_model"],
                    messages=history_need_input,
                    stream=True,
                    **paras_need_input,
                )
            except (FileNotFoundError, KeyError):
                area_error.error(
                    "Missing OpenAI API Key, please config Secrets, or conifg it in web page. "
                    "Detail[Repo](https://github.com/CallmeLins/streamlit-nav-page/blob/main/gpt_page/README.md)。"
                )
            except openai.error.AuthenticationError:
                area_error.error("Invalid OpenAI API Key.")
            except openai.error.APIConnectionError as e:
                area_error.error("Connect timeout, please try again. errot msg: \n" + str(e.args[0]))
            except openai.error.InvalidRequestError as e:
                area_error.error("Invalid request, please try again. errot msg: \n" + str(e.args[0]))
            except openai.error.RateLimitError as e:
                area_error.error("RateLimit, errot msg: \n" + str(e.args[0]))
            else:
                st.session_state["chat_of_r"] = current_chat
                st.session_state["r"] = r
                st.rerun

    if ("r" in st.session_state) and (current_chat == st.session_state["chat_of_r"]):
        if current_chat + "report" not in st.session_state:
            st.session_state[current_chat + "report"] = ""
        try:
            for e in st.session_state["r"]:
                if "content" in e["choices"][0]["delta"]:
                    st.session_state[current_chat + "report"] += e["choices"][0]["delta"][
                        "content"
                    ]
                    show_each_message(
                        st.session_state["pre_user_input_content"],
                        "user",
                        "tem",
                        [area_user_svg.markdown, area_user_content.markdown],
                    )
                    show_each_message(
                        st.session_state[current_chat + "report"],
                        "assistant",
                        "tem",
                        [area_gpt_svg.markdown, area_gpt_content.markdown],
                    )
        except ChunkedEncodingError:
            area_error.error("Network poor, please refresh the page and try again.")
        # for situtation sop 
        except Exception:
            pass
        else:
            # save content
            st.session_state["history" + current_chat].append(
                {"role": "user", "content": st.session_state["pre_user_input_content"]}
            )
            st.session_state["history" + current_chat].append(
                {"role": "assistant", "content": st.session_state[current_chat + "report"]}
            )
            write_data()
        # when a user clicks stop on a webpage, ss may temporarily be empty in some cases
        if current_chat + "report" in st.session_state:
            st.session_state.pop(current_chat + "report")
        if "r" in st.session_state:
            st.session_state.pop("r")
            st.rerun

    # add event monitor
    v1.html(js_code, height=0)
