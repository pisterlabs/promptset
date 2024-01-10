import openai
import streamlit as st
import os

from travel_chatbot.agents import run_francis, bq_run_francis
from travel_chatbot.tools import get_tools
from travel_chatbot.basemodels import TravelDetails
from travel_chatbot.utils import conversation_history
from langchain.callbacks.manager  import collect_runs, trace_as_chain_group
from travel_chatbot.utils import submit_feedback

from langsmith import Client

from streamlit_feedback import streamlit_feedback

client = Client()

openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

def move_focus():
    # inspect the html to determine which control to specify to receive focus (e.g. text or textarea).
    st.components.v1.html(
        f"""
            <script>
                var textarea = window.parent.document.querySelectorAll("textarea[type=textarea]");
                for (var i = 0; i < textarea.length; ++i) {{
                    textarea[i].focus();
                }}
            </script>
        """,
    )


def stick_it_good():

    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
            </style>
        """,
        unsafe_allow_html=True
    )


def main():
    # set model as session state
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # set tools as session state - work to be done
    if "tools" not in st.session_state:
        st.session_state["tools"] = get_tools()

    # user_travel_details as session state
    if "user_travel_details" not in st.session_state:
        st.session_state["user_travel_details"] = TravelDetails(introduction=False,
                                                                    qualification="",
                                                                    country="",
                                                                    departing_after=None,
                                                                    departing_before=None,
                                                                    max_budget=None,
                                                                    max_duration=None,
                                                                    min_duration=None,
                                                                    )

    # list_of_interests as session state
    if "list_of_interests" not in st.session_state:
        st.session_state.list_of_interests = []

    # interest_asked as session state
    if "interest_asked" not in st.session_state:
        st.session_state.interest_asked = []

    # solution_presented as session state
    if "solution_presented" not in st.session_state:
        st.session_state.solution_presented = []

    # asked_for as session state
    if "asked_for" not in st.session_state:
        st.session_state.asked_for = []

    # create message variable a session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.container():
        st.title("Francis: The Travel Agent Bot")
        stick_it_good()

    # first message from francis to iniciate conversation
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({"role": "assistant", "content": "Hello, this is Francis your personal travel chat bot. How can i help you today?"})

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ongoing conversation
    if user_content := st.chat_input("Type your question here."): # using streamlit's st.chat_input because it stays put at bottom, chat.openai.com style.
            st.session_state.messages.append({"role": "user", "content": user_content})
            with st.chat_message("user"):
                st.markdown(user_content)
    run_id = None
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                with collect_runs() as cb:
                    conversation = conversation_history(st.session_state.messages)
                    assistant_content, user_details = bq_run_francis(user_content,
                                            conversation,
                                            st.session_state["user_travel_details"],
                                            st.session_state.list_of_interests,
                                            st.session_state.interest_asked,
                                            st.session_state["tools"],
                                            st.session_state.asked_for,
                                            st.session_state.solution_presented)
                    st.write(assistant_content)
                    print(len(cb.traced_runs))
                    run_id = cb.traced_runs[-1].id
                    st.session_state.run_id = run_id
        message = {"role": "assistant", "content": assistant_content}
        st.session_state.messages.append(message)
        st.session_state["user_travel_details"] = user_details

    feedback_option = 'thumbs' # or 'faces'
    if st.session_state.get("run_id"):
        feedback = streamlit_feedback(
            feedback_type=feedback_option,  # Use the selected feedback option
            optional_text_label="[Optional] Please provide an explanation",  # Adding a label for optional text input
            key=f"feedback_{st.session_state.run_id}",
        )

        # Define score mappings for both "thumbs" and "faces" feedback systems
        score_mappings = {
            "thumbs": {"👍": 1, "👎": 0},
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        # Get the score mapping based on the selected feedback option
        scores = score_mappings[feedback_option]

        if feedback:
            # Get the score from the selected feedback option's score mapping
            score = scores.get(feedback["score"])

            if score is not None:
                # Formulate feedback type string incorporating the feedback option and score value
                feedback_type_str = f"{feedback_option} {feedback['score']}"

                # Record the feedback with the formulated feedback type string and optional comment
                feedback_record = client.create_feedback(
                    st.session_state.run_id,
                    feedback_type_str,  # Updated feedback type
                    score=score,
                    comment=feedback.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Invalid feedback score.")

    # create sidebar for intro text and the open to clear chat history
    st.sidebar.write("Welcome to Francis, your personal travel chat bot.")
    st.sidebar.write("Francis is here to have a conversation with you, understanding your basic travel needs such as your destination and budget. Additionally, he'll inquire about your interests to provide you with tailored group tour recommendations.")
    st.sidebar.write("If you have any issues or want to start again you can clear the conversation with the button below.")

    if st.sidebar.button("Clear Conversation", key='clear_chat_button'):
        st.session_state.messages = []
        st.session_state["user_travel_details"] = TravelDetails(introduction=False,
                                                                    qualification="",
                                                                    country="",
                                                                    departing_after=None,
                                                                    departing_before=None,
                                                                    max_budget=None,
                                                                    max_duration=None,
                                                                    min_duration=None,
                                                                    )
        st.session_state.list_of_interests = []
        st.session_state.interest_asked = []
        st.session_state.asked_for = []
        move_focus()

    st.sidebar.write(st.session_state["user_travel_details"].dict())
    st.sidebar.write(st.session_state.list_of_interests)
    st.sidebar.write(st.session_state.interest_asked)
    st.sidebar.write(st.session_state.asked_for)
    st.sidebar.write(st.session_state.solution_presented)

if __name__ == '__main__':
    main()
