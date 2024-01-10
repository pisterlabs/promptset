import streamlit as st
import openai
from ai_main import load_index, answer_question, clear_chat_history  # Import clear_chat_history function
import json
import pinecone
import pandas as pd
from google.oauth2 import service_account
from google.cloud import storage
from datetime import datetime


pinecone_api_key = st.secrets["PINECONE_API_KEY_Bricks"]
pinecone_environment = st.secrets["PINECONE_environment_Bricks"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
openai.api_key = openai_api_key

index = load_index()
updated_stream = ""

# Create API client for Google Cloud Storage
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)

# Bucket name and file path
bucket_name = "bucket_g_cloud_service_1"


def load_from_gcs(bucket_name, file_path):
    """Load a file from Google Cloud Storage."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    content = blob.download_as_text()  # or use download_as_bytes for binary files
    return json.loads(content)


def save_to_gcs(bucket_name, file_path, content):
    """Save a file to Google Cloud Storage."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    content = json.dumps(content)
    blob.upload_from_string(content, content_type="application/json")


def save_feedback_to_json(feedback):
    """Save the feedback to a JSON file."""
    with open("feedback.json", "a") as file:
        file.write(json.dumps(feedback) + "\n")


unique_categories_file_path = "bricks/unique_categories.json"
instructions_file_path = "bricks/instructions.json"
feedback_data_file_path = "bricks/feedback.json"
# Load unique categories from GCS
unique_categories = load_from_gcs(bucket_name, unique_categories_file_path)

# Load instructions from GCS
instructions = load_from_gcs(bucket_name, instructions_file_path)

# Load feedback data from GCS
feedback_data = load_from_gcs(bucket_name, feedback_data_file_path)


def app():
    st.set_page_config(layout="wide")
    if "show_feedback_form" not in st.session_state:
        st.session_state.show_feedback_form = False
    st.title("Bricks Ai - Support Tool")

    tabs = st.tabs(["Main", "Log"])
    main_tab, log_tab = tabs

    with main_tab:
        st.sidebar.header("How it Works")
        st.sidebar.write(
            "This AI Assistant uses GPT-4 to answer questions based on a chosen set of instructions and categories. Customize GPT-4 parameters and select categories to refine the AI's responses. Send feedback to help us improve."
        )

        # Add a separator
        st.sidebar.markdown(
            "<hr style='height: 1px; border: none; background-color: gray; margin-left: -10px; margin-right: -10px;'>",
            unsafe_allow_html=True,
        )

        # GPT parameter fields

        st.sidebar.subheader("GPT Parameters")
        # Add GPT model selection buttons
        model_col1, model_col2 = st.sidebar.columns(2)
        with model_col1:
            gpt_35_button = st.button("GPT-3.5")
        with model_col2:
            gpt_4_button = st.button("GPT-4")
        # Store the selected model in session state
        if gpt_35_button:
            st.session_state.selected_model = "gpt-3.5-turbo"
        elif gpt_4_button:
            st.session_state.selected_model = "gpt-4"

        # Default to GPT-3.5 if no model is selected
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "gpt-4"

        st.sidebar.write(f"Selected model: {st.session_state.selected_model}")
        max_token_question = st.sidebar.number_input("Max tokens (question):", min_value=1, value=2500)
        max_token_answer = st.sidebar.number_input("Max tokens (answer):", min_value=1, value=1000)
        temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.3)
        # reframing = st.sidebar.checkbox("Enable reframing questions", value=False)
        reframing = False
        # Add a separator
        st.sidebar.markdown(
            "<hr style='height: 1px; border: none; background-color: gray; margin-left: -20px; margin-right: -20px;'>",
            unsafe_allow_html=True,
        )

        # Token usage
        st.sidebar.subheader("Token Usage")

        if not st.session_state.show_feedback_form:
            st.header("Input")
            display_input(max_token_question, max_token_answer, temperature, reframing)

        if "show_feedback_form" in st.session_state and st.session_state.show_feedback_form:
            question = [entry["content"] for entry in st.session_state.chat_history if entry["role"] == "user"][0]
            answer = [entry["content"] for entry in st.session_state.chat_history if entry["role"] == "assistant"][0]
            st.header("Feedback")
            display_feedback(
                question,
                answer,
                [st.session_state.prompt_tokens, st.session_state.completion_tokens, st.session_state.total_tokens],
                [st.session_state.selected_model, max_token_question, max_token_answer, temperature],
            )

            # Add a New Session/Chat button in app.py
        # if st.button("New Session/Chat"):
        #     clear_chat_history()
        #     st.session_state.chat_history = []

    with log_tab:
        # Load and display the chatlog dataframe
        df = pd.DataFrame(feedback_data).iloc[::-1]

        # Convert the list columns to string representation
        df["tokens_used"] = df["tokens_used"].astype(str)
        df["model_parameters"] = df["model_parameters"].astype(str)

        st.dataframe(df, use_container_width=True)


def get_checked_categories(unique_categories, on=[]):
    checked_categories = []
    for key in unique_categories:
        st.write(key)
        col1, col2, col3 = st.columns(3)
        for i, category in enumerate(unique_categories[key]):
            checked = category in on  # Check the category if it's in the "on" list
            if checked and category not in checked_categories:
                checked_categories.append(category)

            if i % 3 == 0:
                checked = col1.checkbox(category, value=checked, key=f"{category}_checkbox")
            elif i % 3 == 1:
                checked = col2.checkbox(category, value=checked, key=f"{category}_checkbox")
            else:
                checked = col3.checkbox(category, value=checked, key=f"{category}_checkbox")

            if checked and category not in checked_categories:
                checked_categories.append(category)
            elif not checked and category in checked_categories:
                checked_categories.remove(category)
    return checked_categories


def update_chat(
    user_input,
    selected_instruction,
    checked_categories,
    chat_container,
    placeholder_response,
    max_token_question,
    max_token_answer,
    temperature,
    reframing,
    selected_model,
):
    prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
    if user_input:
        print(f"{selected_model = }")
        updated_stream = ""
        st.session_state.chat_history, context_details, prompt_tokens, completion_tokens, total_tokens = answer_question(
            question=user_input,
            instruction=instructions[selected_instruction],
            categories=checked_categories,
            index=index,
            debug=False,
            model=selected_model,
            max_tokens=max_token_answer,
            max_len=max_token_question,
            temperature=temperature,
            reframing=reframing,
            callback=lambda text: display_stream_answer(text, placeholder_response),
        )

        display_context_details(context_details)
        display_chat(st.session_state.chat_history[1:-1], chat_container)

        return prompt_tokens, completion_tokens, total_tokens

    return 0, 0, 0


def display_stream_answer(r_text, placeholder_response):
    global updated_stream
    stream_text = ""
    updated_stream += r_text
    stream_text += f'<div margin: 0; padding: 10px;">{updated_stream}</div>'
    placeholder_response.markdown(stream_text, unsafe_allow_html=True)
    return updated_stream  # Return the updated stream to be used in feedback


def display_input(max_token_question, max_token_answer, temperature, reframing):
    with st.form(key="input_form", clear_on_submit=True):
        user_input = st.text_area("Insert customer mail:", key="ask_question")
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            send_button = st.form_submit_button("Send")

    search_options_expander = st.expander("Search Options")
    with search_options_expander:
        selected_instruction = st.radio("Instructions", list(instructions.keys()))

        edit_instructions = st.checkbox("Edit instructions")
        if edit_instructions:
            instruction_key = st.selectbox("Instruction key:", list(instructions.keys()), index=0)  # Change to selectbox
            instruction_value = st.text_area("Instruction value:", value=instructions[instruction_key])  # Add value

            button_row = st.columns(2)
            with button_row[0]:
                update_button = st.button("Update")
            with button_row[1]:
                delete_button = st.button("Delete")

            add_new_key = st.text_input("Add new instruction key:")  # Add new input for a new instruction key
            add_new_value = st.text_area("Add new instruction value:")  # Add new input for a new instruction value
            add_button = st.button("Add")  # Add button for adding new instruction

            if update_button:
                instructions[instruction_key] = instruction_value
                save_to_gcs(bucket_name, instructions_file_path, instructions)
                st.success(f"Updated instruction: {instruction_key}")
                st.experimental_rerun()

            if delete_button and instruction_key in instructions:
                del instructions[instruction_key]
                save_to_gcs(bucket_name, instructions_file_path, instructions)
                st.success(f"Deleted instruction: {instruction_key}")
                st.experimental_rerun()

            if add_button:
                if add_new_key and add_new_value:  # Check if the new instruction key and value are not empty
                    instructions[add_new_key] = add_new_value
                    save_to_gcs(bucket_name, instructions_file_path, instructions)
                    st.success(f"Added new instruction: {add_new_key}")
                    st.experimental_rerun()
                else:
                    st.warning("Please provide both a key and a value for the new instruction.")

        on = ["mail", "actions", "getting-started", "basics", "templates", "features", "controls", "filters", "woocommerce"]
        checked_categories = get_checked_categories(unique_categories, on)

    if send_button:
        st.session_state.chat_history = []
        print(f"\nFresh start: \n {st.session_state.chat_history =}")
        st.session_state.show_feedback_form = True  # Set the variable to show the feedback form
        placeholder_response = st.empty()
        chat_container = st.container()  # Updated container definition
        st.session_state.prompt_tokens, st.session_state.completion_tokens, st.session_state.total_tokens = update_chat(
            user_input,
            selected_instruction,
            checked_categories,
            chat_container,
            placeholder_response,
            max_token_question,
            max_token_answer,
            temperature,
            reframing,
            st.session_state.selected_model,
        )

        # Update the token count in the sidebar
        st.sidebar.write(f"Tokens used for prompt: {st.session_state.prompt_tokens}")
        st.sidebar.write(f"Tokens used for completion: {st.session_state.completion_tokens}")
        st.sidebar.write(f"Total tokens: {st.session_state.total_tokens}")
        if "total_chat_tokens" not in st.session_state:
            st.session_state.total_chat_tokens = 0
        st.session_state.total_chat_tokens += st.session_state.total_tokens
        st.sidebar.write(f"Total tokens in chat session: {st.session_state.total_chat_tokens}")

        st.session_state.show_feedback_form = True  # Set the variable to show the feedback form


def display_feedback(question, answer, tokens_used, model_parameters):
    with st.form(key="feedback_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            feedback_quality = st.radio("Feedback Quality:", ["helpful", "not so helpful"], key="feedback_quality")
        with col2:
            feedback_tag = st.radio("Feedback Category:", ["general", "tech"], key="feedback_tag")

        correct_answer = answer  # Default value
        helpful = True  # Default value
        comment = None  # Default value

        # If feedback is "bad," display text area for correct answer and set helpful to False
        if feedback_quality == "not so helpful":
            helpful = False

        comment = st.text_input("Please provide your comments here:", key="comment")
        helpwise = st.text_input("Please provide your HelpWise ticket URL here:", key="helpwise")
        correct_answer = st.text_area("Please provide the correct answer:", key="correct_answer")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if feedback_quality == "helpful":
            correct_answer = answer
        # Submit button for the form
        submitted = st.form_submit_button("Send Feedback")

        if submitted:
            print("submitting feedback...")
            feedback_data = {
                "date": timestamp,
                "question": question,
                "gpt_answer": answer,
                "correct_answer": correct_answer,
                "comment": comment,
                "tag": feedback_tag,
                "helpful": helpful,
                "ticket": helpwise,
                "tokens_used": tokens_used,
                "model_parameters": model_parameters,
            }

            # Load existing feedback from Google Cloud Storage
            existing_feedback = load_from_gcs(bucket_name, feedback_data_file_path)

            # Append the new feedback
            existing_feedback.append(feedback_data)

            # Save the updated feedback to Google Cloud Storage
            save_to_gcs(bucket_name, feedback_data_file_path, existing_feedback)

            print(f"{st.session_state.chat_history =}")
            st.success("Feedback sent successfully!")
            clear_chat_history()
            st.session_state.chat_history = []
            st.session_state.show_feedback_form = False

            print(f"{st.session_state.chat_history =}")
            print(f"{st.session_state.show_feedback_form =}")
            st.rerun()  # Rerun the script from the top


def display_chat(chat_history, chat_container):
    chat_text = ""
    for entry in reversed(chat_history):
        if entry["role"] == "user":
            chat_text += f'<div style="margin: 0; padding: 10px;">Question: {entry["content"]}</div>'

        else:
            chat_text += f'<div style="margin: 0; padding: 10px;">Question: {entry["content"]}</div>'

    chat_container.write(
        f"""
    <div id="chatBox" style="height: 110px; overflow-y: scroll; ">
        {chat_text}
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_context_details(context_details):
    context_details_expander = st.expander("Context Details")
    with context_details_expander:
        print(f"{context_details = }")
        # Convert context details to a Pandas DataFrame
        df_context_details = pd.DataFrame(context_details)
        # Transform the score into a percentage with two decimal places
        df_context_details["score"] = df_context_details["score"].apply(lambda x: f"{x * 100:.2f}%")
        df_context_details["token"] = df_context_details["token"].apply(lambda x: f"{x:.0f}")
        df_context_details = df_context_details.sort_values(by="score", ascending=False)
        # Display the DataFrame as a table
        st.table(df_context_details[1:])


if __name__ == "__main__":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    app()
