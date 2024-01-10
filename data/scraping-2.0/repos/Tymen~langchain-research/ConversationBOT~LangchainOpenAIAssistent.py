import os

from dotenv import load_dotenv
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from typing import List, Tuple
import gradio as gr
# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_TOKEN')
print("Initializing chat bot...")
# Initialize Vector Database
with open('./data/datacontext.txt') as f:
    mysqldbdocs = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=200, 
    length_function=len, 
    is_separator_regex=False)

texts = text_splitter.create_documents([mysqldbdocs])
vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Initialize OpenAI Assistant
assistant = OpenAIAssistantRunnable.create_assistant(
    name="ASkWatson",
    instructions="""
    Your name is ASkWatson, you are an assistant for interactive tasks.
    There is several rules that you need to follow i'm going to rank them from most important to least important.
    If it's less important you are allowed to be more creative and if it's important you need to be more strict. 
    Points are from 1 - 10 with 1 being least important and 10 being most important
    - [10] Your name is ASkWatson, you are an assistant for interactive tasks.
    - [5] If information is available in chat history, use this.
    - [7] Use a friendly and supportive tone in all responses.
    - [6] You are only allowed to answer the question on the following context or Chat History. You may customize the answer to the question.
    - [10] If there is data that needs to be provided by the user, please ask me for this information before answering my question. 
    - [7] You always end the answer with a follow-up question either to ask for extra context or to ask if you can help about a specific topic that is related to the question.
    - [8] Answer the question only on the following context or Chat History.
    - [10] The answers needs to be inline with AS Watson Group. The context is based on the AS Watson Group documentation.
    If you don't know the answer, just say that you don't know. answer me comprehensively.
    """,
    tools=[{"type": "code_interpreter"}],  # Add other tools if necessary
    model="gpt-3.5-turbo-1106",
)

# Chat history and thread ID
chathistory = []
thread_id = None

def format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "User: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

def process_input(input_text: str, thread_id: str) -> Tuple[str, str]:

    # Retrieve context from the vector database based on the user's input
    docs = retriever.invoke(input_text)
    
    # If 'docs' are Document objects, access their content directly
    retrieved_context = " ".join([doc.page_content for doc in docs]) if docs else ""

    # Combine the retrieved context with the chat history
    combined_context = "\nContext: \n" + retrieved_context
    # Prepare input for the assistant with combined context
    assistant_input = {
        "content": input_text, 
        "context": combined_context
    }
    if thread_id:
        assistant_input["thread_id"] = thread_id

    # Invoke OpenAI Assistant
    output = assistant.invoke(assistant_input)

    # Extract response and update thread ID
    if isinstance(output, list) and len(output) > 0:
        response = ' '.join([msg.text.value for msg in output[0].content if hasattr(msg.text, 'value')])
        thread_id = output[0].thread_id
    else:
        response = ''
    
    return response, thread_id

def chatbot(input_text, history, thread_id):
    global chathistory

    result, thread_id = process_input(input_text, thread_id)

    # Update the history with new conversation entries
    new_entry = [input_text, result]
    history.append(new_entry)
    return history, "", thread_id  # Clear input box after submission

# Define the Gradio Interface using Blocks
css = """
#component-3 {
    min-height: 60vh!important;
}
#component-5{
    border: none!important;
    border-bottom-right-radius: 0!important;
    border-top-right-radius: 0!important;
}

#component-5 textarea {
    border: none!important;
    border-radius: 0!important;
}

#component-6 { 
    max-width: 15%;
    border-bottom-left-radius: 0!important;
    border-top-left-radius: 0!important;
    background-color: orange!important;
}

#component-4 {
    gap: 0;
}

"""

# Define the Gradio Interface using Blocks
with gr.Blocks(css=css) as interface:
    gr.Markdown("### ASkWatson Chatbot")
    with gr.Row():
        history_box = gr.Chatbot(label="Chat")
    with gr.Row():
        input_box = gr.Textbox(label=None, placeholder="Type your message here...", )
        submit_btn = gr.Button("Send")
        thread_id_state = gr.State(None)
    input_box.submit(
        fn=chatbot,
        inputs=[input_box, history_box, thread_id_state],
        outputs=[history_box, input_box, thread_id_state]
    )
    submit_btn.click(
        fn=chatbot,
        inputs=[input_box, history_box, thread_id_state],
        outputs=[history_box, input_box, thread_id_state]
    )

launchedInterface = interface.launch(share=True)
vectorstore.delete_collection()
# def chatbot(input_text, thread_id):
#     global chathistory

#     result, thread_id = process_input(input_text, thread_id)
    
#     if chathistory:
#         chathistory[-1] = [input_text, result]
#     else:
#         chathistory.append([input_text, result])

#     return result, thread_id

# # Initialize Gradio State for the thread ID
# thread_id_state = gr.State(None)

# # Define the Gradio Interface
# iface = gr.Interface(
#     fn=chatbot,
#     inputs=[gr.Textbox(label="Your Message"), thread_id_state],
#     outputs=[gr.Textbox(label="ASkWatson's Response"), thread_id_state],
#     title="ASkWatson Chatbot",
#     css="#component-3 {height: 60vh!important}"
# )

# iface.launch(share=True)

# def chatbot(input_text, history):
#     global thread_id  # Declare thread_id as a global variable

#     # Rest of your code remains the same
#     result, thread_id = process_input(input_text, thread_id)

#     if chathistory:
#         chathistory[-1] = [history[-1][0], result]
#     else:
#         chathistory.append([input_text, result])

#     return result

# gr.ChatInterface(chatbot, title="ASkWatson", css="#component-3 {height: 60vh!important}").launch(share=True)
# Main interaction loop
# while True:
#     input_text = input("Human: ")
#     result, thread_id = process_input(input_text, chathistory, thread_id)
#     chathistory[-1] = [chathistory[-1][0], result]
#     print(chathistory)
#     print("Assistant: " + result)