import os
import streamlit as st
import psycopg2
import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import PostgresChatMessageHistory
from langchain.agents import AgentExecutor
from langchain.callbacks import get_openai_callback
from langchain.tools import BaseTool, Tool
from langchain.vectorstores.pgvector import PGVector
from langchain.schema.messages import HumanMessage, AIMessage
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Load environment variables from .env if it exists.
from dotenv import load_dotenv
load_dotenv()

# Get session info so we can uniquely identify sessions in chat history table.
def get_session_id() -> str:
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None
    except Exception as e:
        return None
    return ctx.session_id

# Initialize connection string for PostgreSQL storage
connection_string="postgresql://{pg_user}:{pg_pass}@{pg_host}/{pg_db}".format(
    pg_user=os.getenv('PG_USER'),
    pg_pass=os.getenv('PG_PASS'),
    pg_host=os.getenv('PG_HOST'),
    pg_db=os.getenv('PG_DB')
)

db_history = PostgresChatMessageHistory(
    connection_string=connection_string,
    session_id=get_session_id() # Unique UUID for each session.
)

# Hack to get multi-input tools working again.
# See: https://github.com/langchain-ai/langchain/issues/3700#issuecomment-1568735481
from langchain.agents.conversational_chat.base import ConversationalChatAgent
ConversationalChatAgent._validate_tools = lambda *_, **__: ...

# Define available OpenAI models.
models = [
    "gpt-3.5-turbo", 
    "gpt-3.5-turbo-0301", 
    "gpt-3.5-turbo-0613", 
    "gpt-3.5-turbo-16k", 
    "gpt-3.5-turbo-16k-0613", 
    "gpt-4", 
    "gpt-4-0314", 
    "gpt-4-0613",
]

# Initialize the OpenAI Class
llm = ChatOpenAI(temperature=0, model=models[2])

# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="special_app_key")

# Initialize chatbot memory
conversational_memory = ConversationBufferMemory(
    memory_key = "chat_history",
    chat_memory = msgs,
    input_key="input",
    return_messages = True,
    ai_prefix="AI Assistant",
)

# Prepare tools for bot
# "Exemplar" - Ideal summary of last week's content provided by instructor. The bot should use this as a comparator for the student's reflection.
class Exemplar (BaseTool):
    name="Exemplar"
    description="Use this tool to receive the instructors example summary of last week's learning materials to compare against student reflections"

    def _run(args, kwargs):
        return "Self-regulated learning (SRL) is a multifaceted process that empowers learners to proactively control and manage their cognitive, metacognitive, and motivational processes in pursuit of learning objectives. Rooted in social-cognitive theory, SRL emphasizes the active role of learners in constructing knowledge, setting and monitoring goals, and employing strategies to optimize understanding. It posits that successful learners are not merely passive recipients of information but are actively involved in the learning process, making decisions about which strategies to employ, monitoring their understanding, and adjusting their efforts in response to feedback. Metacognition, a central component of SRL, involves awareness and regulation of one's own cognitive processes. Successful self-regulated learners are adept at planning their learning, employing effective strategies, monitoring their progress, and adjusting their approaches when necessary. These skills are crucial not only in formal educational settings but also in lifelong learning, as they enable individuals to adapt to evolving challenges and continuously acquire new knowledge and skills throughout their lives."

class Assignment (BaseTool):
    name="Assignment"
    description="Use this tool to obtain the student's assignment"

    def _run(args, kwargs):
        return """Your assignments is to carefully read the two articles provided to you: "Models of Self-regulated Learning: A review" and "Self-Regulated Learning: Beliefs, Techniques, and Illusions.\n"
        Based on your understanding, prepare the following answers in 500 words or less:\n
        a) Definition of SRL: In your own words, provide a definition of self-regulated learning.\n
        b) Model Description: Describe one of the SRL models that you found most interesting. Explain why it resonated with you.\n
        c) Learning Activity Proposal: Suggest an example learning activity or experience that could be integrated into an academic course. This activity should scaffold self-regulated learning for students.\n\n
        Go ahead and submit when you're ready!
        """
    
# Connect to our postgres database vector store via pgvector.
embeddings = OpenAIEmbeddings()
vectorstore = PGVector(
    embedding_function=embeddings,
    collection_name=os.getenv('PGVECTOR_COLLECTION_NAME'),
    connection_string=PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=os.getenv('PG_HOST'),
        port=os.getenv('PG_PORT'),
        database=os.getenv('PG_DB'),
        user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASS'),
    )
)

# Learning Materials: DB of vectorized learning materials from the prior week. 
# The bot should reference these materials when providing feedback on the 
# student reflection, referencing what content was covered in the learning 
# materials, or what materials to review again to improve understanding.
papers = [
    'srlpaper.pdf',
]

# Create vectors if they don't exist.
papers_to_add = []
for paper in papers:
    docs = vectorstore.similarity_search('srlpaper.pdf')
    found_paper = False
    for doc in docs:
        if paper == doc.metadata['source']:
            found_paper = True
            break
    if not found_paper:
        papers_to_add.append(paper)
if papers_to_add:
    # Load course data with PDFPlumber: $ python3 -m pip install pdfplumber
    from langchain.document_loaders import PDFPlumberLoader
    for paper in papers_to_add:    
        loader = PDFPlumberLoader(paper)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len,
        )
        docs = text_splitter.split_documents(pages)
        vectorstore.add_documents(docs)
        print('Added vectors for paper: ', paper)

# See: https://www.pinecone.io/learn/series/langchain/langchain-retrieval-augmentation/
#search_readings_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
search_readings_tool = Tool(
    name="Class Assignment Readings",
    #func=search_readings_chain.run
    func = vectorstore.similarity_search,
    description="useful for when you need to search class readings for a specific topic"
)

# Define tools
tools = [Exemplar(), Assignment(), search_readings_tool]

# Define the input variables
input_variables = {
    "student_name": "Alice",
    "topic_name": "Self-regulated learning"
}


# Create template for system message to provide direction for the agent
role_description = """Your name is Sigma and your goal is to provide feedback to students on their assignment.
"""

analysis_instructions = """Once the student has submitted part or all of their assignment take the following steps:
1) Compare the submission against the assignment tool and note what was missing. Then ask the student if you should evaluate their submission.
2) If the student says yes to evaluating their submission, compare what they have provided against the exemplar tool. Provide feedback to the student on overall quality, what was correct, and where it could be improved.
"""

rules = """Rules:
- Sigma should only talk about the assignment
- Keep the conversation on task to complete the assignment
"""

system_message = role_description + analysis_instructions + rules

# Trying different agent constructor
newAgentPrompt = ConversationalChatAgent.create_prompt(tools=tools, system_message=system_message, input_variables=["chat_history", "input", "agent_scratchpad"])
llm_chain = LLMChain(llm=llm, prompt=newAgentPrompt)
agent = ConversationalChatAgent(llm_chain=llm_chain, tools=tools, verbose=True)
executor = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools,
    memory = conversational_memory,
    early_stopping_method = "force",
    handle_parsing_errors = True,
    max_iterations = 4,
    #return_intermediate_steps = True,
    verbose = True,
)

# Add a callback to count the number of tokens used for each response.
# This callback is not necessary for the agent to function, but it is useful for tracking token usage.
def run_query_and_count_tokens(chain, query):
    with get_openai_callback() as cb:
        print('query \n')
        print(query)
        print('query end \n')
        result = chain.run(query)
        print(cb)
    return result

# Streamlit Code
st.set_page_config(page_title="Sigma - Learning Mentor", page_icon=":robot:")

# Use consistent styling and layout
background_color = "#f4f4f6"
font_color = "#333"

st.markdown(
    f"""
    <style>
        body {{
            background-color: {background_color};
            color: {font_color};
        }}
    @keyframes typing {{
        0% {{ content: '.'; }}
        25% {{ content: '..'; }}
        50% {{ content: '...'; }}
        75% {{ content: '..'; }}
        100% {{ content: '.'; }}
    }}

    .typing-animation::before {{
        content: '...';
        animation: typing 1s steps(5, end) infinite;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Sigma - Learning Mentor")

# Chat Window Styling
chat_window_style = """
    border-radius: 10px;
    border: 1px solid #ddd;
    background-color: #fff;
    padding: 10px;
    height: 400px;
    overflow-y: auto;
    font-size: 16px;
    font-family: Arial, sans-serif;
"""

response_style = """
    border-radius: 5px;
    background-color: #e1f3fd;
    padding: 10px;
    margin: 10px 0;
    display: inline-block;
"""

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
  
# Initialize chat history
if "messages" not in st.session_state:
    welcome_message = """Hello! My name is Sigma and I am here to help you reflect on what you learned last week."""
    st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
    db_history.add_message(AIMessage(
        content=welcome_message, 
        additional_kwargs={'timestamp': datetime.datetime.now().isoformat()}
    ))
  
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role":"user", "content":prompt})
    db_history.add_message(HumanMessage(
        content=prompt,
        additional_kwargs={'timestamp': datetime.datetime.now().isoformat()}
    )
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant thinking animation in chat message container
    with st.chat_message("assistant"):
        # This placeholder will initially show the "thinking" animation
        message_placeholder = st.empty()
        message_placeholder.markdown('<div class="typing-animation"></div>', unsafe_allow_html=True) # Simple 3 dots "thinking" animation
        
        # Get the response from the chatbot
        #response = executor.run(prompt)
        #print(conversational_memory.buffer_as_messages) - Uncomment to see message log
        response = run_query_and_count_tokens(executor, prompt)

        # Replace the "thinking" animation with the chatbot's response
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        db_history.add_message(AIMessage(
            content=response, 
            additional_kwargs={'timestamp': datetime.datetime.now().isoformat()}
        ))



# TODO 
# Serialization for better user experience: https://python.langchain.com/docs/modules/model_io/models/llms/streaming_llm
# LLM inference quality, peformance, and token usage tracking through langsmith: https://docs.smith.langchain.com/
# Guardrails for the conversation to keep it focused and safe for students. Some optionsinclude NVidia's - https://github.com/NVIDIA/NeMo-Guardrails and Guardrails.ai - https://docs.getguardrails.ai/
# Fix 'Could not parse LLM Output' error that is curently being handled automatically on via parameter in agent initialization. This appears to slightly impact performance, but not quality of inference. Some potential conversation to help find the solution:
# https://github.com/langchain-ai/langchain/pull/1707
# https://github.com/langchain-ai/langchain/issues/1358
# Nice video on the difference between map-reduce, stuff, refine, and map-rank document searches with an example:
# https://www.youtube.com/watch?v=OTL4CvDFlro