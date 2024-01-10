import boto3
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, AgentOutputParser, initialize_agent, ConversationalChatAgent, AgentExecutor
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool, StructuredTool, Tool, tool
import streamlit as st
from sagemaker import session
import json
import re
from typing import Dict, List

REGION = boto3.Session().region_name

def format_messages(messages: List[Dict[str, str]]) -> List[str]:
    """
    Format messages for Llama-2 chat models.
    
    The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and 
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    prompt: List[str] = []

    if messages[0]["role"] == "system":
        content = "".join(["<<SYS>>\n", messages[0]["content"], "\n<</SYS>>\n\n", messages[1]["content"]])
        messages = [{"role": messages[1]["role"], "content": content}] + messages[2:]
    for user, answer in zip(messages[::2], messages[1::2]):
        prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ", (answer["content"]).strip(), "</s>"])
    prompt.extend(["<s>", "[INST] ", (messages[-1]["content"]).strip(), " [/INST] "])
    return "".join(prompt)

f = open("endpoint_name.txt", "r")
endpoint_name = f.read()
f.close()

f = open("custom_attribute.txt", "r")
custom_attributes = f.read()
f.close()

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs):
        base_input = [{"role" : "user", "content" : prompt}]
        optz_input = format_messages(base_input)
        input_str = json.dumps({
            "inputs" : optz_input, 
            "parameters" : {**model_kwargs}
        })
        return input_str.encode('utf-8')
    
    def transform_output(self, output):
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]

_prefix = "Chat with your Documents using RAG" 
st.set_page_config(page_title=f"SageMaker & LangChain: {_prefix}", page_icon="🦜")
st.title(f"🦜 SageMaker & LangChain: {_prefix}")

content_handler = ContentHandler()
msgs = StreamlitChatMessageHistory()

llm=SagemakerEndpoint(
             endpoint_name=endpoint_name, 
             region_name=session.Session().boto_region_name, 
             model_kwargs={
                 "max_new_tokens": 512, 
                 "top_p": 0.1, 
                 "temperature": 0.1
             },
             endpoint_kwargs={
                 "CustomAttributes": custom_attributes
             },
             content_handler=content_handler
         )

template = """
Answer the following QUESTION based on the CONTEXT given. If you do not know the answer and the CONTEXT doesn't contain the answer truthfully say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt_template = PromptTemplate(
    template=template, 
    input_variables=['context', 'question']
)


class EmbeddingGenerator:
    def __init__(self):
        self.lambda_client = boto3.client('lambda', region_name=REGION)
    
    def embed_query(self, input_text_sample):
        """Generate embeddings for the input text."""
        
        lambda_client = boto3.client('lambda', region_name=REGION) 
        
        # Prepare the data to send to the Lambda function
        data = {
            "input": input_text_sample
        }

        # Invoke the Lambda function
        response = lambda_client.invoke(
            FunctionName="invokeEmbeddingEndpoint",
            InvocationType="RequestResponse",
            Payload=json.dumps(data)
        )

        # Decode and load the response payload.
        response_payload = json.loads(response['Payload'].read().decode("utf-8"))

        # Extract status and embeddings from the response.
        status_code, embeddings = int(response_payload['statusCode']), json.loads(response_payload['body'])[0]

        return embeddings


f = open("opesearchurl.txt", "r")
opensearch_url = f.read()
f.close()

f = open("indexname.txt", "r")
index_name = f.read()
f.close()

f = open("opensearchlogin.txt", "r")
login_details = f.read()
user, pwd = login_details.split('|||')
f.close()

embedding_generator = EmbeddingGenerator()

docsearch = OpenSearchVectorSearch(
    index_name=index_name,
    embedding_function=embedding_generator,
    opensearch_url=opensearch_url,
    http_auth=(user, pwd),
    engine="faiss"
)

llm_qa_smep_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=docsearch.as_retriever(search_kwargs={
        "k": 10, 
        "space_type": "cosineSimilarity",
        "space_type": "painless_scripting"
    }),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

def pretty_print(chain_op):
    question = chain_op['query']
    response = chain_op['result']
    sources = "-" + "\n-".join([f"{src.metadata['source'].split('/')[-1]} (page: {src.metadata['page']})" for src in chain_op['source_documents']])
    sources = f"""```bash{sources}"""
    stdout = f"""{response}\n\n##### Sources:\n{sources}"""
    return stdout
    

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

st.sidebar.text("Need Suggestion to Chat with your Docs??")

# Replace the for loop that prints the suggestions in the sidebar with the following:
suggestions = [
    "What is a SageMaker Training job and how do you run it?",
    "What types of GPU instance types are supported SageMaker?",
    "How to install packages on EC2 instances using Command line?",
    "How to Create a Training Job using Boto3 SDK?",
    "How can I deploy a model to SageMaker Hosting service?",
    "How can I use the console to add a git repository to my SageMaker account?"
]

# Function to handle click on suggestion
def handle_click(suggestion_text):
    if 'clicked_text' not in st.session_state:
        st.session_state.clicked_text = suggestion_text
    else:
        st.session_state.clicked_text += '\n' + suggestion_text
    st.session_state.prompt = suggestion_text  # This will set the text in the main chat input box


# Create buttons for suggestions
for idx, suggestion in enumerate(suggestions):
    st.sidebar.button(suggestion, key=f"suggestion_{idx}", on_click=handle_click, args=(suggestion,))


avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)
    
# In the main chat input area, check if there's clicked text to be used as prompt
if 'clicked_text' in st.session_state:
    prompt = st.session_state.clicked_text
    print("Clicked Text", prompt)
    st.chat_input(placeholder="Please ask me a question!")
    del st.session_state['clicked_text']  
    st.chat_message("user").write(prompt)
    print("Question Asked ====>", prompt)
    response = pretty_print(llm_qa_smep_chain(prompt))
    print("Response ====>", response)
    st.chat_message("assistant").write(response)
    msgs.add_user_message(prompt)
    msgs.add_ai_message(response)
    
elif prompt := st.chat_input(placeholder="Please ask me a question!"):
    st.chat_message("user").write(prompt)
    print("Question Asked ====>", prompt)
    response = pretty_print(llm_qa_smep_chain(prompt))
    print("Response ====>", response)
    st.chat_message("assistant").write(response)
    msgs.add_user_message(prompt)
    msgs.add_ai_message(response)