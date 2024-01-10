import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
    api_key="<pincone-key>",
    environment="<pincone env>",
)

open_api_key = "<open-ai key>"

os.environ['OPENAI_API_KEY'] = open_api_key

human_template = "Context: {context}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [human_message_prompt])


embedding = OpenAIEmbeddings(openai_api_key=open_api_key)
docsearch = Pinecone.from_existing_index(
    "chemistry-2e", embedding=embedding)

docs = docsearch.similarity_search(
    "Introduction to chemistry and Topics covered in the book", k=10)

contexts = [chat_prompt.format_prompt(
    context=doc.page_content).to_messages() for doc in docs]


chat = ChatOpenAI(temperature=0.5, frequency_penalty=0.7, presence_penalty=0.3)
resp = chat([
    SystemMessage(
        content="Remember the Provided Context and Topic"),
    HumanMessage(
        content="Topic: Introduction to chemistry and Topics covered in the book"),
] + [c[0] for c in contexts]+[
    AIMessage(
        content="I have memorized the context and topic."),
    SystemMessage(content="You are a healer bot. Your job is to generate a script for a pre-recoded video Lecture. The topic and context for thw script has been provided to you. End the script with <EOS> token."),
    HumanMessage(
        content="go"
    )
])

print(resp.json())
