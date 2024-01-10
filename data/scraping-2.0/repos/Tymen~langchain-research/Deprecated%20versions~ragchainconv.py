from operator import itemgetter
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import format_document
from langchain.schema.runnable import RunnableMap
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter

from langchain.memory import ConversationBufferMemory
from typing import List, Tuple
import datetime
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_TOKEN')

with open('./data/datacontext.txt') as f:
    mysqldbdocs = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)

texts = text_splitter.create_documents([mysqldbdocs])

vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

_template = """Given the following conversation and a follow up question, in the english language.
Chat History:
{chat_history}
Follow Up Input: {question}
Question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """If information is available in chat history make sure to use this. 
If there is data that needs to be provided by the user please ask me for this information before answering my question. 
You always end the answer with a follow up question either to ask for extra context or to ask if you can help about a specific topic that is related to the question.

Answer the question only on the following context or Chat History.
Context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

chathistory = [
]

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables),
)

# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["history_chat"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0.5, model="gpt-4")
    | StrOutputParser(),
}

# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

while True:
    input_text = input("Human: ")
    now = datetime.datetime.now()
    formatted_now = "DateTime: " + now.strftime("%Y-%m-%d %H:%M:%S") + " "
    inputs = {"question": formatted_now + input_text, "history_chat": chathistory}
    result = final_chain.invoke(inputs)
    chathistory.append((formatted_now + input_text, formatted_now + result["answer"].content))
    
    print("Answer: " + result["answer"].content)