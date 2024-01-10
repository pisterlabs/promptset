from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.memory import ConversationBufferMemory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import re

# Download word tokenizing data
nltk.download('punkt')

# Define a class for callback functions
class ChatCallbackHandler(BaseCallbackHandler):
    # Initialize a message variable
    message =""
    # When llm starts, an empty box is created
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    # When llm ends, save the created message
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    # Each new token generated, the function is called
    def on_llm_new_token(self, token, *args, **kwargs):
        # append each token to the message variable
        self.message += token
        # each message appened will be shown to the message box
        self.message_box.markdown(self.message)

# Create llms for answers and choice
answers_llm = ChatOpenAI(
    temperature=0.1,
)
choice_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

# Create a memory
memory = ConversationBufferMemory(
    llm=choice_llm,
    max_token_limit=120,
    memory_key="chat_history",
    return_messages=True,
)

answers_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Using ONLY the following context, answer the user's question. If you can't just say you don't know. DO NOT make anything up.
    
    Then, give a score to the answer between 0 and 5, 0 being not helpful and 5 being the most helpful to the user.
    
    Make sure to add scores to the answers
    
    Context: {context}

    Example: 
    Question: How far away is the moon?
    Answer: The moon is 384,000 km away
    Score: 5
    
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
    
    Your turn!
    """), 
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
]
)


def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    chat_history = inputs['chat_history']
    answers_chain = answers_prompt | answers_llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #         "question": question,
    #         "context": doc.page_content,
    #     })
    #     answers.append(result)
    # Return a dictionary of a question and answers to send to choose_answer function
    return {"question": question,
            "chat_history":chat_history,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {
                            "question": question,
                            "chat_history": chat_history,
                            "context": doc.page_content,
                        }
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata["lastmod"],
                }
                for doc in docs
            ],
            }


choice_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         Use ONLY the following pre-existing answers to answer the user's question.
         
         Use the answers that have the highest score(more helpful) and favor the most recent one.
         
         Provide the sources. Return the source without any modification.
         
         Answer: {answers}
         """,
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")

    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    chat_history = inputs['chat_history']
    choice_chain = choice_prompt | choice_llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" for answer in answers)
    return choice_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
            "answers": condensed,
        }
    )


def parse_page(soup):
    # Currently using this to specifically scrape OpenAI site map
    header = soup.find("header")
    footer = soup.find("footer")
    regex = r"Authors(\w+\s\w*View all articles)+"
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    text = str(soup.get_text()).replace(
        "\n", " ").replace("\xa0", " ").replace("CloseSearch Submit Blog", " ")
    result = re.sub(regex, " ", text)
    return result


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    # Create a splitter to split documents
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=200,
    )
    # sitemap loader to filter urls
    loader = SitemapLoader(
        url,
        # filter by regex
        # filter_urls=[
        #     r"^(.*\/blog\/).*",
        # ],
        parsing_function=parse_page
    )
    # Create a cached embeddings
    regex = re.compile(r"(www.)*\w+\.\w{2,3}")
    cache_dir = LocalFileStore(
        f"./.cache/site_embeddings/{re.search(regex, url).group()}")
    embedder = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder,
        cache_dir
    )
    loader.requests_per_second = 2
    # Create documents with splitter
    docs = loader.load_and_split(text_splitter=splitter)
    # Create a vector store with docs and embeddings
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    # retriever has invoke() method, which can be used in chain
    retriever = vector_store.as_retriever()
    return retriever

# Save the message and memory to the session_state
def save_message(message, role):
    st.session_state["messages"].append(
        {"message": message, "role": role}
    )
def save_memory(input, output):
    st.session_state["chat_history"].append(
        {"input": input, "output": output}
    )
    
def send_message(message, role, save=True):
    # shows messages in the beginning, and save them
    with st.chat_message(role):
        st.markdown(message)
    if save:
        # Note that the messages are stored in a dictionary form
        save_message(message, role)
        

st.set_page_config(
    page_title="FullStackGPT SiteGPT",
    page_icon="💼",
)

# Displaying messages without saving them: display saved messages
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def restore_memory():
    for history in st.session_state["chat_history"]:
        memory.save_context({"input": history["input"]}, {"output": history["output"]})
        
def load_memory(input):
    return memory.load_memory_variables({})["chat_history"]

def invoke_chain(message):
    # invoke the chain
    result = chain.invoke(message)
    # save the interaction in the memory
    save_memory(message, result.content)
# Function to calculate similarity between two queries
def calculate_similarity(new_question, memory_question):
    # Tokenize the sentences
    token1 = word_tokenize(new_question.lower())
    token2 = word_tokenize(memory_question.lower())
    # Combine tokens into strings for vectorization
    text1 = ' '.join(token1)
    text2 = ' '.join(token2)
    # Vectorize the texts
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer)[0][1]

# Function to retrieve answers
def get_similar_answer(user_query):
    # Placeholder for the most similar query
    most_similar_question = None
    most_similar_answer = None
    # Placeholder for max similarity
    max_similarity = -1.0
    # Load memory variables
    chat_history = memory.load_memory_variables({})["chat_history"]
    # Iterate memory to find the most similar query to the user input
    stored_queries = [query.content for idx, query in enumerate(chat_history) if idx%2 == 0 or idx == 0]
    stored_answers = [answer.content for idx, answer in enumerate(chat_history) if idx%2 != 0]
    for stored_entry in zip(stored_queries, stored_answers):
        stored_query = stored_entry[0]
        stored_answer = stored_entry[1]
        #  Update max similarity and most similar question
        if stored_query is not None and stored_answer is not None:
            similarity = calculate_similarity(user_query, stored_query)
        if similarity > max_similarity: 
            max_similarity = similarity
            most_similar_question = stored_query
            most_similar_answer = stored_answer
    # Check for a valid similar question and similarity threshold
    if most_similar_question and max_similarity > 0.8:
        return most_similar_answer
    return None
    
st.title("SiteGPT")

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    st.markdown(
    """
    Ask questions about the content of a website.
    """)
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write sitemap URL")
    else:
        retriever = load_website(url)
        send_message("I am ready. Ask away", "ai", save=False)
        # Restore memory and paint the history of previous chat
        restore_memory()
        paint_history()
        query=st.chat_input("Ask a question about the website.")
        if query:
            send_message(query, "human")
            chain = ({"docs": retriever, "chat_history": load_memory, "question": RunnablePassthrough(),} | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
                     )
            similar_answer = get_similar_answer(query)
            with st.chat_message("ai"):
                if similar_answer:
                    similar_answer
                    # Invoking chain saves the message and memory but similar answer has to be manually saved
                    save_memory(query, similar_answer)
                    save_message(similar_answer, 'ai')
                else:    
                    invoke_chain(query)
else:
    # When there is no url(like in the beginning), initialize the session with a blank list
    st.markdown(
    """
    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.

    """)
    st.session_state["messages"] = []
    st.session_state["chat_history"] = []