"""
This module contains handlers that handle messages from users

Handlers:
    echo_handler    - echoes the user's message

Note:
    Handlers are imported into the __init__.py package handlers,
    where a tuple of HANDLERS is assembled for further registration in the application
"""
import asyncio
import os
from typing import Any

import openai
import tiktoken
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import FAISS

from tgbot.utils.environment import env

GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
openai.api_key = env.get_openai_api()


def init_conversation():
    """
    Initialize a conversation for a given chat_id.
    """
    llm = ChatOpenAI(temperature=0)  # Assuming OpenAI has an async interface
    conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=ConversationEntityMemory(
            llm=llm, k=10
        ),  # Assuming ConversationEntityMemory has an async interface
    )
    return conversation
    # vectorstore = Chroma(
    #     embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
    # )
    # search = GoogleSearchAPIWrapper()
    # web_research_retriever = WebResearchRetriever.from_llm(
    #     vectorstore=vectorstore,
    #     llm=llm,
    #     search=search,
    # )
    # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm, retriever=web_research_retriever
    # )
    # return qa_chain


llm = init_conversation()
prompt_template_male = """
Character: John, analytical and curious, often tackles {user_topic}-related challenges methodically.
Occupation & Background: [Derived from {user_topic}]
"""

prompt_template_female = """
Character: Jane, creative and empathetic, addresses {user_topic} issues with innovative, human-centered solutions.
Occupation & Background: [Derived from {user_topic}]
"""
personality = {"Male": prompt_template_male, "Female": prompt_template_female}


async def get_conversation(user_data, query):
    """
    Get the conversation output for a given chat_id and query.
    """
    # Initialize the conversation if it doesn't exist for the chat_id
    # Choose the correct prompt based on the persona
    selected_prompt = (
        personality[user_data["PERSONA"]]
        if user_data["PERSONA"] in personality
        else None
    )
    google_search = await search_token(f"{query}. Topic: {user_data['TOPIC']}")
    # Run the conversation and get the output
    prompt = f"""
    As the Daily Advisor AI, your role is to be a knowledgeable and friendly companion to {user_data["NAME"]}. 
    You're tasked with providing accurate, reliable answers about {user_data["TOPIC"]}—a topic described as 
    {user_data["DESCRIPTION"]}. Your responses should be grounded in verifiable facts to ensure trustworthiness.

    Embody the character traits assigned to you, maintaining this persona consistently to build rapport with the user. 
    Your character is defined as follows: {selected_prompt.format(user_topic=user_data["TOPIC"])}. 

    Above all, your goal is to support {user_data["NAME"]}'s curiosity and learning about {user_data["TOPIC"]} with 
    engaging and informative dialogue. You responses have to be professional and cosine. Answer only based on subject 
    with no additional info.\n    
    Google Search results: {google_search}

    User query: {query}
    """
    # response = llm({"question": query})
    # output = response["answer"]
    output = await llm.arun(input=prompt)  # Assuming run is an async method
    return output


async def generate_category(topic, description, level):
    prompt = f"""
    Based on the main topic of '{topic}', which is briefly described as '{description}', and considering the user's 
    knowledge level of '{level}', generate a list of specific subcategories or areas of interest within this topic. 
    These subcategories should be relevant and tailored to the user's understanding, providing avenues for deeper 
    exploration or advice. Think broadly and include various aspects such as technologies, methodologies, applications, 
    and any other pertinent divisions related to '{topic}'.

    For instance, if the topic is 'AI', potential categories might include 'Machine Learning', 'AI Tools', 'Diffusion Models', 
    'Language Models', etc. List out similar categories that fit the scope of '{topic}'.
    """

    category = await generate_completion(prompt)
    return category.strip()


async def search_google_for_data(category, topic):
    search_query = f"Recent developments in {category} related to {topic}"
    search_results = await search_token(search_query)
    return search_results


async def generate_advice(user_data, google_data):
    prompt = f"""
Given the user's specific interests and needs as outlined in their profile: {user_data}, and incorporating the 
latest findings and data obtained from recent Google searches: {google_data}, formulate a piece of advice. This 
advice should be actionable, insightful, and tailored to the user's context. It should leverage the depth of 
knowledge available within the AI's database as well as the freshness and relevance of the information sourced 
from the web. Ensure that the guidance provided is coherent, directly applicable to the user's situation, and 
reflects the most current understanding of the topic at hand.
"""

    advice = await generate_chat_completion(prompt)
    return advice


async def create_advice(topic, description, level):
    # Step 1: Generate a category from the topic
    category = await generate_category(topic, description, level)

    # Step 2: Google search for data about the category
    google_data = await search_google_for_data(category, topic)

    # Step 3: Generate advice using the GPT model
    user_data = {
        "topic": topic,
        "description": description,
        "level": level,
        "category": category,
    }
    advice = await generate_advice(user_data, google_data)

    return advice


async def search_token(prompt: str) -> Any:
    search = GoogleSearchAPIWrapper()

    tool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search.run,  # Synchronous method
    )

    # Run the synchronous code in a thread pool executor
    return await asyncio.get_event_loop().run_in_executor(None, tool.run, prompt)


async def generate_chat_completion(input_data):
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {
                "role": "system",
                "content": "You will be given a task by user to create advices on some topic based on you train data "
                           "and given google search data. You have to generate good structured advice text for "
                           "telegram format. ",
            },
            {"role": "user", "content": input_data},
        ],
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 0.4,
        "frequency_penalty": 1.5,
        "presence_penalty": 1,
    }
    response = await openai.ChatCompletion.acreate(**data)

    responses = response["choices"][0]["message"]["content"]

    return responses


async def generate_chat(input_data, message):
    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {
                "role": "system",
                "content": f"You are a chat bot, created to chat with user on its topic: {input_data['TOPIC']}, "
                           f"for the level: {input_data['LEVEL']}",
            },
            {"role": "user", "content": f"User message: {message}"},
        ],
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 0.4,
        "frequency_penalty": 1.5,
        "presence_penalty": 1,
    }
    response = await openai.ChatCompletion.acreate(**data)

    responses = response["choices"][0]["message"]["content"]

    return responses


async def generate_completion(query: str) -> str:
    data = {
        "engine": "text-davinci-003",
        "prompt": query,
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 0,
        "frequency_penalty": 0.43,
        "presence_penalty": 0.35,
        "best_of": 2,
    }
    response = await openai.Completion.acreate(**data)
    # Extract the bot's response from the generated text
    answer = response["choices"][0]["text"]
    return answer


def ask_question(qa, question: str, chat_history):
    query = ""

    result = qa({"question": query, "chat_history": chat_history})
    print(result)
    print("Question:", question)
    print("Answer:", result["answer"])

    print(result)

    return result["answer"]


async def generate_response(query: str, vectorstore) -> str:
    knowledge = []
    # TODO: Test different things like similarity
    for doc in vectorstore.max_marginal_relevance_search(query, k=10):
        knowledge.append(doc)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": ()},
            {"role": "system", "content": ""},
            {"role": "user", "content": " "},
        ],
        temperature=0,
        max_tokens=3000,
        top_p=0.4,
        frequency_penalty=1.5,
        presence_penalty=1,
    )
    bot_response = response["choices"][0]["message"]["content"]
    return bot_response


def tiktoken_len(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def process_recursive(documents) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )
    embeddings = OpenAIEmbeddings()
    text_chunks = text_splitter.split_text(documents)
    db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return db


# Create a vector store indexes from the pdfs
def get_vectorstore(text_chunks: list[str]) -> FAISS:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
