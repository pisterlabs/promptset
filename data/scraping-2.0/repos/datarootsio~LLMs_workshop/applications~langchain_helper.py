from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.output_parsers import PydanticOutputParser
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from pydantic import BaseModel, Field, validator
from typing import Sequence
import pandas as pd


MODEL_NAME = "gpt-3.5-turbo-1106"


def generate_post_dataroots(
    openai_api_key, temperature, social_media, position, tone, max_words, extra_info=""
):
    llm = ChatOpenAI(
        temperature=temperature, openai_api_key=openai_api_key, model_name=MODEL_NAME
    )

    my_prompt = PromptTemplate(
        input_variables=["social_media", "position", "tone", "max_words", "extra_info"],
        input_types={
            "social_media": str,
            "position": str,
            "tone": str,
            "max_words": int,
            "extra_info": str,
        },
        template="""
    You are Dataroots assistant and you are here to help HR create new posts on {social_media} to recruite new people.
    The post should be about looking for a {position}.
    The max number of words for the post should be {max_words}.
    Be {tone} in the post!

    {extra_info}
    """,
    )
    recruitment_chain = LLMChain(llm=llm, prompt=my_prompt, output_key="generated_post")

    output = recruitment_chain(
        {
            "social_media": social_media,
            "position": position,
            "tone": tone,
            "max_words": max_words,
            "extra_info": extra_info,
        }
    )

    return output


def extract_info_from_text(openai_api_key, file):
    llm = ChatOpenAI(
        temperature=0, openai_api_key=openai_api_key, model_name=MODEL_NAME
    )

    file_content = file.getvalue().decode("utf-8")

    class PlayerInfo(BaseModel):
        player_name: str = Field(description="This is the name of the player")
        player_role: str = Field(description="This is the football role of the player")
        player_team: str = Field(description="This is the team the player belongs to")
        player_goals: int = Field(
            description="This is an integer that indicates how many goals the player scored"
        )

        @validator("player_team")
        def check_player_team(cls, info):
            if info not in ["Rootsball", "DataFoots"]:
                raise ValueError(
                    f"The player is part of {info} which is not a possible team"
                )
            return info

    class Players(BaseModel):
        players: Sequence[PlayerInfo] = Field(
            ..., description="The players that participated in the game"
        )

    pydantic_parser = PydanticOutputParser(pydantic_object=Players)
    format_instructions = pydantic_parser.get_format_instructions()

    my_prompt = PromptTemplate(
        input_variables=["file_content", "format_instructions"],
        template="""
    You are a diligent assistant whose task is to extract information from a piece of text and parse it in the correct format.
    The following is the piece of text
    '''
    {file_content}
    '''
    {format_instructions}
    """,
    )

    extraction_chain = LLMChain(llm=llm, prompt=my_prompt)

    output = extraction_chain(
        {"file_content": file_content, "format_instructions": format_instructions}
    )

    parsed_output = pydantic_parser.parse(output["text"])
    df = pd.DataFrame([dict(obj) for obj in parsed_output.players])
    return df


def create_vector_store(urls):
    loader = WebBaseLoader(urls)
    dataroots_website = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    documents_chunks = text_splitter.split_documents(dataroots_website)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents_chunks, embeddings)
    return vector_store


def ask_dataroots_chatbot(openai_api_key, temperature, vector_store, user_input):
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(
            temperature=temperature,
            openai_api_key=openai_api_key,
            model_name=MODEL_NAME,
        ),
        retriever=vector_store.as_retriever(),
    )

    output = qa_chain({"question": user_input})
    return output
