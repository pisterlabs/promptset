from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


def get_llm() -> ChatOpenAI:
    load_dotenv()
    # Base model uses is gpt3.5-turbo
    return ChatOpenAI(temperature=0)  # type: ignore


def gpt_summary(llm: ChatOpenAI, city_text: str, city: str) -> str:
    city_string = f"Combine all the summaries on {city} provided within backticks "
    combine_prompt = PromptTemplate(
        template=(
            city_string
            + """```{text}```.
            Can you summarize it as a tourist destination in 8-10 sentences.\n"
            """
        ),
        input_variables=["text"],
    )

    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(city_text)

    docs = [Document(page_content=t) for t in texts]

    chain = load_summarize_chain(
        llm, chain_type="map_reduce", combine_prompt=combine_prompt
    )
    return chain.run(docs)
