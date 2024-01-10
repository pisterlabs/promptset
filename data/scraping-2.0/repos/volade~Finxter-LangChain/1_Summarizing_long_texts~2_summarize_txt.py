from decouple import config
from langchain.chains import (
    LLMChain,
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
    StuffDocumentsChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from prompt_templates import map_prompt, reduce_prompt
from utils import get_tokens

model = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=config("OPENAI_API_KEY"),
)


def load_text(path: str) -> list[Document]:
    text = TextLoader(path).load()
    print(f"Loaded text contains {get_tokens(text[0].page_content)} tokens")
    return text


speech: list[Document] = load_text("data/speech.txt")


def split_text(text: list[Document]) -> list[Document]:
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000)
    return text_splitter.split_documents(text)


speech_chunks: list[Document] = split_text(speech)


map_prompt = PromptTemplate.from_template(map_prompt)
single_map_call = LLMChain(llm=model, prompt=map_prompt)


reduce_prompt = PromptTemplate.from_template(reduce_prompt)
single_reduce_call = LLMChain(llm=model, prompt=reduce_prompt)

single_stuff_and_reduce_call = StuffDocumentsChain(
    llm_chain=single_reduce_call, document_variable_name="text_summaries"
)

send_groups_to_single_stuff_and_reduce_call = ReduceDocumentsChain(
    combine_documents_chain=single_stuff_and_reduce_call,
    token_max=3500,
)


map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=single_map_call,
    reduce_documents_chain=send_groups_to_single_stuff_and_reduce_call,
    document_variable_name="text_chunk",
    return_intermediate_steps=True,
)


with open("data/test_output.py", "w") as f:
    print(map_reduce_chain.invoke(speech_chunks), file=f)
