from langchain.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.chat_model import ChatCallbackHandler, ChatModel
from src.manage import intro
from src.utils import load_file


def configure_chat_model():
    chat_model = ChatModel()
    chat_model.llm = ChatOllama(
        model="mistral:latest",
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )
    messages = [
        SystemMessagePromptTemplate.from_template(
            load_file("./prompt_templates/private_gpt/system_message.txt")
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    chat_model.prompt = ChatPromptTemplate.from_messages(messages=messages)
    chat_model.memory_llm = ChatOllama()
    return chat_model


def run_chat_session(chat_model):
    embeddings = OllamaEmbeddings(model="mistral:latest")
    intro_config = {
        "page_title": "PrivateGPT",
        "page_icon": "🔒",
        "title": "PrivateGPT",
        "markdown": load_file("./markdowns/private_gpt.md"),
        "history_file_path": "./.cache/chat_history/history.json",
        "prompt": chat_model.prompt,
        "llm": chat_model.llm,
        "chat_session_args": {
            "_file_path": "private_files",
            "_cache_dir": "private_embeddings",
            "_embeddings": embeddings,
        },
    }
    intro(**intro_config)


def main() -> None:
    chat_model = configure_chat_model()
    chat_model.configure_chat_memory(chat_model.memory_llm)
    run_chat_session(chat_model)


if __name__ == "__main__":
    main()
