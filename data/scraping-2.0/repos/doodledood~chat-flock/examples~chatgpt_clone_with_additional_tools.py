import typer
from dotenv import load_dotenv
from halo import Halo
from langchain.text_splitter import TokenTextSplitter

from chatflock.backing_stores import InMemoryChatDataBackingStore
from chatflock.base import Chat
from chatflock.code import LocalCodeExecutor
from chatflock.code.langchain import CodeExecutionTool
from chatflock.conductors.round_robin import RoundRobinChatConductor
from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.participants.user import UserChatParticipant
from chatflock.renderers.terminal import TerminalChatRenderer
from chatflock.web_research import WebSearch
from chatflock.web_research.page_analyzer import OpenAIChatPageQueryAnalyzer
from chatflock.web_research.page_retrievers.selenium_retriever import SeleniumPageRetriever
from chatflock.web_research.search import GoogleSerperSearchResultsProvider
from chatflock.web_research.web_research import WebResearchTool
from examples.common import create_chat_model, get_max_context_size


def chatgpt_clone_with_additional_tools(
    model: str = "gpt-4-1106-preview",
    model_for_page_analysis: str = "gpt-3.5-turbo-1106",
    temperature: float = 0.0,
    temperature_for_page_analysis: float = 0.0,
) -> None:
    chat_model = create_chat_model(model=model, temperature=temperature)
    chat_model_for_page_analysis = create_chat_model(
        model=model_for_page_analysis, temperature=temperature_for_page_analysis
    )

    max_context_size_for_page_analysis = get_max_context_size(chat_model_for_page_analysis) or 12_000

    page_retriever = SeleniumPageRetriever()
    web_search = WebSearch(
        chat_model=chat_model,
        search_results_provider=GoogleSerperSearchResultsProvider(),
        page_query_analyzer=OpenAIChatPageQueryAnalyzer(
            chat_model=chat_model_for_page_analysis,
            # Should `pip install selenium webdriver_manager` to use this
            page_retriever=page_retriever,
            text_splitter=TokenTextSplitter(
                chunk_size=max_context_size_for_page_analysis, chunk_overlap=max_context_size_for_page_analysis // 5
            ),
            use_first_split_only=True,
        ),
    )

    spinner = Halo(spinner="dots")
    ai = LangChainBasedAIChatParticipant(
        name="Assistant",
        chat_model=chat_model,
        tools=[
            CodeExecutionTool(executor=LocalCodeExecutor(spinner=spinner), spinner=spinner),
            WebResearchTool(web_search=web_search, n_results=3, spinner=spinner),
        ],
        spinner=spinner,
    )

    user = UserChatParticipant(name="User")
    participants = [user, ai]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(), renderer=TerminalChatRenderer(), initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    chat_conductor.initiate_dialog(chat=chat)

    page_retriever.close()


if __name__ == "__main__":
    load_dotenv()

    typer.run(chatgpt_clone_with_additional_tools)
