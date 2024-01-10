from typing import Any, List, Optional, Tuple, Type

import re

from halo import Halo
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models.base import BaseChatModel
from langchain.tools import BaseTool, Tool
from pydantic.v1 import BaseModel, Field
from tenacity import RetryError

from chatflock.backing_stores import InMemoryChatDataBackingStore
from chatflock.base import Chat
from chatflock.conductors import RoundRobinChatConductor
from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.participants.user import UserChatParticipant
from chatflock.renderers import NoChatRenderer
from chatflock.structured_string import Section, StructuredString
from chatflock.web_research.errors import NonTransientHTTPError, TransientHTTPError
from chatflock.web_research.page_analyzer import PageQueryAnalyzer
from chatflock.web_research.search import SearchResultsProvider

video_watch_urls_patterns = [
    r"youtube.com/watch\?v=([a-zA-Z0-9_-]+)",
    r"youtu.be/([a-zA-Z0-9_-]+)",
    r"vimeo.com/([0-9]+)",
    r"dailymotion.com/video/([a-zA-Z0-9]+)",
    r"dailymotion.com/embed/video/([a-zA-Z0-9]+)",
    r"tiktok.com/@([a-zA-Z0-9_]+)/video/([0-9]+)",
]


def url_unsupported(url):
    # List of unsupported file types
    unsupported_types = ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "rtf", "jpg", "png", "gif"]

    # Extract file extension from the URL
    file_extension = re.findall(r"\.([a-zA-Z0-9]+)(?:[\?\#]|$)", url)

    # Check if the file extension is in the list of unsupported types
    if file_extension and file_extension[0] in unsupported_types:
        return True

    # Check if URL is a video or video site
    for pattern in video_watch_urls_patterns:
        if re.search(pattern, url):
            return True

    return False


class WebSearch:
    def __init__(
        self,
        chat_model: BaseChatModel,
        search_results_provider: SearchResultsProvider,
        page_query_analyzer: PageQueryAnalyzer,
        skip_results_if_answer_snippet_found: bool = True,
    ):
        self.chat_model = chat_model
        self.search_results_provider = search_results_provider
        self.page_query_analyzer = page_query_analyzer
        self.skip_results_if_answer_snippet_found = skip_results_if_answer_snippet_found

    def get_answer(
        self, query: str, n_results: int = 3, urls: Optional[List[str]] = None, spinner: Optional[Halo] = None
    ) -> Tuple[bool, str]:
        original_spinner_text = None if spinner is None else spinner.text
        qna = []

        if urls is None:
            if spinner is not None:
                spinner.start(f'Getting search results for "{query}"...')

            try:
                search_results = self.search_results_provider.search(query=query, n_results=n_results)
            except (TransientHTTPError, NonTransientHTTPError) as e:
                return False, f'Failed to get search results for "{query}" because of an error: {e}'

            if spinner is not None:
                spinner.succeed(f'Got search results for "{query}".')

            if len(search_results.organic_results) == 0 and search_results.answer_snippet is None:
                return False, "Nothing was found on the web for this query."

            if search_results.knowledge_graph_description is not None:
                qna.append({"answer": search_results.knowledge_graph_description, "source": "Knowledge Graph"})

            if search_results.answer_snippet is not None:
                qna.append({"answer": search_results.answer_snippet, "source": "Answer Snippet"})

            if not self.skip_results_if_answer_snippet_found or search_results.answer_snippet is None:
                for result in search_results.organic_results:
                    if url_unsupported(result.link):
                        continue

                    if spinner is not None:
                        spinner.start(f'Reading & analyzing #{result.position} result "{result.title}"')

                    try:
                        page_result = self.page_query_analyzer.analyze(
                            url=result.link, title=result.title, query=query, spinner=spinner
                        )
                        answer = page_result.answer

                        if spinner is not None:
                            spinner.succeed(f'Read & analyzed #{result.position} result "{result.title}".')
                    except Exception as e:
                        if type(e) in (RetryError, TransientHTTPError, NonTransientHTTPError):
                            if spinner is not None:
                                spinner.warn(
                                    f'Failed to read & analyze #{result.position} result "{result.title}", moving on.'
                                )

                            answer = "Unable to answer query because the page could not be read."
                        else:
                            raise

                    qna.append({"answer": answer, "source": result.link})
        else:
            # Urls were provided, search in those urls instead of searching using a search engine
            for url in urls:
                if url_unsupported(url):
                    continue

                if spinner is not None:
                    spinner.start(f'Reading & analyzing URL "{url}"')

                try:
                    page_result = self.page_query_analyzer.analyze(
                        url=url, title="Unknown", query=query, spinner=spinner
                    )
                    answer = page_result.answer

                    if spinner is not None:
                        spinner.succeed(f'Read & analyzed URL "{url}".')
                except Exception as e:
                    if type(e) in (RetryError, TransientHTTPError, NonTransientHTTPError):
                        if spinner is not None:
                            spinner.warn(f'Failed to read & analyze URL "{url}", moving on.')

                        answer = "Unable to answer query because the page could not be read."
                    else:
                        raise

                qna.append({"answer": answer, "source": url})

        if spinner is not None:
            spinner.start(f"Processing results...")

        formatted_answers = "\n".join([f'{i + 1}. {q["answer"]}; Source: {q["source"]}' for i, q in enumerate(qna)])

        chat = Chat(
            backing_store=InMemoryChatDataBackingStore(),
            renderer=NoChatRenderer(),
            initial_participants=[
                UserChatParticipant(),
                LangChainBasedAIChatParticipant(
                    name="Query Answer Aggregator",
                    role="Query Answer Aggregator",
                    personal_mission="Analyze query answers, discard unlikely ones, and provide an aggregated final response.",
                    chat_model=self.chat_model,
                    other_prompt_sections=[
                        Section(
                            name="Aggregating Query Answers",
                            sub_sections=[
                                Section(
                                    name="Process",
                                    list=[
                                        "Receive query and answers with sources.",
                                        "Analyze answers, discard unlikely or minority ones.",
                                        "Formulate final answer based on most likely answers.",
                                        'If no data found, respond "The answer could not be found."',
                                    ],
                                    list_item_prefix=None,
                                ),
                                Section(
                                    name="Aggregation",
                                    list=[
                                        "Base final answer on sources.",
                                        "Incorporate sources as inline citations in Markdown format.",
                                        'Example: "Person 1 was [elected president in 2012](https://...)."',
                                        "Only include sources from provided answers.",
                                        "If part of an answer is used, use the same links inline.",
                                    ],
                                ),
                                Section(
                                    name="Final Answer Notes",
                                    list=[
                                        "Do not fabricate information. Stick to provided data.",
                                        "You will be given the top search results from a search engine, there is a reason they are the top results. You should pay attention to all of them and think about the query intent."
                                        "If the answer is not found in the page data, state it clearly.",
                                        "Should be formatted in Markdown with inline citations.",
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
            max_total_messages=2,
        )
        chat_conductor = RoundRobinChatConductor()
        final_answer = chat_conductor.initiate_dialog(
            chat=chat,
            initial_message=str(
                StructuredString(
                    sections=[Section(name="Query", text=query), Section(name="Answers", text=formatted_answers)]
                )
            ),
        )

        if spinner is not None:
            spinner.succeed(f"Done searching the web.")

            if original_spinner_text is not None:
                spinner.start(original_spinner_text)

        return True, final_answer


class WebSearchToolArgs(BaseModel):
    query: str = Field(
        description="The query to search the web for (or what to look for in the page in case urls are provided)."
    )
    urls: Optional[List[str]] = Field(
        description="A list of urls to search for the query in. If provided, the query will be searched in these urls. If not provided, the query will be searched in the top search results from a search engine. Provide urls only when the user mentions a URL (if applicable)"
    )


class WebResearchTool(BaseTool):
    web_search: WebSearch
    n_results: int = 3
    spinner: Optional[Halo] = None
    name: str = "web_search"
    description: str = "Research the web. Use that to get an answer for a query you don't know or unsure of the answer to, for recent events, or if the user asks you to. This will evaluate answer snippets, knowledge graphs, and the top N results from google and aggregate a result."
    args_schema: Type[BaseModel] = WebSearchToolArgs
    progress_text: str = "Searching the web..."

    def _run(
        self,
        query: str,
        urls: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        return self.web_search.get_answer(query=query, n_results=self.n_results, urls=urls, spinner=self.spinner)[1]
