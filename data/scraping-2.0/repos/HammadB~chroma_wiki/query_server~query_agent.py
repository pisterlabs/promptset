import tiktoken
import openai
from typing import Optional
from pydantic import BaseModel
from enum import Enum
from wikidb import WikipediaDatabase
import wikitextparser as wtp
from prompts import answer_with_context, wikipedia_query_generation, chat_entry_template, chat_with_context_template, chat_summarize_template
from token_consts import MAX_COMPLETION_TOKENS

ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

class Author(Enum):
    AGENT = 0
    USER = 1

class ChatEntry(BaseModel):
    content: str
    author: Author
    context: Optional[str]
    isTransient: Optional[bool]
    isStop: Optional[bool]


class QueryAgent():

    def __init__(self, dataframe_path: str, index_path: str):
        self._wikidb = WikipediaDatabase(dataframe_path=dataframe_path, index_path=index_path)
        self.tokenizer = tiktoken.get_encoding(ENCODING)

    # Return response types
    def query(self, query) -> str:
        return self.answer_query_with_context(query)

    def _do_completion(self, query=False, max_tokens=MAX_COMPLETION_TOKENS) -> str:
        COMPLETIONS_MODEL = "text-davinci-003"
        COMPLETIONS_API_PARAMS = {
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "model": COMPLETIONS_MODEL,
        }

        # TODO: handle errors
        response = openai.Completion.create(
            prompt=query,
            **COMPLETIONS_API_PARAMS
        )

        return response["choices"][0]["text"]

    def generate_searches_for_wikipedia(self, query: str) -> list[str]:
        wiki_queries = self._do_completion(
            wikipedia_query_generation.format(question=query)).split("\n")
        wiki_queries = [q.strip()
                        for q in wiki_queries if (q is not None and q.strip() != '')] + [query]
        return wiki_queries
    
    def get_relevant_wikipedia_pages(self, wiki_queries: list[str]) -> list[tuple[str, wtp.WikiText]]:
        res = []
        articles = []
        for wiki_query in wiki_queries:
            relevant_articles = self._wikidb.search(wiki_query)
            articles.extend(relevant_articles)
        # Filter duplicates
        articles = [*set(articles)]
        for title in articles:
            page = self._wikidb.get_page(title)
            if page:
                res.append((title, page))
        return res

    def update_index_for_query(self, query: str):
        wiki_queries = self.generate_searches_for_wikipedia(query)
        print(f"Queries {wiki_queries}")
        titles_pages = self.get_relevant_wikipedia_pages(wiki_queries)
        for title, page in titles_pages:
            self._wikidb.add_page(title, page)

    def update_index_for_query_streaming(self, query: str):
        wiki_queries = self.generate_searches_for_wikipedia(query)
        titles_pages = self.get_relevant_wikipedia_pages(wiki_queries)
        for title, page in titles_pages:
            yield title
            self._wikidb.add_page(title, page)

    def answer_query_with_context(self, query: str) -> ChatEntry:
        context = self.get_context_for_question(query)
        prompt = answer_with_context.format(context=context, question=query)

        # TODO: fuzzy match
        answer = self._do_completion(prompt).strip("\n").strip()
        if answer == "I don't know.":
            self.update_index_for_query(query)

            # Try again
            # TODO: maybe forcefully insert neighbors
            context = self.get_context_for_question(query)
            prompt = answer_with_context.format(
                context=context, question=query)
            answer = self._do_completion(prompt).strip("\n")

        return ChatEntry(content=answer, author=Author.AGENT, context=context)

    def summarize_chat(self, chat: list[ChatEntry]) -> str:
        formatted_chats = []
        i = 0
        # TODO: use a token budget for history
        while i < len(chat) - 1:
            formatted_chats.append(chat_entry_template.format(
                human_text=chat[i].content, agent_text=chat[i+1].content))
            i += 2
        query = chat[len(chat)-1].content
        history = "\n".join(formatted_chats)

        if len(chat) > 2:
            summarize_history_prompt = chat_summarize_template.format(
                chat_history=history, question=query)
            query = self._do_completion(
                summarize_history_prompt).strip("\n").strip()
            print(f"New query is {query}")
        return query

    def answer_chat_query(self, query) -> tuple[str]:
        context = self.get_context_for_question(query)
        chat_prompt = chat_with_context_template.format(
            context=context, question=query)
        answer = self._do_completion(chat_prompt).strip("\n").strip()
        return answer, context

    def chat(self, chat: list[ChatEntry]) -> ChatEntry:
        query = self.summarize_chat(chat)
        answer, context = self.answer_chat_query(query)
        if answer == "I don't know.":
            self.update_index_for_query(query)
            context = self.get_context_for_question(query)
            prompt = chat_with_context_template.format(
                context=context, question=query)
            answer = self._do_completion(prompt).strip("\n")

        return ChatEntry(content=answer, author=Author.AGENT, context=context)

    def chat_streaming(self, chat: list[ChatEntry]):
        query = self.summarize_chat(chat)
        answer, context = self.answer_chat_query(query)
        if answer == "I don't know.":
            yield ChatEntry(content="I don't know, let me see if I can find out", author=Author.AGENT, context=context)
            for title in self.update_index_for_query_streaming(query):
                yield ChatEntry(content=f"I'm reading... {title}", author=Author.AGENT, context='', isTransient=True, isStop=False)
            yield ChatEntry(content=f"I'm synthesizing what I just read...", author=Author.AGENT, context='', isTransient=True, isStop=False)
            answer, context = self.answer_chat_query(query)
            yield ChatEntry(content=answer, author=Author.AGENT, context=context, isStop=True, isTransient=True)
        yield ChatEntry(content=answer, author=Author.AGENT, context=context, isStop=True, isTransient=False)

    def get_context_for_question(self, question: str) -> tuple[str, str]:
        """
        Fetch relevant sections and insert as many as possible
        """

        MAX_SECTION_LEN = 2000
        SEPARATOR = "\n* "

        encoding = tiktoken.get_encoding(ENCODING)
        separator_len = len(encoding.encode(SEPARATOR))

        most_relevant_indices = self._wikidb.index.get_closest_indices(
            question)
        print(f"found: {len(most_relevant_indices)} neighbors")
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        for section_index in most_relevant_indices:
            # Add contexts until we run out of space.
            document_section = self._wikidb.get_section_by_id(section_index)
            chosen_sections_len += document_section.tokens + separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                break

            chosen_sections.append(
                SEPARATOR + document_section.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))

        context = "".join(chosen_sections)
        return context
