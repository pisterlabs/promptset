from __future__ import annotations

from typing import AsyncIterable

from fastapi_poe import PoeBot
from fastapi_poe.client import MetaMessage, stream_request
from fastapi_poe.types import QueryRequest
from langchain.document_loaders import RedditPostsLoader
from sse_starlette.sse import ServerSentEvent

REDDIT_CLIENT_ID = "XOAl2Nb8FjPmtPblFZHBew"
REDDIT_CLIENT_SECRET = "ZY7FjOcw03rMyZcm8OuHJcnAF-2bZQ"

PROMPT_TEMPLATE = """
You are given posts from Subreddit {subreddit}.
Write a summary for the theme of the subreddit from the posts."""


class RedditSummaryBot(PoeBot):
    def __init__(self, *, reddit_client_id, reddit_client_secret):
        super().__init__()
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret

    async def get_response(self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        subreddit = query.query[-1].content

        try:
            loader = RedditPostsLoader(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent="tomzhou94-test",
                categories=["hot"],  # List of categories to load posts from
                mode="subreddit",
                search_queries=[subreddit],  # List of subreddits to load posts from
                number_posts=10,  # Default value is 10
            )
            documents = loader.load()
        except Exception:
            yield self.text_event(f"\n\n**Could not find Subreddit: {subreddit}**\n")
            yield self.text_event("\n\n**Please try again**\n")
            return

        # truncate to the first 200 chars per post.
        post_tags = [
            f"<post><title>{document.metadata['post_title']}</title>"
            f"<content>{document.page_content[:200]}</content></post>"
            for document in documents
        ]
        query_content = PROMPT_TEMPLATE + "".join(post_tags)
        query.query[-1].content = query_content
        yield self.text_event(f"\n\n**Summary of Subreddit {subreddit}**:\n")
        async for msg in stream_request(query, "Claude-instant", query.access_key):
            if isinstance(msg, MetaMessage):
                continue
            elif msg.is_suggested_reply:
                yield self.suggested_reply_event(msg.text)
            elif msg.is_replace_response:
                yield self.replace_response_event(msg.text)
            else:
                yield self.text_event(msg.text)
