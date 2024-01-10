import re
import json
from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
from azure.search.documents.models import Vector

from approaches.approach import ChatApproach
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit
from text import nonewlines
from approaches.prompt_data import (
    USER,
    ASSISTANT,
    ai_response_prompt_template,
    follow_up_questions_prompt_content,
    search_query_prompt,
    search_query_prompt_few_shots,
    response_val_prompt,
    source_clf_prompt,
)


class ChatReadRetrieveReadApproach(ChatApproach):
    RESPONSE_TEMP = 0.1
    VALIDATION_TEMP = 0.1
    id_to_data_source = {
        "1": "SoftServe Website",
        "2": "Wikipedia",
        "0": "Unknown",
    }

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    def __init__(
        self,
        search_client: SearchClient,
        chatgpt_deployment: str,
        chatgpt_model: str,
        gpt4_deployment: str,
        gpt4_model: str,
        embedding_deployment: str,
        sourcepage_field: str,
        content_field: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.gpt4_deployment = gpt4_deployment
        self.gpt4_model = gpt4_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)
        self.gpt4_token_limit = get_token_limit(gpt4_model)

    async def run(
        self, history: list[dict[str, str]], overrides: dict[str, Any]
    ) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in [
            "vectors",
            "hybrid",
            None,
        ]
        use_semantic_captions = (
            True if overrides.get("semantic_captions") and has_text else False
        )
        top = overrides.get("top") or 3
        original_user_question = history[-1]["user"]
        print(f">>>>>>>HISTORY:\n\n{json.dumps(history, indent=4)}\n\n")

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        search_query_messages = self.get_messages_from_history(
            search_query_prompt,
            self.chatgpt_model,
            history,
            "Generate search query for: " + original_user_question,
            search_query_prompt_few_shots,
            max_tokens=self.chatgpt_token_limit - len(search_query_prompt),
        )

        search_query_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=search_query_messages,
            temperature=0.0,
            max_tokens=32,
            n=1,
        )

        search_query = search_query_completion.choices[0].message.content
        if search_query.strip() == "0":
            search_query = history[-1][
                "user"
            ]  # Use the last user input if we failed to generate a better query
        print(f">>>>>>>SEARCH_QUERY: {search_query}\n\n")

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (
                await openai.Embedding.acreate(
                    engine=self.embedding_deployment, input=search_query
                )
            )["data"][0]["embedding"]
            title_embedding = Vector(
                value=query_vector, k=top, fields="title_embedding"
            )
            content_embedding = Vector(
                value=query_vector, k=top, fields="content_embedding"
            )
            summary_embedding = Vector(
                value=query_vector, k=top, fields="summary_embedding"
            )
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            search_query = None

        # Classify the data source that is most likely to contain the answer
        source_clf_messages = self.get_messages_from_history(
            source_clf_prompt,
            self.chatgpt_model,
            # history,
            [],
            # "User question: " + original_user_question,
            "User Question: " + search_query,
            max_tokens=self.chatgpt_token_limit - len(source_clf_prompt),
        )

        source_clf_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=source_clf_messages,
            temperature=0.0,
            max_tokens=32,
            n=1,
        )
        data_source_ids = source_clf_completion.choices[0].message.content
        data_sources = [
            self.id_to_data_source.get(data_source_id.strip(), "Unknown")
            for data_source_id in data_source_ids.split(",")
        ]

        filter = (
            f"search.in(Storage, '{','.join(data_sources)}', ',')"
            if "Unknown" not in data_sources
            else None
        )
        print(f">>>>>>>SEARCH_FILTER: {filter}\n\n")

        # Use semantic L2 reranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(
                search_query,
                filter=filter,
                query_type=QueryType.SEMANTIC,
                query_language="en-us",
                query_speller="lexicon",
                semantic_configuration_name="default",
                top=top,
                query_caption=None,
                vectors=[title_embedding, content_embedding, summary_embedding]
                if query_vector
                else None,
            )
        else:
            r = await self.search_client.search(
                search_query,
                filter=filter,
                top=top,
                vectors=[title_embedding, content_embedding, summary_embedding]
                if query_vector
                else None,
            )
        if use_semantic_captions:
            results_formatted = [
                {
                    "title": doc["FileName"],
                    "content": nonewlines(
                        " . ".join([c.text for c in doc["@search.captions"]])
                    ),
                    "summary": nonewlines(doc["Summary"]),
                    "url": doc[self.sourcepage_field],
                }
                async for doc in r
            ]
        else:
            results_formatted = [
                {
                    "title": doc["FileName"],
                    "content": nonewlines(doc[self.content_field]),
                    "summary": nonewlines(doc["Summary"]),
                    "url": doc[self.sourcepage_field],
                }
                async for doc in r
            ]
        print(
            f">>>>>>>RESULTS_FORMATTED:\n\n{json.dumps(results_formatted, indent=4)}\n\n"
        )

        results = [
            res["url"] + ": " + res["content"] for res in results_formatted
        ]
        content = "\n".join(results)

        follow_up_questions_prompt = (
            follow_up_questions_prompt_content
            if overrides.get("suggest_followup_questions")
            else ""
        )

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_override")
        if prompt_override is None:
            ai_response_prompt = ai_response_prompt_template.format(
                injected_prompt="",
                follow_up_questions_prompt=follow_up_questions_prompt,
            )
        elif prompt_override.startswith(">>>"):
            ai_response_prompt = ai_response_prompt_template.format(
                injected_prompt=prompt_override[3:] + "\n",
                follow_up_questions_prompt=follow_up_questions_prompt,
            )
        else:
            ai_response_prompt = prompt_override.format(
                follow_up_questions_prompt=follow_up_questions_prompt
            )

        ai_response_messages = self.get_messages_from_history(
            ai_response_prompt,
            self.gpt4_model,
            history,
            "User Question:\n"
            + original_user_question
            + "\n\nSources:\n"
            + content,  # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            max_tokens=self.gpt4_token_limit - len(ai_response_prompt),
        )
        print(
            f">>>>>>>AI_RESPONSE_MESSAGES:\n\n{json.dumps(ai_response_messages, indent=4)}\n\n"
        )

        ai_response_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.gpt4_deployment,
            model=self.gpt4_model,
            messages=ai_response_messages,
            temperature=overrides.get("temperature") or self.RESPONSE_TEMP,
            max_tokens=1024,
            n=1,
        )

        ai_response = ai_response_completion.choices[0].message.content
        print(f">>>>>>>ORIGINAL_RESPONSE:\n\n{ai_response}\n\n")

        msg_to_display = "\n\n".join(
            [json.dumps(message, indent=4) for message in ai_response_messages]
        )

        # STEP 4: Validate the response and make sure it complies with the rules
        response_val_messages = self.get_messages_from_history(
            response_val_prompt,
            self.chatgpt_model,
            [],
            "Response:\n"
            + ai_response
            + "\n\n\Source URLs:\n"
            + "\n".join([res["url"] for res in results_formatted]),
            max_tokens=self.chatgpt_token_limit - len(response_val_prompt),
        )
        response_val_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=response_val_messages,
            temperature=self.VALIDATION_TEMP,
            max_tokens=1024,
            n=1,
        )
        validated_response = response_val_completion.choices[0].message.content
        print(f">>>>>>>VALIDATED_RESPONSE:\n\n{validated_response}\n\n")

        thoughts = (
            f"Searched for:<br>{search_query}<br><br>Conversations:<br>"
            + msg_to_display.replace("\n", "<br>").replace("\\n", "<br>")
        )

        return {
            "data_points": results,
            "results_formatted": results_formatted,
            "answer": validated_response,
            "thoughts": thoughts,
        }

    def get_messages_from_history(
        self,
        system_prompt: str,
        model_id: str,
        history: list[dict[str, str]],
        user_conv: str,
        few_shots=[],
        max_tokens: int = 4096,
    ) -> list:
        message_builder = MessageBuilder(system_prompt, model_id)

        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        for shot in few_shots:
            message_builder.append_message(
                shot.get("role"), shot.get("content")
            )

        user_content = user_conv
        append_index = len(few_shots) + 1

        message_builder.append_message(USER, user_content, index=append_index)

        for h in reversed(history[:-1]):
            if bot_msg := h.get("bot"):
                message_builder.append_message(
                    ASSISTANT, bot_msg, index=append_index
                )
            if user_msg := h.get("user"):
                message_builder.append_message(
                    USER, user_msg, index=append_index
                )
            if message_builder.token_length > max_tokens:
                break

        messages = message_builder.messages
        return messages
