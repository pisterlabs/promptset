import json
import logging
from typing import Any, AsyncGenerator, Optional, Union

import aiohttp
import openai
from msgraph.generated.search.query.query_post_request_body import QueryPostRequestBody
from msgraph.generated.models.search_request import SearchRequest
from msgraph.generated.models.entity_type import EntityType
from msgraph.generated.models.search_query import SearchQuery
from approaches.approach import Approach
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit
from core.graphclientbuilder import GraphClientBuilder

class ChatReadRetrieveReadApproach(Approach):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    NO_RESPONSE = "0"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """
    system_message_chat_conversation = """Assistant helps the company employees with their healthcare plan questions, and questions about the employee handbook. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
{follow_up_questions_prompt}
{injected_prompt}
"""
    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about their healthcare plan and employee handbook.
Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
Try not to repeat questions that have already been asked.
Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """Below is a history of previous conversations and new questions from users that need to be searched and answered in the knowledge base about the company.
You have access to the Microsoft Search index, which contains over 100 documents.
Generate a search query based on the conversation and the new question.
Do not include the name of the cited file or document (e.g. info.txt or doc.pdf) in the search query term.
Only display search terms, do not output quotation marks, etc.
Do not include text in [] or <>> in search query terms.
Do not include special characters such as [].
If the question is not in English, generating the search query in the language used in the question.
If you cannot generate a search query, return only the number 0.
"""
    query_prompt_few_shots = [
        {"role": USER, "content": "私のヘルスプランについて教えてください。"},
        {"role": ASSISTANT, "content": "利用可能 ヘルスプラン"},
        {"role": USER, "content": "私のプランには有酸素運動は含まれていますか？"},
        {"role": ASSISTANT, "content": "ヘルスプラン 有酸素運動 適用範囲"},
    ]

    def __init__(
        self,
        openai_host: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        chatgpt_model: str,
    ):
        self.openai_host = openai_host
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run_simple_chat(
        self,
        history: list[dict[str, str]],
        obo_token,
        should_stream: bool = False,
    ) -> tuple:
        # Step.1 ユーザーの入力からクエリを作成する
        original_user_query = history[-1]["content"]
        user_query_request = "Generate search query for: " + original_user_query
        
        query_messages = self.get_messages_from_history(
            system_prompt=self.query_prompt_template,
            model_id=self.chatgpt_model,
            history=history,
            user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - len(user_query_request),
            few_shots=self.query_prompt_few_shots,
        )

        chatgpt_args = {"deployment_id": self.chatgpt_deployment} if self.openai_host == "azure" else {}
        chat_completion = await openai.ChatCompletion.acreate(
            **chatgpt_args,
            model=self.chatgpt_model,
            messages=query_messages,
            temperature=0.0,
            max_tokens=100,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1
        )

        generated_query = chat_completion["choices"][0]["message"]["content"]

        if generated_query == self.NO_RESPONSE: 
            # TODO: クエリがない場合は通常の会話をする
            query_not_found_msg ={
                'choices':[
                    {
                        'message':{
                            'role':"assistant",
                            'content':"あなたの入力には知りたいことが含まれていないようです。何について調べますか？"
                        }
                    }
                ]
            }
            return ({}, query_not_found_msg)

        #print("Generated_query:"+generated_query)

        # Step2. クエリを使ってGraphを検索する
        client = GraphClientBuilder().get_client(obo_token)

        request_body = QueryPostRequestBody(
            requests=[
                SearchRequest(
                    entity_types=[EntityType.ListItem],
                    query=SearchQuery(
                        query_string=generated_query
                    ),
                    size=1 #取得するページのサイズ。いっぱい取得してもtoken上限で使わないので1でいい
                )
            ]
        )
        
        search_result = await client.search.query.post(body = request_body)
        
        #search_resultがない場合は、クエリ生成したクエリを返す
        if search_result.value[0].hits_containers[0].total == 0:
            source_not_found_msg ={
                'choices':[
                    {
                        'message':{
                            'role':"assistant",
                            'content':f"「{generated_query}」で検索しましたが、情報源を見つけられませんでした。"
                        }
                    }
                ]
            }
            return ({}, source_not_found_msg)

        #print(search_result)
        #ここではsummaryをソースにしているが、文章量によってはコンテンツ別にデータを取ったほうがいいかもしれない
        results = [
                hit.resource.id + ": " + hit.summary
                for hit in search_result.value[0].hits_containers[0].hits
        ]
        content = "\n".join(results)

        citaion_source = [
            {
                "id": hit.resource.id,
                "web_url": hit.resource.web_url,
                "hit_id": hit.hit_id,
                "name": hit.resource.name or hit.resource.web_url.split("/")[-1]
            } for hit in search_result.value[0].hits_containers[0].hits
        ]

        # Step3. Graphから取得した結果をから回答を生成する
        response_token_limit = 1024
        messages_token_limit = self.chatgpt_token_limit - response_token_limit
        answer_messages = self.get_messages_from_history(
            system_prompt=self.system_message_chat_conversation,
            model_id=self.chatgpt_model,
            history=history,
            user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=messages_token_limit,
        )

        extra_info = {
            "data_points": citaion_source,
        }

        chat_coroutine = await openai.ChatCompletion.acreate(
            **chatgpt_args,
            model=self.chatgpt_model,
            messages=answer_messages,
            temperature=0,
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
        )
        
        return (extra_info, chat_coroutine)

    
    async def run_without_streaming(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        obo_token,
        session_state: Any = None,
    ) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_simple_chat(
            history, obo_token, should_stream=False
        )

        #extra_info, chat_coroutine = await self.run_until_final_call(
        #    history, overrides, auth_claims, should_stream=False
        #)
        chat_resp = dict(chat_coroutine)
        chat_resp["choices"][0]["context"] = extra_info
        chat_resp["choices"][0]["session_state"] = session_state
        return chat_resp

    async def run_with_streaming(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        obo_token,
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:
        extra_info, chat_coroutine = await self.run_simple_chat(
            history, overrides, should_stream=True
        )
        yield {
            "choices": [
                {
                    "delta": {"role": self.ASSISTANT},
                    "context": extra_info,
                    "session_state": session_state,
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "object": "chat.completion.chunk",
        }

        async for event in await chat_coroutine:
            # "2023-07-01-preview" API version has a bug where first response has empty choices
            if event["choices"]:
                yield event

    async def run(
        self, messages: list[dict], stream: bool = False, session_state: Any = None, context: dict[str, Any] = {}
    ) -> Union[dict[str, Any], AsyncGenerator[dict[str, Any], None]]:
        overrides = context.get("overrides", {})
        obo_token = context.get("obo_token", {})
        if stream is False:
            # Workaround for: https://github.com/openai/openai-python/issues/371
            async with aiohttp.ClientSession() as s:
                openai.aiosession.set(s)
                response = await self.run_without_streaming(messages, overrides,obo_token, session_state)
            return response
        else:
            return self.run_with_streaming(messages, overrides, obo_token, session_state)

    def get_messages_from_history(
        self,
        system_prompt: str,
        model_id: str,
        history: list[dict[str, str]],
        user_content: str,
        max_tokens: int,
        few_shots=[],
    ) -> list:
        message_builder = MessageBuilder(system_prompt, model_id)

        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        for shot in few_shots:
            message_builder.append_message(shot.get("role"), shot.get("content"))

        append_index = len(few_shots) + 1

        message_builder.append_message(self.USER, user_content, index=append_index)
        total_token_count = message_builder.count_tokens_for_message(message_builder.messages[-1])

        newest_to_oldest = list(reversed(history[:-1]))
        for message in newest_to_oldest:
            potential_message_count = message_builder.count_tokens_for_message(message)
            if (total_token_count + potential_message_count) > max_tokens:
                logging.debug("Reached max tokens of %d, history will be truncated", max_tokens)
                break
            message_builder.append_message(message["role"], message["content"], index=append_index)
            total_token_count += potential_message_count
        return message_builder.messages

    def get_search_query(self, chat_completion: dict[str, Any], user_query: str):
        response_message = chat_completion["choices"][0]["message"]
        if function_call := response_message.get("function_call"):
            if function_call["name"] == "search_sources":
                arg = json.loads(function_call["arguments"])
                search_query = arg.get("search_query", self.NO_RESPONSE)
                if search_query != self.NO_RESPONSE:
                    return search_query
        elif query_text := response_message.get("content"):
            if query_text.strip() != self.NO_RESPONSE:
                return query_text
        return user_query
    


    '''参考元コード
    async def run_until_final_call(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple:
        # content = clientからのリクエストボディのmessages→ユーザーからの最新の入力を取得している
        original_user_query = history[-1]["content"]
        # 検索クエリを作るためのリクエストを作成
        user_query_request = "Generate search query for: " + original_user_query


        # Doc検索のためのファンクションを定義
        functions = [
            {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure Cognitive Search index",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                        }
                    },
                    "required": ["search_query"],
                },
            }
        ]

        # STEP 1: チャット履歴と最後の質問に基づいて、最適化されたキーワード検索クエリを生成します。
        # システムプロンプトにクエリ生成用のテンプレートをセットしてクエリを生成するためのメッセージリストを生成
        messages = self.get_messages_from_history(
            system_prompt=self.query_prompt_template,
            model_id=self.chatgpt_model,
            history=history,
            user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - len(user_query_request),
            few_shots=self.query_prompt_few_shots,
        )

        chatgpt_args = {"deployment_id": self.chatgpt_deployment} if self.openai_host == "azure" else {}
        chat_completion = await openai.ChatCompletion.acreate(
            **chatgpt_args,
            model=self.chatgpt_model,
            messages=messages,
            temperature=0.0,
            max_tokens=100,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
            functions=functions,
            function_call="auto",
        )

        # GPTから得られた結果(chat_completion)からFunction Calling用の引数またはGPTの返信そのものを使用してクエリを取得する。クエリが生成できなかった場合(chat_completion=0)は、ユーザーの質問をそのままクエリとする。
        query_text = self.get_search_query(chat_completion, original_user_query)

        # STEP 2: GPTに最適化されたクエリで検索インデックスから関連文書を取得する。
        follow_up_questions_prompt = (
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        )

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(
                injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt
            )
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(
                injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt
            )
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

        response_token_limit = 1024
        messages_token_limit = self.chatgpt_token_limit - response_token_limit
        messages = self.get_messages_from_history(
            system_prompt=system_message,
            model_id=self.chatgpt_model,
            history=history,
            # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            #Document 1: This is the content of document 1.
            #Document 2: This is the content of document 2.
            user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=messages_token_limit,
        )
        msg_to_display = "\n\n".join([str(message) for message in messages])

        #「sourcepage_field」フィールドには、検索結果が見つかったページまたはドキュメントの名前が含まれており、「content_field」フィールドには、検索結果の実際のコンテンツが含まれています。
        #サンプルresults = [
        #    "Document 1: This is the content of document 1.",
        #    "Document 2: This is the content of document 2."
        #]
        extra_info = {
            "data_points": results,
            "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>"
            + msg_to_display.replace("\n", "<br>"),
        }

        chat_coroutine = openai.ChatCompletion.acreate(
            **chatgpt_args,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.7,
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
        )
        return (extra_info, chat_coroutine)
    '''
