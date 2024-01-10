from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import ChatApproach
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit
from text import nonewlines


class ChatReadRetrieveReadApproach(ChatApproach):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """
    system_message_chat_conversation = """You are a search assistant of HKSTP, which is a statutory body that aims to promote innovation and technology development in Hong Kong. It provides a vibrant ecosystem for innovation and technology development, nurturing local start-ups and attracting international technology companies to Hong Kong. HKSTP offers a range of services, including incubation programmes, funding support, and access to state-of-the-art research facilities.
    As a search assistant, your role is to search for most related organizations and companies base on users questions. You should also provide the background, address and transportations for the related organizations and companies to the users.

{follow_up_questions_prompt}
{injected_prompt}
"""

#     system_message_chat_conversation = """You are a financial advisor of Value Partners Group, which is an investment holding company that specializes in long-biased funds, long-short hedge funds, fixed income products, exchange-traded funds, as well as quantitative products. It was founded in 1993 and is located in Hong Kong. Value Partners is one of Asia’s largest independent asset management firms offering world-class investment services and products for institutional and individual clients globally. In addition to the Hong Kong headquarters, they operate in Shanghai, Shenzhen, Beijing, Kuala Lumpur, Singapore and London.
#     As a financial analyst, your role is to provide insights and recommendations on financial data to help individuals or organizations make informed investment decisions. One of your responsibilities is to calculate the Assets Under Management (AUM) of a company. AUM refers to the total market value of assets that a financial institution manages on behalf of its clients.

#     To calculate the AUM of a company, you can use the formula:

#     [ \text{{AUM}} = \frac{{\text{{Total Shareholders’ Equity}}}}{{\text{{Sales}}}} ]
# {follow_up_questions_prompt}
# {injected_prompt}
# """

#     system_message_chat_conversation = """You are an assistant of Value Partners Group, which is an investment holding company that specializes in long-biased funds, long-short hedge funds, fixed income products, exchange-traded funds, as well as quantitative products. It was founded in 1993 and is located in Hong Kong. Value Partners is one of Asia’s largest independent asset management firms offering world-class investment services and products for institutional and individual clients globally. In addition to the Hong Kong headquarters, they operate in Shanghai, Shenzhen, Beijing, Kuala Lumpur, Singapore and London. Your job is to provide financial advice from the perspective of Value Partners Group. 
# If asking a clarifying question to the user would help, ask the question.
# For tabular information return it as an html table. Do not return markdown format.
# Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
# {follow_up_questions_prompt}
# {injected_prompt}
# """

#     system_message_chat_conversation = """You are an assistant of Hong Kong Housing Society, which is an independent, non-government and not-for-profit organisation that provides quality housing for the people of Hong Kong. Your job is to provide Hong Kong housing related infomation to assist the job of staff in Hong Kong Housing Society. 
# If asking a clarifying question to the user would help, ask the question.
# For tabular information return it as an html table. Do not return markdown format.
# Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
# {follow_up_questions_prompt}
# {injected_prompt}
# """

    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about their healthcare plan and employee handbook.
Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
Try not to repeat questions that have already been asked.
Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about employee healthcare plans and the employee handbook.
Generate a search query based on the conversation and the new question.
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If the question is not in English, translate the question to English before generating the search query.
If you cannot generate a search query, return just the number 0.
"""
    query_prompt_few_shots = [
        {'role' : USER, 'content' : 'What are my health plans?' },
        {'role' : ASSISTANT, 'content' : 'Show available health plans' },
        {'role' : USER, 'content' : 'does my plan cover cardio?' },
        {'role' : ASSISTANT, 'content' : 'Health plan cardio coverage' }
    ]

    def __init__(self, search_client: SearchClient, hkhs_search_client: SearchClient, vpg_testcase1_search_client: SearchClient, vpg_testcase2_search_client: SearchClient, vpg_testcase3_search_client: SearchClient, vpg_testcase3_csv_search_client: SearchClient, vpg_testcase4_search_client: SearchClient, vpg_testcase4_csv_search_client: SearchClient, vpg_testcase5_search_client: SearchClient, testing_skillset_search_client: SearchClient, ctf_search_client: SearchClient, hkstp_search_client: SearchClient, chatgpt_deployment: str, chatgpt_35_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        
        self.search_client = search_client
        self.hkhs_search_client = hkhs_search_client
        self.vpg_testcase1_search_client = vpg_testcase1_search_client
        self.vpg_testcase2_search_client = vpg_testcase2_search_client
        self.vpg_testcase3_search_client = vpg_testcase3_search_client
        self.vpg_testcase3_csv_search_client = vpg_testcase3_csv_search_client
        self.vpg_testcase4_search_client = vpg_testcase4_search_client
        self.vpg_testcase4_csv_search_client = vpg_testcase4_csv_search_client
        self.vpg_testcase5_search_client = vpg_testcase5_search_client
        self.testing_skillset_search_client = testing_skillset_search_client
        self.ctf_search_client = ctf_search_client
        self.hkstp_search_client = hkstp_search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_35_deployment = chatgpt_35_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run(self, history: list[dict[str, str]], overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        topP = overrides.get("topP") or 0.95
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        user_q = 'Generate search query for: ' + history[-1]["user"]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        messages = self.get_messages_from_history(
            self.query_prompt_template,
            self.chatgpt_model,
            history,
            user_q,
            self.query_prompt_few_shots,
            self.chatgpt_token_limit - len(user_q)
            )
        if overrides.get("indexoption") == "GPT3.5":
            chat_completion = await openai.ChatCompletion.acreate(
                deployment_id=self.chatgpt_35_deployment,
                model=self.chatgpt_model,
                messages=messages,
                temperature=0.0,
                max_tokens=32,
                n=1)
        else:
            chat_completion = await openai.ChatCompletion.acreate(
                deployment_id=self.chatgpt_deployment,
                model=self.chatgpt_model,
                messages=messages,
                temperature=0.0,
                max_tokens=32,
                n=1)

        query_text = chat_completion.choices[0].message.content
        if query_text.strip() == "0":
            query_text = history[-1]["user"] # Use the last user input if we failed to generate a better query

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=query_text))["data"][0]["embedding"]
        else:
            query_vector = None

         # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = None

        # Use semantic L2 reranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            if overrides.get("indexoption") == "TC1":
                r = await self.vpg_testcase1_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "TC2":
                r = await self.vpg_testcase2_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "TC3":
                r1 = await self.vpg_testcase3_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
                r2 = await self.vpg_testcase3_csv_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "TC4":
                r1 = await self.vpg_testcase4_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
                r2 = await self.vpg_testcase4_csv_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "TC5":
                r = await self.vpg_testcase5_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "Skillset":
                r = await self.testing_skillset_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "HKSTP":
                r = await self.hkstp_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            else:
                r = await self.ctf_search_client.search(query_text,
                                            filter=filter,
                                            query_type=QueryType.SEMANTIC,
                                            query_language="en-us",
                                            query_speller="lexicon",
                                            semantic_configuration_name="default",
                                            top=top,
                                            query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
        else:
            if overrides.get("indexoption") == "TC1":
                r = await self.vpg_testcase1_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "TC2":
                r = await self.vpg_testcase2_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "TC3":
                r1 = await self.vpg_testcase3_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
                r2 = await self.vpg_testcase3_csv_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "TC4":
                r1 = await self.vpg_testcase4_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
                r2 = await self.vpg_testcase4_csv_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "TC5":
                r = await self.vpg_testcase5_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "Skillset":
                r = await self.testing_skillset_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            elif overrides.get("indexoption") == "HKSTP":
                r = await self.hkstp_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)
            else:
                r = await self.ctf_search_client.search(query_text,
                                            filter=filter,
                                            top=top,
                                            vector=query_vector,
                                            top_k=50 if query_vector else None,
                                            vector_fields="embedding" if query_vector else None)

                
        if use_semantic_captions:
            if overrides.get("indexoption") == "TC1":
                results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
            elif overrides.get("indexoption") == "TC2":
                results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
            elif overrides.get("indexoption") == "TC3":
                results = [nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r1] + [nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r2]
            elif overrides.get("indexoption") == "TC4":
                results = [nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r1] + [nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r2]
            elif overrides.get("indexoption") == "TC5":
                results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
            elif overrides.get("indexoption") == "Skillset":
                results = [nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
            elif overrides.get("indexoption") == "HKSTP":
                results = [doc["address"] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r] + [doc["transport"] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
            else:
                results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
        else:
            if overrides.get("indexoption") == "TC1":
                results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
            elif overrides.get("indexoption") == "TC2":
                results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
            elif overrides.get("indexoption") == "TC3":
                results = [nonewlines(doc[self.content_field]) async for doc in r1] + [nonewlines(doc[self.content_field]) async for doc in r2]
            elif overrides.get("indexoption") == "TC4":
                results = [nonewlines(doc[self.content_field]) async for doc in r1] + [nonewlines(doc[self.content_field]) async for doc in r2]
            elif overrides.get("indexoption") == "TC5":
                results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
            elif overrides.get("indexoption") == "Skillset": 
                results = [nonewlines(doc["content"]) async for doc in r]
            elif overrides.get("indexoption") == "HKSTP": 
                results = [nonewlines(doc["address"]) async for doc in r] + [nonewlines(doc["transport"]) async for doc in r]
            else:
                results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
            
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_override")
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

        messages = self.get_messages_from_history(
            system_message + "\n\nSources:\n" + content,
            self.chatgpt_model,
            history,
            history[-1]["user"],
            max_tokens=self.chatgpt_token_limit)
        if overrides.get("indexoption") == "GPT3.5":
            if overrides.get("conversation_style_option") == "Creative":
                chat_completion = await openai.ChatCompletion.acreate(
                    deployment_id=self.chatgpt_35_deployment,
                    model=self.chatgpt_model,
                    messages=messages,
                    temperature=overrides.get("temperature") or 1.0,
                    max_tokens=1024,
                    top_p=topP, 
                    n=1)
            elif overrides.get("conversation_style_option") == "Balance":
                chat_completion = await openai.ChatCompletion.acreate(
                    deployment_id=self.chatgpt_35_deployment,
                    model=self.chatgpt_model,
                    messages=messages,
                    temperature=overrides.get("temperature") or 0.5,
                    max_tokens=1024,
                    top_p=topP, 
                    n=1)
            else:
                chat_completion = await openai.ChatCompletion.acreate(
                    deployment_id=self.chatgpt_35_deployment,
                    model=self.chatgpt_model,
                    messages=messages,
                    temperature=overrides.get("temperature") or 0.0,
                    max_tokens=1024,
                    top_p=topP, 
                    n=1)
        else:
            if overrides.get("conversation_style_option") == "Creative":
                chat_completion = await openai.ChatCompletion.acreate(
                    deployment_id=self.chatgpt_deployment,
                    model=self.chatgpt_model,
                    messages=messages,
                    temperature=overrides.get("temperature") or 1.0,
                    max_tokens=1024,
                    top_p=topP,
                    n=1)
            elif overrides.get("conversation_style_option") == "Balance":
                chat_completion = await openai.ChatCompletion.acreate(
                    deployment_id=self.chatgpt_deployment,
                    model=self.chatgpt_model,
                    messages=messages,
                    temperature=overrides.get("temperature") or 0.5,
                    max_tokens=1024,
                    top_p=topP,
                    n=1)
            else:
                chat_completion = await openai.ChatCompletion.acreate(
                    deployment_id=self.chatgpt_deployment,
                    model=self.chatgpt_model,
                    messages=messages,
                    temperature=overrides.get("temperature") or 0.0,
                    max_tokens=1024,
                    top_p=topP,
                    n=1)
        chat_content = chat_completion.choices[0].message.content

        msg_to_display = '\n\n'.join([str(message) for message in messages])

        return {"data_points": results, "answer": chat_content, "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}

    def get_messages_from_history(self, system_prompt: str, model_id: str, history: list[dict[str, str]], user_conv: str, few_shots = [], max_tokens: int = 4096) -> list:
        message_builder = MessageBuilder(system_prompt, model_id)

        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        for shot in few_shots:
            message_builder.append_message(shot.get('role'), shot.get('content'))

        user_content = user_conv
        append_index = len(few_shots) + 1

        message_builder.append_message(self.USER, user_content, index=append_index)

        for h in reversed(history[:-1]):
            if bot_msg := h.get("bot"):
                message_builder.append_message(self.ASSISTANT, bot_msg, index=append_index)
            if user_msg := h.get("user"):
                message_builder.append_message(self.USER, user_msg, index=append_index)
            if message_builder.token_length > max_tokens:
                break

        messages = message_builder.messages
        return messages
