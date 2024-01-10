from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
from core.evaluation import calculate_relevance_score

from approaches.approach import AskApproach
from core.messagebuilder import MessageBuilder
from text import nonewlines


class RetrieveThenReadApproach(AskApproach):
    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = \
"You are an intelligent assistant helping employees with their questions regarding their different pay calculations as per the mentioned rules " + \
"Use 'you' to refer to the individual asking the questions even if they ask with 'I'. " + \
"Answer the following question using only the data provided in the sources below. " + \
"It's important to generate response in HTML table format when asked for tabular information. Do not use markdown format. "  + \
"Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. " + \
"If you cannot answer using the sources below, say you don't know. Use below examples to answer"

    #shots/sample conversation
    question = """
'What is the deductible for the employee plan for a visit to Overlake in Bellevue?'

Sources:
info1.txt: deductibles depend on whether you are in-network or out-of-network. In-network deductibles are $500 for employee and $1000 for family. Out-of-network deductibles are $1000 for employee and $2000 for family.
info2.pdf: Overlake is in-network for the employee plan.
info3.pdf: Overlake is the name of the area that includes a park and ride near Bellevue.
info4.pdf: In-network institutions include Overlake, Swedish and others in the region
"""
    answer = "In-network deductibles are $500 for employee and $1000 for family [info1.txt] and Overlake is in-network for the employee plan [info2.pdf][info4.pdf]."

    question1 = """
'compare Sunday pay penalties for different type of employees. Give the answer in table format.'
 
Sources:
info1.txt: deductibles depend on whether you are in-network or out-of-network. In-network deductibles are $500 for employee and $1000 for family. Out-of-network deductibles are $1000 for employee and $2000 for family.
info2.pdf: Overlake is in-network for the employee plan.
info3.pdf: Overlake is the name of the area that includes a park and ride near Bellevue.
info4.pdf: In-network institutions include Overlake, Swedish and others in the region
"""
    answer1 = "<table>\n  <tr>\n    <th>Employee Type</th>\n    <th>Sunday Pay Penalty</th>\n  </tr>\n  <tr>\n    <td>Part-time employees who work less than 40 hours in a workweek</td>\n    <td>Time and one quarter (1-1/4x) the employee's regular straight-time rate of pay [info1.txt]</td>\n  </tr>\n  <tr>\n    <td>Courtesy Clerks</td>\n    <td>Regular rate of pay plus fifty cents (50¢) per hour [info2.pdf]</td>\n  </tr>\n  <tr>\n    <td>Employees hired on or after March 27, 2005</td>\n    <td>Not eligible for Sunday Premium [info3.pdf]</td>\n  </tr>\n</table>\n\nNote: The premium rate for work performed on Sunday as such shall be time and one quarter (1-1/4x) the employee's regular straight-time rate of pay (exclusive of Courtesy Clerks). The Sunday premium, for hours worked up to eight (8), shall in no instance be offset against any weekly overtime which may be due under subparagraphs b and d of Section 26 because of the fact that the employee worked over forty (40) hours or thirty-two (32) hours in the particular workweek. The Sunday premium shall not be averaged into the employee's straight-time rate for the purpose of determining the rate upon which daily or weekly overtime is based in any workweek under Section 26 hereof..[info4.pdf]"

    def __init__(self, search_client: SearchClient, openai_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    async def run(self, q: str, overrides: dict[str, Any]) -> Any:
        print("-----Ask Step 3---------------")
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = None
        if exclude_category:
            filter = " and ".join(f"category ne '{category}'" for category in exclude_category)
            filter = f"({filter})"
        print("-----Ask Step 4---------------")
        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=q))["data"][0]["embedding"]
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        query_text = q if has_text else ""
        print(query_text)
        print("-----Ask Step 5---------------")
        # Use semantic ranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            print("-----Ask Step 6.1---------------")
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language="en-us",
                                          query_speller="lexicon",
                                          semantic_configuration_name="my-semantic-config",
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="contentVector" if query_vector else None)
        else:
            print("-----Ask Step 6.2---------------")
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          top=top,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="contentVector" if query_vector else None)

        #print(str(r))
        print("-----Ask Step 7---------------")
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
        content = "\n".join(results)
        # print(content)
        print("-----Ask Step 8---------------")
        message_builder = MessageBuilder(overrides.get("prompt_template") or self.system_chat_template, self.chatgpt_model)

        # add user question
        #user_content = q + "\n" + f"Sources:\n {content}"
        message_builder.append_message('user', q)
        message_builder.append_message('user', content)
        message_builder.append_message('assistant', self.answer1)
        message_builder.append_message('user', self.question1)
        # Add shots/samples. This helps model to mimic response and make sure they match rules laid out in system message.
        message_builder.append_message('assistant', self.answer)
        message_builder.append_message('user', self.question)

        messages = message_builder.messages
        # print(messages)
        print("-----Ask Step 9---------------")
        
        try :
            chat_completion = await openai.ChatCompletion.acreate(
                deployment_id=self.openai_deployment,
                model=self.chatgpt_model,
                messages=messages,
                temperature=overrides.get("temperature") or 0.1,
                max_tokens=1024,
                n=3)
            print("-----Ask Step 10---------------")
            response = chat_completion.choices[0].message.content
            # Token count for each request
            token_count = chat_completion.usage.total_tokens
            print("-----Ask Step 11---------------")

            score = calculate_relevance_score(content,q,response)

            if score > 3:
                answer = response #['output']
            else :
                print ("-"*10, "Context : ",content)
                message_builder = MessageBuilder(overrides.get("prompt_template") or self.system_chat_template, "gpt-35-turbo")
                #user_content = q + "\n" + f"Sources:\n {content}"
                message_builder.append_message('user', q)
                message_builder.append_message('user', content)
                messages = message_builder.messages
                chat_completion = await openai.ChatCompletion.acreate(
                    deployment_id=self.openai_deployment,
                    model="gpt-35-turbo",
                    messages=messages,
                    temperature=overrides.get("temperature") or 0.1,
                    max_tokens=1024,
                    n=1)
                answer = chat_completion.choices[0].message.content
                score = calculate_relevance_score(content,q,answer)


        except openai.error.InvalidRequestError as e:
            if e.error.code == "content_filter": # and e.error.innererror:
                #content_filter_result = e.error.innererror.content_filter_result
                # print the formatted JSON
                print("*"*10,"content_filter_result")
                answer = "Triggering content please modify and retry "
                score =0
                print("-----Ask Step 12---------------")

        return {"data_points": results, "answer": answer, "thoughts": f"Question:<br>{query_text}<br><br>Prompt:<br>" + '\n\n'.join([str(message) for message in messages]),"token_usage":token_count,"relevance_score":score}
