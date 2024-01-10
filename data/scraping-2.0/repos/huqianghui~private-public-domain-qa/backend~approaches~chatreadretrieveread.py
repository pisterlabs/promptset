import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines
import requests
import json
# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class ChatReadRetrieveReadApproach(Approach):
    prompt_prefix = """<|im_start|>系统助理帮助公司员工解决他们的问题，回答要简短。
仅回答以下来源列表中列出的事实。如果下面没有足够的信息，请说您不知道。不要生成不使用以下来源的答案。不要使用年代久远的来源信息。如果向用户提出澄清问题会有所帮助，请提出问题。
每个源都有一个名称，后跟冒号和实际信息，始终包括您在响应中使用的每个事实的源名称。使用方形制动器来引用源。
例如:
来源：
info1.txt: 内容 <https://XXX.blob.core.windows.net/content/info1.txt>
输出: 根据[info1.txt](https://XXX.blob.core.windows.net/content/info1.txt)，内容
不要合并来源，而是单独列出每个来源，例如 [info1.txt][info2.pdf].
不要使用引用，而是始终将来源路径放在()中，例如(https://XXX.blob.core.windows.net/content/info1.txt)(https://XXX.blob.core.windows.net/content/info2.pdf).
对于表格形式的数据，请以HTML表格形式输出，不要使用Markdown表格.
回答内容一定要包括来源，一定要要把来源路径放在()中，例如(https://XXX.blob.core.windows.net/content/info1.txt)(https://XXX.blob.core.windows.net/content/info2.pdf).

{follow_up_questions_prompt}
{injected_prompt}
来源:
{sources}
<|im_end|>
{chat_history}
"""

    follow_up_questions_prompt_content = """生成三个非常简短的后续问题。
    使用双尖括号来引用问题，例如<<面条是否可以用体外模拟进行GI测试？>>。
    尽量不要重复已经问过的问题。
    仅生成问题，不生成问题前后的任何文本，例如不要生成"下一个问题" """

    query_prompt_template = """以下是到目前为止的对话历史，以及用户提出的一个新问题，需要通过在知识库中搜索来回答。
    根据对话和新问题生成查询条件。
    请勿在搜索查询词中包含引用的源文件名和文档名称，例如信息.txt或文档.pdf。
    不要在搜索查询词的 [] 或<<>>中包含任何文本。

历史对话:
{chat_history}

问题:
{question}

查询条件:
"""

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str, sourcepage_path_field: str, subscription_key: str, bing_search_endpoint: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.bing_search_subscriptin_key = subscription_key
        self.bing_search_endpoint = bing_search_endpoint
        self.sourcepage_path_field = sourcepage_path_field

    def run(self, history: list[dict], overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 1
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None
        question = history[-1]["user"]
        
        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(chat_history=self.get_chat_history_as_text(history, include_last_turn=False), question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=500, 
            n=1, 
            stop=["\n"])
        q = completion.choices[0].text

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="en-US", 
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            # r = self.search_client.search(q, filter=filter, top=top)
            r = self.search_client.search(question, filter=filter, top=top)
            
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) + " <" + doc[self.sourcepage_path_field] + ">" for doc in r if doc['@search.score'] > 5]
        question=history[-1]["user"]
        
        if len(results) > 0: 
            search_result = []         
            cognitive_search_result = "\n".join(results)
            content = cognitive_search_result
            supporting_facts = results
        else:
            search_result = self.get_bing_search_result(question, top)
            bing_search_result = "\n".join(search_result)
            content = bing_search_result
            supporting_facts = bing_search_result
        
        # content = cognitive_search_result
        # Use Bing Search to provide more information besides knowledge base
        # search_result = self.get_bing_search_result(question, top)
        # bing_search_result = "\n".join(search_result)
        # cognitive_search_result = "\n".join(results)
        # content = cognitive_search_result + "\n" + bing_search_result           
        
  
        # supporting_facts = results + search_result
        
        # STEP 3: Generate a response with the retrieved documents as prompt
        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            prompt = prompt_override.format(sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)

        # STEP 4: Generate a contextual and content specific answer using the search results and chat history
        completion = openai.Completion.create(
            engine=self.chatgpt_deployment, 
            prompt=prompt, 
            temperature=overrides.get("temperature") or 0.0, 
            max_tokens=2000, 
            n=1, 
            stop=["<|im_end|>", "<|im_start|>"])

        wrap_upped_answer = completion.choices[0].text
        return {"data_points": supporting_facts, "answer": wrap_upped_answer, "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
    
    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text
    
    def get_bing_search_result(self, question, top):
        mkt = 'zh-CN'
        params = { 'q': question, 'mkt': mkt , 'answerCount': top}
        headers = { 'Ocp-Apim-Subscription-Key': self.bing_search_subscriptin_key }
        r = requests.get(self.bing_search_endpoint, headers=headers, params=params)
        json_response = json.loads(r.text)
        # print(json_response)
        result = [page['name'] + ": " + nonewlines(page['snippet']) + " <" + page['url'] + ">" for page in list(json_response['webPages']['value'])[:top]]
        return result
