import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from approaches.index.gptindex import GPTKGIndexer
from text import nonewlines
import requests
import json
# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class ChatReadRetrieveReadApproach(Approach):
    prompt_prefix = """
    你是一名系统助理，你帮助公司员工解决他们的问题，回答要简短。仅回答以下来源列表中列出的事实。如果下面没有足够的信息，请说您不知道。不要生成不使用以下来源的答案。不要使用年代久远的来源信息。如果向用户提出澄清问题会有所帮助，请提出问题。
    来源是一个数组，数组中的每个源都是一个JSON对象, 包含sourcepage、content和sourcepage_path。始终包括您在响应中使用的每个事实的sourcepage和sourcepage_path。使用[]()来引用来源。
    For tabular information return it as an html table. Do not return markdown format.

    来源：
    EXAMPLES:
    来源:
    [{{
    "sourcepage":"info1.txt",
    "content": "面条可以用体外模拟进行GI测试。",
    "sourcepage_path": "http://www.somedomain1.com/info1.txt"
    }}]

    问题: 
    面条是否可以用体外模拟进行GI测试？
    回答: 根据[info1.txt](http://www.somedomain2.com/info1.txt)，面条可以用体外模拟进行GI测试。

    来源:
    [
    {{
    "sourcepage":"info1.txt",
    "content": "内容1",
    "sourcepage_path": "http://www.somedomain1.com/info1.txt"
    }},
    {{
    "sourcepage":"info2.pdf",
    "content": "内容2",
    "sourcepage_path": "http://www.somedomain2.com/info2.pdf"
    }}
    ]
    问题: 
    内容1和内容2是什么？
    回答：根据[info1.txt](http://www.somedomain2.com/info1.txt)和[info2.pdf](http://www.somedomain2.com/info2.pdf)，内容1和内容2

    来源:
    []
    问题：
    内容1是什么？
    回答：对不起，我无法找到合适的信息回答这个问题。

    Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
    Don't use reference, ALWAYS keep source page pth in (), e.g. (http://www.somedomain1.com/info1.txt)(http://www.somedomain2.com/info2.pdf).
    
    {follow_up_questions_prompt}
    {injected_prompt}

    """

    user_question_prompt = """
    来源:
    {sources}
    问题:
    {question}
    回答：
    """
    
    follow_up_questions_prompt_content = """生成三个非常简短的后续问题。
    使用双尖括号来引用问题，例如<<面条是否可以用体外模拟进行GI测试?>>。
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

    index_router_prompt = """
    Based on the question, determine what kind of question is user asking.
    If the question is related in compare/contrast information, return 'graph'.
    If the question is related in asking fact or relationship about something or some things, return 'graph'.
    If the question is related in asking other information about something, return 'text'
    If the question is complicated or contain multiple intentions, return 'all'.
    Output MUST be in ['graph','text','all']
    Here is the user question:
    {question}
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
        # self.kg_search = GPTKGIndexer()

    def run(self, history: list[dict], overrides: dict) -> any:
        top = overrides.get("top") or 3
        question = history[-1]["user"]
        useBingSearch = overrides.get("use_bing_search") or False
        # STEP 0: Use OpenAI to determine index
        search_processor = {
            # "graph": self.knowledge_graph_search,
            "text" : self.text_search,
            "all"  : self.combine_search
        }
        method = self.determine_search_method(question)
        print(f"answering '{question}' using {method} index")
        # response = search_processor[method](question,history, overrides)
        response = self.combine_search(question, history, overrides)

        print(response)
        if bool(useBingSearch) == False:          
            supporting_facts = response
            print("supporting_facts from cognitive search: " + str(supporting_facts))
        else:
            search_result = self.get_bing_search_result(question, top)
            bing_search_result = "\n".join(search_result)
            response = bing_search_result
            supporting_facts = search_result
            print("supporting_facts from bing search: " + str(supporting_facts))
        
        # STEP 3: Generate a response with the retrieved documents as prompt
        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            prompt = prompt_override.format(chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)

        # STEP 4: Generate a contextual and content specific answer using the search results and chat history
        system_message = [{
            "role": "system",
            "content": f"{prompt}"
        }]
        chat_histroy_message = self.get_chat_history_as_text(history, include_last_turn=False)
        user_question_with_sources = self.user_question_prompt.format(sources = response, question=question)
        user_question_message = {
            "role": "user",
            "content": user_question_with_sources
        }
        chat_message=[]
        if len(chat_histroy_message) > 0:
            chat_message = system_message + chat_histroy_message
            chat_message.append(user_question_message)
        else:
            chat_message.append(user_question_message)
        
        completion = openai.ChatCompletion.create(
            engine=self.chatgpt_deployment,
            messages=chat_message,
            temperature=0.0
        )
        wrap_upped_answer = completion['choices'][0]['message']['content']
        print(wrap_upped_answer)
        return {"data_points": supporting_facts, "answer": wrap_upped_answer, "thoughts": f"{prompt}" + user_question_with_sources.replace('\n', '<br>')}
    
    def determine_search_method(self, question):
        message = [{
            "role": "user",
            "content": self.index_router_prompt.format(question=question)
        }]
        completion = openai.ChatCompletion.create(
            engine=self.chatgpt_deployment,
            messages=message,
            temperature=0.0
        )
        search_method = completion['choices'][0]['message']['content']
        if search_method not in ['text','graph','all']:
            search_method = 'text'
        return search_method
    
    # def knowledge_graph_search(self, question, history: list[dict], overrides: dict):
    #     result = self..query(question)
    #     json_result = {"sourcepage": result.get_formatted_sources(), "content": result.response, "sourcepage_path":""}
    #     print(json_result)
    #     return json.dumps(json_result, ensure_ascii=False)

    def text_search(self, question,history: list[dict], overrides: dict):
        # use_semantic_captions = True if overrides.get("semantic_captions") else False
        use_semantic_captions = False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None
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

        print("Key word search query: " + q)
        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="zh-CN", 
                                        #   query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)

        for doc in r:
            print(doc)

        if use_semantic_captions:
            json_results = [{"sourcepage": doc[self.sourcepage_field], "content": nonewlines(" . ".join([c.text for c in doc['@search.captions']])), "sourcepage_path": doc[self.sourcepage_path_field]} for doc in r]     
        else:
            json_results = [{"sourcepage": doc[self.sourcepage_field], "content": nonewlines(doc[self.content_field]), "sourcepage_path": doc[self.sourcepage_path_field]} for doc in r if doc['@search.score']]

        json_results = json.dumps(json_results, ensure_ascii=False)

        print("Search result: "+json_results);
        return json_results

    def combine_search(self, question, history: list[dict], overrides: dict):
        # kg_response = self.knowledge_graph_search(question, history, overrides)
        text_response = self.text_search(question, history, overrides)
        # return kg_response+text_response
        return text_response

    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000):
        history_text = ""
        history_message = []
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            history_message.append({
                "role": "user",
                "content": f"{h['user']}"
            })
            history_message.append({
                "role": "assistant",
                "content": f"{h.get('bot')}"
            })
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_message
    
    def get_bing_search_result(self, question, top):
        mkt = 'zh-CN'
        params = { 'q': question, 'mkt': mkt , 'answerCount': top}
        headers = { 'Ocp-Apim-Subscription-Key': self.bing_search_subscriptin_key }
        r = requests.get(self.bing_search_endpoint, headers=headers, params=params)
        json_response = json.loads(r.text)
        print(json_response)
        result = [page['name'] + ": " + nonewlines(page['snippet']) + " <" + page['url'] + ">" for page in list(json_response['webPages']['value'])[:top]]
        return result