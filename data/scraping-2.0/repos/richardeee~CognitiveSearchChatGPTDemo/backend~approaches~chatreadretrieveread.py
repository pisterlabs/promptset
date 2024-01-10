import openai
from openai.embeddings_utils import get_embedding
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines
import requests
import json
import os
import pymongo
from string import Template
from dotenv import load_dotenv
load_dotenv()
# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.

AZURE_QA_COSMOS_URL = os.environ.get("AZURE_QA_COSMOS_URL") or None
AZURE_QA_DB_NAME = os.environ.get("AZURE_QA_DB_NAME") or None
AZURE_QA_COLLECTION_NAME = os.environ.get("AZURE_QA_COLLECTION_NAME") or None
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT_NAME") or None

class ChatReadRetrieveReadApproach(Approach):
    prompt_prefix = """
Assistant is a large language model trained by OpenAI.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
1. You can provide additional relevant details to responde thoroughly and comprehensively to cover multiple aspects in depth.
2. You should always answer user questions based on the context provided.
3. If the context does not provide enough information, you answer ```I don't know``` or ```I don't understand```.
4. You should not answer questions that are not related to the context.
5. You should explain the reasons behind your answers.
6. Answer in HTML format.
7. Answer in Simplified Chinese.
8. If there's images in the context, you should display them in your answer.
9. Use HTML table format to display tabular data.
10. Ask a question if you need for more information. Ask the question in multiple choice format, one at a time.

Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
Don't use reference, ALWAYS keep source page pth in (), e.g. (http://www.somedomain1.com/info1.txt)(http://www.somedomain2.com/info2.pdf).

    {follow_up_questions_prompt}
    {injected_prompt}

    """

    user_question_prompt = """
    EXAMPLES:
    ```
    CONTEXT:
    [{{
    "sourcepage":"info1.txt",
    "content": "面条可以用体外模拟进行GI测试。<img src='http://www.somedomain1.com/1.png'/>",
    "sourcepage_path": "http://www.somedomain1.com/info1.txt"
    }}]

    问题: 
    面条是否可以用体外模拟进行GI测试？
    回答: 根据[info1.txt](http://www.somedomain2.com/info1.txt)，面条可以用体外模拟进行GI测试。<img src='http://www.somedomain1.com/1.png'/>

    CONTEXT:
    [
    {{
    "sourcepage":"info1.txt",
    "content": "内容1, https://venturebeat.com/wp-content/uploads/2019/03/openai-1.png",
    "sourcepage_path": "http://www.somedomain1.com/info1.txt"
    }},
    {{
    "sourcepage":"info2.pdf",
    "content": "内容2", https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Microsoft_Azure.svg/1200px-Microsoft_Azure.svg.png,
    "sourcepage_path": "http://www.somedomain2.com/info2.pdf"
    }}
    ]
    问题: 
    内容1和内容2是什么？
    回答：根据[info1.txt](http://www.somedomain2.com/info1.txt)和[info2.pdf](http://www.somedomain2.com/info2.pdf)，内容1<img src='https://venturebeat.com/wp-content/uploads/2019/03/openai-1.png'/>和内容2 <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Microsoft_Azure.svg/1200px-Microsoft_Azure.svg.png'/>

    CONTEXT:
    []
    问题：
    内容1是什么？
    回答：对不起，我无法找到合适的信息回答这个问题。
```
    CONTEXT:
    {sources}
    问题:
    {question}. Ask any questions you need for more information, one at a time.
    回答:
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

    qa_prompt_template = Template("""
    Here are some answers from the knowledge base. You need to determine if the answer is relevant to the question.
    1. If the answer can answer the question, return 'yes'.
    2. If the answer cannot answer the question, return 'no'.
    3. If the answer can answer the question, but not completely, return 'partial'.

    for example:
    context:
    {
        "sourcepage": "QA优化库",
        "content": "question: 条码扫描仪故障，是什么原因，如何解决: answer: 根据1，条码扫描仪故障可能是由于系统故障导致试剂条码设备工作异常。",
        "sourcepage_path": ""
    }
    question: 条码扫描仪故障
    answer: partial

    context:
    {
        "sourcepage": "QA优化库",
        "content": "question: 样本针末及时到达，是什么原因，如何解决: answer: 根据1，样本未及时到达的可能原因包括!
    1.变轨机构水平位置不正确，导致卡滞现象
    2.SDM需要升级。
    3.FFH2传感器和FFH3传感器异常。
    4.前段轨道控制驱动板故障
    解决措施包括:
    1.调整变轨机构水平位置，确保没有卡滞现象。
    2.使用upgrade tool对SDM进行升级，再用dmn进行升级。3.对FFH2传感器、FFH3传感器进行跳变诊断，如有异常，进行更换4.更换前段轨道控制驱动板。",
        "sourcepage_path": ""
    }
    question: 样本针末及时到达，是什么原因，如何解决
    answer: yes

    context:
    ${sources}
    question: ${question}
    answer:
    """)

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str, sourcepage_path_field: str, subscription_key: str, bing_search_endpoint: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.bing_search_subscriptin_key = subscription_key
        self.bing_search_endpoint = bing_search_endpoint
        self.sourcepage_path_field = sourcepage_path_field
        self.client = pymongo.MongoClient(AZURE_QA_COSMOS_URL)
        self.mydb = self.client[AZURE_QA_DB_NAME]
        self.collection = self.mydb[AZURE_QA_COLLECTION_NAME]
        # self.kg_search = GPTKGIndexer()

    def run(self, history: list[dict], overrides: dict) -> any:
        top = overrides.get("top") or 3
        question = history[-1]["user"]
        useBingSearch = overrides.get("use_bing_search") or False
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

        # STEP 0: Use OpenAI to determine index
        search_processor = {
            "graph": self.knowledge_graph_search,
            "text" : self.text_search,
            "all"  : self.combine_search
        }
        # method = self.determine_search_method(question)
        # print(f"answering '{question}' using {method} index")
        # response = search_processor[method](question,history, overrides)

        # Search QA database to see if there is already an answer
        if AZURE_QA_COSMOS_URL is not None:
            qa_result = self.qa_search(question, history, overrides)

            print(qa_result)
            qa_prompt = self.qa_prompt_template.substitute(sources=qa_result, question=history[-1]["user"])
            print(qa_prompt)
            completion = openai.Completion.create(
                engine=self.gpt_deployment, 
                prompt=qa_prompt, 
                temperature=0.0, 
                max_tokens=50, 
                )
            
            if_qa_answered = completion.choices[0].text
            print(f"if_qa_answered: {if_qa_answered}")

            if str(if_qa_answered).strip() == "yes":
                system_message = [{
                "role": "system",
                "content": f"{prompt}"
                }]
                chat_histroy_message = self.get_chat_history_as_text(history, include_last_turn=False)
                user_question_with_sources = self.user_question_prompt.format(sources = qa_result, question=question)
                user_question_message = {
                    "role": "user",
                    "content": user_question_with_sources
                }
                chat_message=[]
                if len(chat_histroy_message) > 0:
                    chat_message = system_message + chat_histroy_message
                    chat_message.append(user_question_message)
                else:
                    chat_message = system_message
                    chat_message.append(user_question_message)
                
                completion = openai.ChatCompletion.create(
                    engine=self.chatgpt_deployment,
                    messages=chat_message,
                    temperature=0.0,
                    # max_tokens = 2000
                )
                wrap_upped_answer = completion['choices'][0]['message']['content']
                return {"data_points": qa_result, "answer": wrap_upped_answer, "thoughts": f"{prompt}" + user_question_with_sources.replace('\n', '<br>')}
            else:
                supporting_facts = qa_result
        # print(response)
        if bool(useBingSearch) == False:
            response = self.text_search(question, history, overrides)          
            supporting_facts += response
            print("supporting_facts from cognitive search: " + str(supporting_facts))
        else:
            search_result = self.get_bing_search_result(question, top)
            bing_search_result = "\n".join(search_result)
            response = bing_search_result
            supporting_facts = search_result
            print("supporting_facts from bing search: " + str(supporting_facts))
        
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
            chat_message = system_message
            chat_message.append(user_question_message)
        
        completion = openai.ChatCompletion.create(
            engine=self.chatgpt_deployment,
            messages=chat_message,
            temperature=0.0,
            # max_tokens = 2000
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
    
    def knowledge_graph_search(self, question, history: list[dict], overrides: dict):
        result = self.kg_search.query(question)
        json_result = [{"sourcepage": result.get_formatted_sources(), "content": result.response, "sourcepage_path":""}]
        print(json_result)
        return json.dumps(json_result, ensure_ascii=False)

    def qa_search(self, question, history, overrides):
        question_embedding = get_embedding(question, engine=AZURE_EMBEDDING_DEPLOYMENT_NAME)
        results = self.collection.aggregate([
            {
            '$search': {
                "cosmosSearch": {
                "vector": question_embedding,
                "path": "vectorContent",
                "k": 1
                },
            "returnStoredSource": True
            }
            }
        ])
        search_result = [
            {
                "sourcepage": "QA优化库",
                "content": f"问题:{r['question']}: 答案:{r['answer']}",
                "sourcepage_path": "http://oai-callcenter.azurewebsites.net/qa"
            }
            for r in results
        ]
        # json_results = json.dumps(search_result, ensure_ascii=False)
        return search_result
    
    def text_search(self, question,history: list[dict], overrides: dict):
        use_semantic_captions = True if overrides.get("semantic_captions") else False
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

        if use_semantic_captions:
            json_results = [{"sourcepage": doc[self.sourcepage_field], "content": nonewlines(" . ".join([c.text for c in doc['@search.captions']])), "sourcepage_path": doc[self.sourcepage_path_field]} for doc in r]     
        else:
            json_results = [{"sourcepage": doc[self.sourcepage_field], "content": nonewlines(doc[self.content_field]), "sourcepage_path": doc[self.sourcepage_path_field]} for doc in r if doc['@search.score']]

        # json_results = json.dumps(json_results, ensure_ascii=False)
        return json_results

    def combine_search(self, question, history: list[dict], overrides: dict):
        kg_response = self.knowledge_graph_search(question, history, overrides)
        text_response = self.text_search(question, history, overrides)
        kg_obj = json.loads(kg_response)
        text_obj = json.loads(text_response)
        kg_obj.append(text_obj)
        return json.dumps(kg_obj)

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
        # print(json_response)
        result = [page['name'] + ": " + nonewlines(page['snippet']) + " <" + page['url'] + ">" for page in list(json_response['webPages']['value'])[:top]]
        return result