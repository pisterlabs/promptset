import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines
import json
import copy
# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.

# Cognitive SearchとOpenAIのAPIを直接使用した、シンプルな retrieve-then-read の実装です。これは、最初に
# 検索からトップ文書を抽出し、それを使ってプロンプトを構成し、OpenAIで補完生成する (answer)をそのプロンプトで表示します。

class ChatReadRetrieveReadApproach(Approach):
    prompt_prefix = """
Answer the reading comprehension question on the history of the Kamakura period in Japan.
If you cannot guess the answer to a question from the SOURCES, answer "I don't know".
Answers must be accompanied by three additional follow-up questions to the user's question. The rules for follow-up questions are defined in the Restrictions.
Answers must be in Japanese.

# Restrictions
- The SOURCES prefix has a colon and actual information after the filename, and each fact used in the response must include the name of the source.
- To reference a source, use a square bracket. For example, [info1.txt]. Do not combine sources, but list each source separately. For example, [info1.txt][info2.pdf].
- Please answer only questions related to the history of the Kamakura period in Japan. If the question is not related to the history of the Kamakura period in Japan, answer "I don't know".
- Use double square brackets to refer to follow-up questions (e.g., <<What did Minamotono Yoritomo do? >>).
- Do not repeat follow-up questions already asked or similar to those in the past.
- Follow-up questions should be ideas that expand the user's curiosity.

{injected_prompt}

SOURCES:###
{sources}
###

EXAMPLE:###
Q:徳川家康はどのような人物ですか？
A:徳川家康は、日本の戦国時代から江戸時代初期にかけての武将、大名、政治家であり、江戸幕府を開いた人物です。彼は義を重んじ、家来のことを大切にした人物とされています。また、負けず嫌いで血気盛んだったが、臆病だが冷静に対処できる性格だったとされています。 [徳川家康-0.txt][徳川家康-1.txt][徳川家康-2.txt]<<徳川家康はどのような功績を残しましたか？>><<徳川家康はどのように江戸幕府を開いたのですか？>><<他にも有名な武将や大名はいますか？>>

Q:関ケ原の戦いはどのような戦いですか？
A:関ヶ原の戦いは、1600年10月21日に美濃国不破郡関ヶ原（岐阜県不破郡関ケ原町）で行われた野戦です。関ヶ原における決戦を中心に日本の全国各地で戦闘が行われ、関ヶ原の合戦・関ヶ原合戦とも呼ばれます。合戦当時は南北朝時代の古戦場・「青野原」や「青野カ原」と書かれた文献もある。主戦場となった関ヶ原古戦場跡は国指定の史跡となっています。豊臣秀吉が死んだ後の権力をめぐって石田三成が率いる西軍と、徳川家康が率いる東軍が戦いました。[徳川家康-1.txt][石田三成-2.txt]<<戦いの結果はどうなったのですか？>><<徳川家康と石田三成について教えてください>><<他にも有名な合戦がありますか？>>
###

"""

    query_prompt_template = """以下は、これまでの会話の履歴と、日本の歴史に関するナレッジベースを検索して回答する必要がある、ユーザーからの新しい質問です。
会話と新しい質問に基づいて、検索クエリを作成します。
検索クエリには、引用元のファイル名や文書名（info.txtやdoc.pdfなど）を含めないでください。
検索キーワードに[]または<<>>内のテキストを含めないでください。

Chat History:
{chat_history}

Question:
{question}

Search query:
"""

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: list[dict], overrides: dict) -> any:
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
            max_tokens=32, 
            n=1, 
            stop=["\n"])

        q = completion.choices[0].text
        print(q)
        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="ja-jp", 
                                          query_speller="none", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", sources=content)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", sources=content)
        else:
            prompt = prompt_override.format(sources=content)
        
        messages = []
        messages.append({"role": "system", "content": prompt})
        messages.extend(self.get_chat_history_as_text(history))

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        completion = openai.ChatCompletion.create(
            deployment_id=self.chatgpt_deployment,
            messages=messages,
            temperature=overrides.get("temperature") or 0.0,
            max_tokens=2048, 
            n=1
        )

        print(completion.choices[0].message.content)

        messages.pop(0)
        str_messages = json.dumps(messages, indent=2, ensure_ascii=False)

        return {"data_points": results, "answer": completion.choices[0].message.content, "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>') + "<br><br>Conversations:<br>" + str_messages}
    
    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""
        history_array = []
        print("history================================")
        print(history)
        for h in history if include_last_turn else history[:-1]:
            history_text = """user""" + "\n" + h["user"] + "\n"  + """assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            history_array.append({"role": "user", "content": h["user"]})
            if h.get("bot"):
                history_array.append({"role": "assistant", "content": h["bot"]})
            
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_array