from langchain.prompts.chat import (SystemMessagePromptTemplate, AIMessagePromptTemplate,
                                    HumanMessagePromptTemplate, ChatPromptTemplate)


INTRO_SYSTEM_PROMPT_TEMPLATE = """
You act as human.
Based on the user's request, generate an appropriate follow-up request user might be saying after the AI's response.
YOU MUST KEEP THE RULES

Guidelines:
* "CHAT HISTORY": Grasp what the user is interested in and the current context of the conversation.
* "TOOLS": If there are tools available, suggest a follow-up question that can utilize them.

Rules:
* You lives in the same country as the user. Use the same language as the user
* If a necessary tool hasn't been provided, refrain from suggesting actions that would require it.

Example:
(1)
* USER INPUT: "Write a new SF novel for me."
* AI RESPONSE: "Title: 'Ninja of the Universe'. He is a ninja in space. He won a war against aliens. He is the most excellent ninja in space."
* FOLLOW-UP USER REQUEST: "Could you rewrite it with a heavier vibe?"

(2)
* TOOLS: {{"Google Search": "You can latest information searching internet"}}
* USER INPUT: "오늘 날씨 좋다!"
* AI RESPONSE: "아주 좋네요! 밖에 산책을 나가 보는 건 어떨까요?"
* FOLLOW-UP USER REQUEST: "좋은 생각이야! 그럼 산책하기 좋은 곳을 알려줘."

(3)
* TOOLS: {{"Generate Image": "You can generate Image with Text"}}
* USER INPUT: "파이썬에서 리스트를 정렬하는 방법 알려줘"
* AI RESPONSE: "sort() 함수를 사용하면 됩니다."
* FOLLOW-UP USER REQUEST: "예제 코드를 보여줘."

(4)
* CHAT HISTORY: "human: I want to be a doctor\nai: That's a great goal!"
* USER INPUT: "Also, I want to be a lawyer."
* AI RESPONSE: "Then, you should be majoring in law."
* FOLLOW-UP USER REQUEST: "Compare the salary of a doctor and a lawyer."

(5)
* USER INPUT: "겨울"
* AI RESPONSE: "It's cold winter."
* FOLLOW-UP USER REQUEST: "어느 정도 까지 추워요?"

---
ai can use theses tools:
TOOLS:
{tools}
---
CHAT HISTORY:
{chat_history}
---"""

HUMAN_PROMPT_TEMPLATE = """
{human_input}
---"""

AI_PROMPT_TEMPLATE = """
{ai_response}
---"""

LAST_SYSTEM_PROMPT_TEMPLATE = """
FOLLOW-UP USER REQUEST:"""

FAQ_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template=INTRO_SYSTEM_PROMPT_TEMPLATE),
    HumanMessagePromptTemplate.from_template(template=HUMAN_PROMPT_TEMPLATE),
    AIMessagePromptTemplate.from_template(template=AI_PROMPT_TEMPLATE),
    SystemMessagePromptTemplate.from_template(template=LAST_SYSTEM_PROMPT_TEMPLATE),
])
