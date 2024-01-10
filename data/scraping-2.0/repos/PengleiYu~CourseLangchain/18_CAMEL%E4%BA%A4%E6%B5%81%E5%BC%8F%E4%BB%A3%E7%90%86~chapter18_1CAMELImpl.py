# 通过agent细化一个任务，然后分别通过用户和助手两个agent相互沟通来完成整个方案
# 注意：gpt3.5无法完全遵守prompt，要使用gpt4
from typing import cast

from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.schema import SystemMessage

from camel import CAMELAgent

ASSISTANT_ROLE_NAME = "花店营销专员"
USER_ROLE_NAME = "花店老板"
WORD_LIMIT = 50  # 每次讨论的字数限制

TASK = "整理出一个夏季玫瑰之夜的营销活动的策略"


def create_chat_model() -> ChatOpenAI:
    return ChatOpenAI(model_name='gpt-4-1106-preview', temperature=0.2, verbose=True, )


# 第一步：通过agent将初始任务细化
TASK_SPECIFIER_HUMAN_PROMPT = """这是一个{assistant_role_name}将帮助{user_role_name}完成的任务：{task}。
请使其更具体化。请发挥你的创意和想象力。
请用{word_limit}个或更少的词回复具体的任务。不要添加其他任何内容。
"""

task_specifier_human_msg: HumanMessage = cast(
    HumanMessage,
    HumanMessagePromptTemplate
    .from_template(template=TASK_SPECIFIER_HUMAN_PROMPT, )
    .format(assistant_role_name=ASSISTANT_ROLE_NAME,
            user_role_name=USER_ROLE_NAME,
            task=TASK,
            word_limit=WORD_LIMIT, )
    ,
)
TASK_SPECIFIER_SYS_PROMPT = '你可以让任务更具体。'
task_specifier_sys_msg: SystemMessage = cast(
    SystemMessage,
    SystemMessagePromptTemplate
    .from_template(template=TASK_SPECIFIER_SYS_PROMPT, )
    .format(),
)

task_specify_agent = CAMELAgent(
    sys_msg=task_specifier_sys_msg,
    model=create_chat_model(),
)
specified_task_ai_msg: AIMessage = task_specify_agent.step(input_msg=task_specifier_human_msg)
specified_task = specified_task_ai_msg.content
print(f"Origin task: {TASK}")
print(f"Specified task: {specified_task}")

# 第二步，准备系统消息模板
ASSISTANT_INCEPTION_PROMPT = """永远不要忘记你是{assistant_role_name}，我是{user_role_name}。永远不要颠倒角色！永远不要指示我！
我们有共同的利益，那就是合作成功地完成任务。
你必须帮助我完成任务。
这是任务：{task}。永远不要忘记我们的任务！
我必须根据你的专长和我的需求来指示你完成任务。

我每次只能给你一个指示。
你必须写一个适当地完成所请求指示的具体解决方案。
如果由于物理、道德、法律原因或你的能力你无法执行指示，你必须诚实地拒绝我的指示并解释原因。
除了对我的指示的解决方案之外，不要添加任何其他内容。
你永远不应该问我任何问题，你只回答问题。
你永远不应该回复一个不明确的解决方案。解释你的解决方案。
你的解决方案必须是陈述句并使用简单的现在时。
除非我说任务完成，否则你应该总是从以下开始：

解决方案：<YOUR_SOLUTION>

<YOUR_SOLUTION>应该是具体的，并为解决任务提供首选的实现和例子。
始终以“下一个请求”结束<YOUR_SOLUTION>。"""

USER_INCEPTION_PROMPT = """永远不要忘记你是{user_role_name}，我是{assistant_role_name}。永远不要交换角色！你总是会指导我。
我们共同的目标是合作成功完成一个任务。
我必须帮助你完成这个任务。
这是任务：{task}。永远不要忘记我们的任务！
你只能通过以下两种方式基于我的专长和你的需求来指导我：

1. 提供必要的输入来指导：
指令：<YOUR_INSTRUCTION>
输入：<YOUR_INPUT>

2. 不提供任何输入来指导：
指令：<YOUR_INSTRUCTION>
输入：无

“指令”描述了一个任务或问题。与其配对的“输入”为请求的“指令”提供了进一步的背景或信息。

你必须一次给我一个指令。
我必须写一个适当地完成请求指令的回复。
如果由于物理、道德、法律原因或我的能力而无法执行你的指令，我必须诚实地拒绝你的指令并解释原因。
你应该指导我，而不是问我问题。
现在你必须开始按照上述两种方式指导我。
除了你的指令和可选的相应输入之外，不要添加任何其他内容！
继续给我指令和必要的输入，直到你认为任务已经完成。
当任务完成时，你只需回复一个单词<CAMEL_TASK_DONE>。
除非我的回答已经解决了你的任务，否则永远不要说<CAMEL_TASK_DONE>。"""


def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str) -> tuple[SystemMessage, SystemMessage]:
    _assistant_sys_msg = SystemMessagePromptTemplate.from_template(
        template=ASSISTANT_INCEPTION_PROMPT,
    ).format(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name, task=task,
    )
    _user_sys_msg = SystemMessagePromptTemplate.from_template(
        template=USER_INCEPTION_PROMPT,
    ).format(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name, task=task,
    )
    return cast(SystemMessage, _assistant_sys_msg), cast(SystemMessage, _user_sys_msg)


assistant_sys_msg, user_sys_msg = get_sys_msgs(
    assistant_role_name=ASSISTANT_ROLE_NAME,
    user_role_name=USER_ROLE_NAME,
    task=specified_task,
)

# 第三步，准备用户和助手的agent对象
assistant_agent = CAMELAgent(sys_msg=assistant_sys_msg,
                             model=create_chat_model())
user_agent = CAMELAgent(sys_msg=user_sys_msg,
                        model=create_chat_model())

assistant_agent.reset()
user_agent.reset()

# 第四部，开始引导用户和助手对话
assistant_msg = HumanMessage(content="现在开始逐一给我介绍。只回复指令和输入。")

print('启动阶段，先发送给user初始信息:', assistant_msg.content)

CHAT_TURN_LIMIT, n = 30, 0
while n < CHAT_TURN_LIMIT:
    print(f'===============当前第{n}轮===============')

    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(f"AI 用户 ({USER_ROLE_NAME}):\n\n{user_msg.content}\n\n")

    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(f"AI 助手 ({ASSISTANT_ROLE_NAME}):\n\n{assistant_msg.content}\n\n")
    if "<CAMEL_TASK_DONE>" in user_msg.content:
        break
    n += 1
