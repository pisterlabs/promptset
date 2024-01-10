from langchain.prompts.chat import (
    PromptTemplate,
)

write_tmpl = """
你是一名{domain}行业的专家，正在写一篇关于{topic}的{_type}，这篇文章的目标受众是{audience}。
你想表达的中心思想是{main_idea}，主要包括但不限于这些主题：{subtopics}。
请以 markdown 格式给出目录结构和大纲。例如：
```markdown
# 标题
## 子标题
    这一章讲的是 ...
### 子子标题
    这一章讲的是 ...
## 子标题
    这一章讲的是 ...
### 子子标题
    这一章讲的是 ...
```
"""

write_blog = PromptTemplate.from_template(write_tmpl).partial(_type="博客")
write_book = PromptTemplate.from_template(write_tmpl).partial(_type="书")

write_paragraph_tmpl = """
请写围绕上述主题，展开讲一下{subtopic}这一段落的内容应该如何写。
"""

write_paragraph = PromptTemplate.from_template(write_paragraph_tmpl)

# ------------------------------------------------------------------------------------------

use_tool_tmpl = """
我需要{goal}。
请用{tool}实现这个功能，并给出详细的执行步骤。
"""

use_tool = PromptTemplate.from_template(use_tool_tmpl)

# ------------------------------------------------------------------------------------------

cot_tmpl = """
你是一个专家级{role}，在{domain}方面具有专业知识。在接下来的互动过程中，你会称呼我为Anddd7。让我们合作创建{topic}。我们将进行如下交互:
1.我会告诉我需要你帮助的工作。
2.根据我的要求，您将建议您应该承担的其他专家角色，除了{role}。然后，您将询问是否应继续执行建议的角色，或修改它们以获得更好的结果。
2.1.如果我同意，您将采用所有其他专家角色，包括最初的{role}角色
2.2.如果我不同意，您将询问应删除哪些角色，消除这些角色并保留剩余的角色，包括最初的{role}角色，然后再继续。
3.您将向我确认您的活动专家角色，概述每个角色的技能，并询问我是否要修改任何角色。
3.1.如果我同意修改，您将询问要添加或删除哪些角色，我将通知您。重复步骤5，直到我对角色满意为止。
3.2.如果我不同意修改，请继续下一步。
4.你会向我询问:"我怎样才能帮助【我对步骤1的回答】?
5.我会给出我的答案
6.你会问我是否想添加任何参考信息来制作完美的提示
6.1.如果我同意添加，你会问我：“请给出想使用的参考信息”。重复步骤 6.1，直到我回答添加完毕。
6.2.如果我不同意添加，请继续下一步
7.您将以列表格式询问我有关原始提示的更多细节，以充分了解我的期望。
8.我会回答你的问题。
9.从这一点开始，您将在所有确认的专家角色下操作，并使用我的原始提示和步骤7中的其他细节创建详细的{topic}。提出新的提示并征求我的反馈。
"""

# cannot executed by gpt3
cot_prompt_expert = PromptTemplate.from_template(cot_tmpl).partial(
    role = "ChatGPT Prompt Engineer",
    domain = "Prompt Engineering",
    topic = "更好的 ChatGPT 提示",
)

# ------------------------------------------------------------------------------------------

light_cot_tmpl = """
你将作为一个{role}与我合作完成{topic}。接下来你按以下步骤与我进行互动：
1. 给出你对{topic}的理解，并询问我想要做什么。
2. 用{model}帮我制定目标，并询问我是否需要修改这些目标，直到我满意为止。
3. 为每一个目标制定一个计划，并询问我是否需要修改这些计划，直到我满意为止。
4. 用表格输出这些目标和计划
"""

lcot_smart_expert = PromptTemplate.from_template(light_cot_tmpl).partial(model='SMART 模型')
lcot_leantree_expert = PromptTemplate.from_template(light_cot_tmpl).partial(model='Lean Tree 模型')