from typing import List
from schema import Message, ScheduleItem, NextRoundSignal
import os
import copy
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(base_url=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY"))

assign_prompt_tpl = """
请根据仔细分析以下对话记录，针对最后一条记录从以下提供的对话角色中，选择需要回复的角色:

对话角色:
{role_description}

请使用 json 格式回复，格式如下，必须包含以上所有提供的角色且不能出现以上没有提供的角色:
{{
    \"角色名称1\": true,  // 表示角色1需要回复
    \"角色名称2\": false, // 表示角色2不需要回复
    ...
}}

历史对话记录:
{history_conversation}

最后一条记录:
{last_conversation}
"""


class AgentScheduler:
    DEFAULT_COORDINATOR_SYS_MSG = '你是一个优秀的对话场景分析师，你可以根据对话内容，找出下一个需要回复的角色'

    def __init__(self,
                 human_role: ScheduleItem,
                 agent_role_ls: List[ScheduleItem],
                 default_schedule_role: ScheduleItem,
                 history: List[Message],
                 system_msg: str | None = None):

        self.human_role = human_role
        self.schedule_role_ls = [human_role] + agent_role_ls
        self.default_schedule_role = default_schedule_role
        self.system_msg = system_msg or self.DEFAULT_COORDINATOR_SYS_MSG
        self.history = copy.deepcopy(history)
        self.reply_msgs: List[Message] = []

    def run(self, max_round: int = 5) -> List[Message]:

        current_signal = NextRoundSignal.CONTINUE
        current_round = 0

        while current_signal == NextRoundSignal.CONTINUE and current_round < max_round:
            current_round += 1
            current_signal = self.step()

        return self.reply_msgs

    def step(self) -> NextRoundSignal:

        # 调度需要回复的角色
        roles_to_reply = self.assign_job()

        # 非用户角色的进行调用回复
        for role in roles_to_reply:
            if role != self.human_role:
                msg = self.agent_reply(role)

                # 记录到回复消息中
                self.reply_msgs.append(msg)

        # 判断下一轮信号
        if self.human_role in roles_to_reply:
            return NextRoundSignal.BREAK
        else:
            return NextRoundSignal.CONTINUE

    @property
    def total_history(self):
        return self.history + self.reply_msgs

    def agent_reply(self, agent_role: ScheduleItem) -> Message:
        content = input(f'`角色{agent_role.id}`输入回复内容: ')
        return Message(role=agent_role.role.role, name=agent_role.role.name, content=content)

    def assign_job(self) -> List[ScheduleItem]:

        prompt = assign_prompt_tpl.format(role_description=self.prompt_role_description(),
                                          history_conversation=self.prompt_history_conversation(self.total_history),
                                          last_conversation=self.prompt_last_conversation(self.total_history))
        print(prompt)
        # return
        completion = client.chat.completions.create(
            model='gpt-4',
            seed=1,
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": self.system_msg},
                {"role": "user", "content": prompt},
            ]
        )

        # 解析返回 json 数据
        result = json.loads(completion.choices[0].message.content)

        # 提取返回的角色
        roles_to_reply = [role for role in self.schedule_role_ls if result.get(f'角色{role.id}', False) is True]
        print('openai 返回: ', [(x.role.name, x.id) for x in roles_to_reply])

        # 如果没有角色回复，则选择默认角色进行回复
        if len(roles_to_reply) == 0:
            roles_to_reply = [self.default_schedule_role]

        return roles_to_reply

    def prompt_default_role(self) -> str:
        return f'角色{self.default_schedule_role.id}'

    def prompt_role_description(self) -> str:
        prompt_ls = []
        for schedule_role in self.schedule_role_ls:
            prompt = f'- `角色{schedule_role.id}`: {schedule_role.role.description}; '
            prompt += f'{schedule_role.skill.skill_desc if schedule_role.skill else ""}'
            prompt_ls.append(prompt)

        return '\n'.join(prompt_ls)

    @staticmethod
    def prompt_history_conversation(history: List[Message]) -> str:
        return '\n'.join([f'{msg.name}: {msg.content}' for msg in history])

    @staticmethod
    def prompt_last_conversation(history: List[Message]) -> str:
        last_ctx_msg = history[-1]
        return f'{last_ctx_msg.name}: {last_ctx_msg.content}'
