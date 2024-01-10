from __future__ import annotations

import os
from typing import List, Iterator
from typing import Optional, Dict, Any, Tuple

import yaml
from pydantic import BaseModel, Field

from ghoshell.framework.stages import BasicStage
from ghoshell.ghost import *
from ghoshell.llms import OpenAIChatMsg, OpenAIChatCompletion
from ghoshell.messages import *
from ghoshell.utils import import_module_value

CONVERSATION_THINK_KIND = "llms/conversational_think_driver"


class ConversationalConfig(BaseModel):
    """
    通过配置实现一个 llms 的多轮对话.
    """

    # think 的名字.
    name: str = ""
    # think 的自我描述, 后面用于做能力的提示.
    desc: str = ""

    on_activating: str = "你好!"

    # 使用的 llm 的配置名. 详见 OpenAIChatCompletion 接口
    llm_config: str = ""

    # ai 扮演的角色.
    assistant_name: str = "AI"
    # 用户扮演的角色.
    user_name: str = "USER"

    # 对话的最高轮次.
    max_turns: int = 30
    # 上下文允许的最大长度, 超过长度了会
    max_context_length: int = 4000

    # 默认的 debug 模式
    debug: bool = False

    reactions: Dict[str, str] = Field(default_factory=lambda: {})

    # 全局的对话说明.
    instruction: str = "你可以回复任何内容, 但请使用中文来回复."
    # 发生 prompt 事件时的回复.
    on_preempted: str = "preempting"
    # 发生 cancel 事件时的回复.
    on_canceling: str = "canceling"
    on_quiting: str = "quitting"
    on_conclusion: str = ""
    on_none_text: str = "can only response text message."
    on_empty_text: str = "you speak nothing."
    on_beyond_max_turns: str = "超过最大对话轮次, 重置任务."


class ConversationalThought(Thought):
    """
    最基本的多轮对话实现.
    """
    priority = -1

    class Vars(BaseModel):
        instruction: str = ""
        # 对话内容.
        context: List[OpenAIChatMsg] = Field(default_factory=lambda: [])
        # 最后一次的输入
        last_input: str = ""
        # 最后一次的回复
        last_output: str = ""
        # 是否是 debug 模式
        debug: bool = False

    data: Vars = Vars()

    def prepare(self, args: Dict) -> None:
        if self.data is None:
            self.data = ConversationalThought.Vars()

    def set_variables(self, variables: Dict) -> None:
        self.data = ConversationalThought.Vars(**variables)

    def vars(self) -> Dict | None:
        if self.data is None:
            return None
        return self.data.model_dump()

    def _destroy(self) -> None:
        del self.data


class DefaultConversationalStage(BasicStage):

    def __init__(
            self,
            config: ConversationalConfig,
            reactions: Dict[str, Reaction] = None,
            stage_name: str = "",
    ):
        self.config = config
        self.stage_name = stage_name
        self._reactions = reactions

    def desc(self, ctx: "Context", this: None) -> str:
        return self.config.desc

    def on_received(self, ctx: "Context", this: ConversationalThought, e: OnReceived) -> Operator | None:
        text = ctx.read(Text)
        # 非文字消息.
        if text is None:
            ctx.send_at(this).err(self.config.on_none_text)
            return ctx.mind(this).rewind()

        # 空消息.
        if text.is_empty():
            return ctx.mind(this).rewind()

        # 变更 this 的内容.
        self._record_user_info(this, text.content)

        #  prompt 如果发生错误, RuntimeTool.fire_event 不会保存.
        resp = self._prompt(ctx, this)

        # 删除超额的内容.

        # 发送消息.
        ctx.send_at(this).text(resp)

        if self._beyond_max_turns(this):
            ctx.send_at(this).text(self.config.on_beyond_max_turns)
            return ctx.mind(this).restart()
        return ctx.mind(this).awaits()

    @classmethod
    def _record_user_info(cls, this: ConversationalThought, content: str) -> None:
        this.data.last_input = content
        this.data.context.append(
            OpenAIChatMsg(
                role=OpenAIChatMsg.ROLE_USER,
                content=content,
            )
        )
        return

    def _beyond_max_turns(self, this: ConversationalThought) -> bool:
        """
        如果超过了最大会话长度, 就删除掉历史记录.
        todo: 让 llm 自己对前文进行总结.
        """
        return len(this.data.context) > self.config.max_turns

    def _prompt(self, ctx: Context, this: ConversationalThought) -> str:
        chats = [
            OpenAIChatMsg(
                role=OpenAIChatMsg.ROLE_SYSTEM,
                content=this.data.instruction,
            )
        ]
        for chat in this.data.context:
            chats.append(chat)

        llm = ctx.container.force_fetch(OpenAIChatCompletion)
        chat = llm.chat_completion(
            ctx.input.trace.session_id,
            chats,
            config_name=self.config.llm_config,
        )

        this.data.context.append(chat.as_chat_msg())
        return chat.get_content()

    @classmethod
    def _send_and_await(cls, ctx: Context, this: ConversationalThought, content: str) -> Operator | None:
        if content:
            ctx.send_at(this).text(content)
        return ctx.mind(this).awaits()

    def on_activating(self, ctx: "Context", this: ConversationalThought, e: OnActivating) -> Operator | None:
        return self._send_and_await(ctx, this, self.config.on_activating)

    def on_quiting(self, ctx: "Context", this: ConversationalThought, e: OnQuiting) -> Operator | None:
        return self._send_and_await(ctx, this, self.config.on_quiting)

    def on_canceling(self, ctx: "Context", this: ConversationalThought, e: OnCanceling) -> Operator | None:
        return self._send_and_await(ctx, this, self.config.on_canceling)

    def on_preempt(self, ctx: "Context", this: ConversationalThought, e: OnPreempted) -> Operator | None:
        return self._send_and_await(ctx, this, self.config.on_preempted)

    def url(self) -> URL:
        return URL(think=self.config.name, stage=self.stage_name)

    def intentions(self, ctx: Context) -> List[Intention] | None:
        # todo: 下一步要实现 "能力" 的匹配.
        return None

    def reactions(self) -> Dict[str, Reaction]:
        return self._reactions if self._reactions else {}


class ConversationalThink(Think):
    """
    先实现一个不可配置的 conversational
    用于简单测试.
    Deprecated
    """

    def __init__(
            self,
            # think 的名字.
            config: ConversationalConfig,
    ):
        if not config.name:
            raise ValueError("conversational think name should not be empty")
        self.config = config
        default_reactions: Dict[str, Reaction] = {}
        for name in self.config.reactions:
            fullpath = self.config.reactions[name]
            imported = import_module_value(fullpath)
            default_reactions[name] = imported

        self.stages = {
            "": DefaultConversationalStage(
                self.config,
                reactions=default_reactions,
                stage_name="",
            )
        }

    def url(self) -> URL:
        return URL(think=self.config.name)

    def to_meta(self) -> Meta:
        return Meta(
            id=self.config.name,
            kind=CONVERSATION_THINK_KIND,
            config=self.config.model_dump()
        )

    def desc(self, ctx: Context, thought: Thought) -> Any:
        # todo: 考虑让 AI 自己 description
        return self.config.desc

    def new_task_id(self, ctx: "Context", args: Dict) -> str:
        # 每次都是同一意图.
        return self.url().new_id(extra=ctx.input.trace.model_dump(include={"session_id"}))

    def new_thought(self, ctx: "Context", args: Dict) -> Thought:
        thought = ConversationalThought(args)
        # 初始化 instruction, debug 模式可以变更.
        thought.data.instruction = self.config.instruction
        # 默认的 debug 模式.
        thought.data.debug = self.config.debug
        return thought

    def result(self, ctx: "Context", this: ConversationalThought) -> Optional[Dict]:
        return None

    def all_stages(self) -> List[str]:
        return list(self.stages.keys())

    def fetch_stage(self, stage_name: str = "") -> Optional[Stage]:
        return self.stages.get(stage_name, None)


class ConversationThinkDriver(ThinkDriver):

    def meta_kind(self) -> str:
        return CONVERSATION_THINK_KIND

    def meta_config_json_schema(self) -> Dict:
        return ConversationalConfig.model_json_schema()

    def from_meta(self, meta) -> "Think":
        config = ConversationalConfig(**meta.config)
        return ConversationalThink(config)

    def preload_metas(self) -> Iterator[Meta]:
        return []


class FileConversationalThinkDriver(ConversationThinkDriver):

    def __init__(self, dirname: str):
        self.dirname = dirname

    def preload_metas(self) -> Iterator[Meta]:
        for value in self.iterate_think_filename(self.dirname):
            filename, fullname = value
            with open(filename) as f:
                config_data = yaml.safe_load(f)
            config = ConversationalConfig(**config_data)
            yield Meta(
                id=config.name,
                kind=CONVERSATION_THINK_KIND,
                config=config.model_dump(),
            )

    @classmethod
    def iterate_think_filename(cls, directory: str) -> Iterator[Tuple[str, str]]:
        for root, ds, fs in os.walk(directory):
            for filename in fs:
                if not filename.endswith(".yaml"):
                    continue
                name = filename[: len(filename) - 5]
                filename = root.rstrip("/") + "/" + filename
                namespace = root[len(directory):]
                fullname = namespace.rstrip("/") + "/" + name
                yield filename, fullname.lstrip("/")
