from __future__ import annotations

from typing import Dict, List, Type, Optional, AnyStr

from pydantic import BaseModel, Field

from ghoshell.ghost import Context, Reaction, Intention, Think
from ghoshell.ghost import OnReceived
from ghoshell.ghost import Operator
from ghoshell.ghost import Stage, Thought, Meta, URL
from ghoshell.llms import OpenAIChatMsg
from ghoshell.llms.thinks import AgentStage, AgentThought, AgentStageConfig, LLMFunc
from ghoshell.messages import Text
from ghoshell.prototypes.playground.sphero.sphero_commands import defined_commands, Say, LambdaSpeak
from ghoshell.prototypes.playground.sphero.sphero_ghost_core import SpheroGhostCore
from ghoshell.prototypes.playground.sphero.sphero_llm_func import SpheroLLMFunc
from ghoshell.prototypes.playground.sphero.sphero_messages import SpheroEventMessage, SpheroCommandMessage


class SpheroDirection(BaseModel):
    direction: str = Field(
        description="用自然语言形式描述的命令"
    )


class SpheroRuntimeThought(AgentThought):
    priority = -1

    def say(self, ctx: Context, message: str, name: str | None = None):
        _output = SpheroCommandMessage()
        _output.say(message)
        ctx.send_at(self).output(_output)
        self.data.add_ai_message(message, name)


class SpheroRuntimeModeThink(Think, AgentStage):

    def __init__(self, core: SpheroGhostCore):
        self._core = core
        config = self._core.config.runtime_mode
        self._mode_config = config
        stage_config = AgentStageConfig(
            name="",
            desc=config.desc,
            instruction=config.instruction,
            on_activate_text=config.on_activate_text,
            on_receive_prompt=config.on_receive_prompt,
            llm_config_name=self._core.config.use_llm_config,
        )
        super().__init__(config.name, stage_config)

    def url(self) -> URL:
        return URL.new(think=self._mode_config.name)

    def to_meta(self) -> Meta:
        return Meta(
            id=self._mode_config.name,
            kind=self._core.config.driver_name,
        )

    def desc(self, ctx: Context, thought: Thought | None) -> AnyStr:
        return self._mode_config.desc

    def new_task_id(self, ctx: "Context", args: Dict) -> str:
        return self.url().new_id()

    def new_thought(self, ctx: "Context", args: Dict) -> Thought:
        return SpheroRuntimeThought(args)

    def result(self, ctx: Context, this: Thought) -> Optional[Dict]:
        return None

    def all_stages(self) -> List[str]:
        return [""]

    def fetch_stage(self, stage_name: str = "") -> Optional[Stage]:
        if stage_name == "":
            return self
        return None

    def on_received(self, ctx: "Context", this: SpheroRuntimeThought, e: OnReceived) -> Operator | None:
        """
        runtime 模式可能收到三种类型的消息.
        1. 命令被中断了.
        2. 命令运行完成.
        """
        # 自然语言消息.
        text = ctx.read(Text)
        if text is not None:
            if text.is_empty():
                return ctx.mind(this).rewind()
            return self._on_receive_text(ctx, this, text)

        # 事件类消息
        event = ctx.read(SpheroEventMessage)
        if event is not None:
            return self._on_receive_event(ctx, this, event)

        return ctx.mind(this).rewind()

    def _on_receive_text(self, ctx: Context, this: SpheroRuntimeThought, text: Text):
        """
        处理用户的文字消息.
        """
        this.data.add_user_message(text.content)
        return self.on_receive_prompt(ctx, this)

    def _llm_basic_chat_context(self, ctx: Context, this: AgentThought) -> List[OpenAIChatMsg]:
        chat_context = super()._llm_basic_chat_context(ctx, this)
        chat_context.append(OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_SYSTEM,
            content=f"当前拥有的技能: {self._core.ability_names()}"
        ))
        return chat_context

    def _on_receive_event(self, ctx: Context, this: SpheroRuntimeThought, event: SpheroEventMessage) -> Operator:
        is_ran: bool = False
        for log in event.runtime_logs:
            index = log.find("|")
            method = log[:index]
            log_text = log[index + 1:]

            # hack 一下
            if method == Say.method or method == LambdaSpeak.method:
                this.data.add_system_message(f"you've spoke: `{log_text}`")
            else:
                is_ran = True
                this.data.add_system_message(f"you called method: `{method}`; result is : `{log_text}`")
        #
        # if event.stopped:
        #     message = f"指令运行中断, 原因: {event.stopped}"
        #     this.data.add_system_message(message)
        if is_ran:
            return self.on_receive_prompt(ctx, this)
        return ctx.mind(this).awaits()

    #
    # def on_llm_text_message(self, ctx: Context, this: AgentThought, message: str) -> Operator:
    #     """
    #     llm 返回了一个文字消息, 而不是函数调用.
    #     """
    #     this.say(ctx, message)
    #     return ctx.mind(this).awaits()

    def _llm_funcs(self, ctx: Context) -> List[LLMFunc]:
        funcs = super()._llm_funcs(ctx)
        for cmd_method in defined_commands:
            cmd = defined_commands[cmd_method]
            funcs.append(SpheroLLMFunc(self._core, cmd))
        return funcs

    def method_as_funcs(self) -> Dict[str, Type[BaseModel] | None]:
        return {
            # "fn_run_direction": SpheroDirection,
            "fn_await": Say,
            "fn_restart": None,
        }

    def fn_await(self, ctx: Context, this: SpheroRuntimeThought, args: Say):
        """
        说一句话, 不做任何事情, 等待用户的下一条消息的输入.
        """
        if args and args.content:
            args.content = args.content.replace("fn_await", "")
            this.data.add_ai_message(args.content)
            msg = SpheroCommandMessage()
            msg.add(args)
            ctx.send_at(this).output(msg)
        return ctx.mind(this).awaits()

    def fn_restart(self, ctx: Context, this: SpheroRuntimeThought, args: None):
        """
        清空上下文, 重新开始对话. 当用户说 "重新开始", "从头开始" 之类指令时执行.
        """
        return ctx.mind(this).restart()

    def fn_run_direction(self, ctx: Context, this: SpheroRuntimeThought, args: SpheroDirection):
        """
        用自然语言描述一系列的指令, 执行完毕后等待用户输入. 可以用来实现复合动作. 但只能用自然语言描述命令.
        """
        commands, ok = self._core.parse_direction(ctx, args.direction)
        if ok:
            message = SpheroCommandMessage(direction=args.direction, runtime_mode=True)
            message.commands = commands
            ctx.send_at(this).output(message)
        else:
            this.data.add_system_message(f"direction is invalid: {args.direction}")
        return ctx.mind(this).awaits()

    def intentions(self, ctx: Context) -> List[Intention] | None:
        return None

    def reactions(self) -> Dict[str, Reaction]:
        return {}
