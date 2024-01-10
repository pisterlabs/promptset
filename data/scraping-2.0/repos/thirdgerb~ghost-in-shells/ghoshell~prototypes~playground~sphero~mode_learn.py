from __future__ import annotations

from typing import Optional, Dict, AnyStr, List

from pydantic import BaseModel, Field

from ghoshell.framework.reactions.commands import ProcessCmdReaction
from ghoshell.framework.thinks import SingleStageThink
from ghoshell.ghost import Context, Thought, Meta, URL
from ghoshell.ghost import Operator, Reaction, Intention
from ghoshell.llms import OpenAIChatMsg
from ghoshell.messages import Text
from ghoshell.prototypes.playground.sphero.sphero_ghost_configs import SpheroLearningModeConfig, LearningModeOutput
from ghoshell.prototypes.playground.sphero.sphero_ghost_core import SpheroGhostCore
from ghoshell.prototypes.playground.sphero.sphero_messages import SpheroCommandMessage


class DialogMessage(BaseModel):
    """
    多轮对话消息.
    """
    role: str
    text: str


class LearningModeThought(Thought):
    """
    学习模式的思维结构.
    """

    priority = -1

    class Data(BaseModel):
        max_turns: int = 2
        round: int = 0
        dialog: List[OpenAIChatMsg] = Field(default_factory=lambda: [])
        directions: List[str] = Field(default_factory=lambda: [])
        title: str = ""

    data: Data = Data()

    def add_system_message(self, message: str):
        self.data.dialog.append(OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_SYSTEM,
            content=message,
        ))

    def add_message(self, message: str, from_user: bool):
        self.data.dialog.append(OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_USER if from_user else OpenAIChatMsg.ROLE_ASSISTANT,
            content=message,
        ))
        # max_lines = self.data.max_turns * 2
        # exists_lines = len(self.data.dialog)
        # if exists_lines > max_lines:
        #     self.data.dialog = self.data.dialog[exists_lines - max_lines:]

    def prepare(self, args: Dict) -> None:
        return

    def set_variables(self, variables: Dict) -> None:
        self.data = self.Data(**variables)

    def vars(self) -> Dict | None:
        return self.data.model_dump()

    def _destroy(self) -> None:
        return


class SpheroLearningModeThink(SingleStageThink):
    """
    多轮对话模式, 支持教学, 技能记忆等等等.
    需要实现的基本功能:
    1. welcome: 自我介绍
    2. 将理解的自然语言指令记录到上下文中.
    3. 测试: 将上下文中的自然语言指令, 生成为一组命令. 然后运行.
    4. 保存为技能: 将当前上下文中形成的指令, 保存为一个技能. 需要用户提供技能的名称.
    5. 要返回
    """

    def __init__(self, core: SpheroGhostCore):
        self._core = core
        self._config: SpheroLearningModeConfig = core.config.learn_mode

    def on_activate(self, ctx: "Context", this: LearningModeThought) -> Operator | None:
        this.add_message(self._config.on_activate, False)

        msg = SpheroCommandMessage()
        msg.say(self._config.on_activate)
        ctx.send_at(this).output(msg)
        return ctx.mind(this).awaits()

    def on_received(self, ctx: "Context", this: LearningModeThought) -> Operator | None:
        text = ctx.read(Text)
        if text is None or text.is_empty():
            return ctx.mind(this).rewind()
        this.data.round += 1
        prompter = self._core.get_prompter(ctx)
        session_id = ctx.input.trace.session_id
        chat_context = self._config.generate_chat_context(
            nature_direction_instruction=self._core.nature_directions_instruction(),
            title=this.data.title,
            directions=this.data.directions,
            dialog=this.data.dialog,
            last_user_direction=text.content
        )

        resp = prompter.chat_completion(session_id, chat_context, config_name=self._core.config.use_llm_config)
        if resp.as_chat_msg().content == "await":
            return ctx.mind(this).awaits()

        this.add_message(text.content, True)
        parsed = self._core.unpack_learning_mode_resp(resp)
        return self._receive_parsed_output(ctx, parsed, this)

    def _receive_parsed_output(self, ctx: Context, parsed: LearningModeOutput, this: LearningModeThought) -> Operator:
        out = SpheroCommandMessage()
        if self._config.debug:
            ctx.send_at(this).json(parsed.model_dump())

        # 完成赋值.
        if parsed.title:
            this.data.title = parsed.title
        if parsed.directions:
            this.data.directions = parsed.directions

        # 发送消息.
        if parsed.reply:
            reply = parsed.reply
            out.say(reply)
            this.add_message(reply, False)

        # 解决 title 为空的问题.
        if parsed.reaction == "save":
            if not this.data.title:
                out.say(self._config.ask_for_title)
                this.add_message(self._config.ask_for_title, False)
                ctx.send_at(this).output(out)
                return ctx.mind(this).awaits()

        if parsed.reaction == "restart":
            ctx.send_at(this).output(out)
            return ctx.mind(this).restart()
        elif parsed.reaction == "test":
            return self._run_test(ctx, this, out)
        elif parsed.reaction == "save":
            this.add_system_message(f"保存所有指令为 `{parsed.title}`")
            return self._save_case(ctx, this, out)
        elif parsed.reaction == "finish":
            ctx.send_at(this).output(out)
            return ctx.mind(this).finish()

        ctx.send_at(this).output(out)
        return ctx.mind(this).awaits()

    def _save_case(self, ctx: Context, this: LearningModeThought, message: SpheroCommandMessage) -> Operator:
        for direction in this.data.directions:
            commands, ok = self._core.parse_direction(ctx, direction)
            if not ok:
                return self._send_unknown_message(ctx, this)
            for cmd in commands:
                message.commands.append(cmd)

        self._core.cache_command(this.data.title, message.commands, True)
        return ctx.mind(this).awaits()

    def _send_unknown_message(self, ctx: Context, this: LearningModeThought) -> Operator:
        message = SpheroCommandMessage()
        text = self._core.config.unknown_order
        message.say(text)
        this.add_message(text, False)

        ctx.send_at(this).output(message)
        return ctx.mind(this).awaits()

    def _run_test(self, ctx: Context, this: LearningModeThought, message: SpheroCommandMessage) -> Operator:
        for direction in this.data.directions:
            commands, ok = self._core.parse_direction(ctx, direction)
            if not ok:
                return self._send_unknown_message(ctx, this)
            for cmd in commands:
                message.commands.append(cmd)

        direction_str = "\n -".join(this.data.directions)
        this.add_system_message(f"运行了以下指令: \n- {direction_str}")

        ctx.send_at(this).output(message)
        return ctx.mind(this).awaits()

    def url(self) -> URL:
        return URL.new_think(self._config.name)

    def to_meta(self) -> Meta:
        return Meta(
            id=self._config.name,
            kind=self._core.config.driver_name,
        )

    def desc(self, ctx: Context, thought: Thought) -> AnyStr:
        return self._config.desc

    def new_task_id(self, ctx: "Context", args: Dict) -> str:
        return self.url().new_id()

    def new_thought(self, ctx: "Context", args: Dict) -> Thought:
        thought = LearningModeThought(args)
        thought.data.max_turns = self._config.max_turns
        return thought

    def result(self, ctx: Context, this: LearningModeThought) -> Optional[Dict]:
        return None

    def intentions(self, ctx: Context) -> List[Intention] | None:
        return None

    def reactions(self) -> Dict[str, Reaction]:
        return {
            "process": ProcessCmdReaction(),
        }
