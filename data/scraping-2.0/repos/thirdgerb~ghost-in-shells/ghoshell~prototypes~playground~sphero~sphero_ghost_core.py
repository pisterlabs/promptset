import os
from typing import Dict, List, Tuple

import yaml
from pydantic import BaseModel, Field

from ghoshell.ghost import Context, CtxTool
from ghoshell.llms import OpenAIChatCompletion, OpenAIChatMsg, OpenAIChatChoice
from ghoshell.prototypes.playground.sphero.sphero_commands import Say, commands_yaml_instruction, loop_check, \
    ability_check
from ghoshell.prototypes.playground.sphero.sphero_ghost_configs import SpheroGhostConfig, LearningModeOutput


class SpheroCommandsCache(BaseModel):
    """
    做一个假的本地 cache, 方便测试时重复使用指令但不用每次都去 prompt.
    """

    abilities: List[str] = Field(default_factory=lambda: [])
    # 命令的索引.
    indexes: Dict[str, List[Dict]] = Field(default_factory=lambda: {})


class SpheroGhostCore:

    def __init__(self, runtime_path: str, config: SpheroGhostConfig):
        self.app_runtime_path = runtime_path
        self.config = config
        self._cached_commands: SpheroCommandsCache = SpheroCommandsCache()
        self._load_commands()

    def _load_commands(self):
        filename = self._cached_commands_file()
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                yaml.safe_dump(dict(), f)
        with open(filename) as f:
            data = yaml.safe_load(f)
            self._cached_commands = SpheroCommandsCache(**data)

    def _cached_commands_file(self) -> str:
        return "/".join([
            self.app_runtime_path.rstrip("/"),
            self.config.relative_runtime_path.strip("/"),
            "commands.yaml",
        ])

    @classmethod
    def unpack_learning_mode_resp(cls, msg: OpenAIChatChoice) -> LearningModeOutput:
        """
        理解学习模式的输出.
        """
        yaml_str = cls._unpack_yaml_in_text(msg.as_chat_msg().content)
        if yaml_str.startswith("yaml\n"):
            yaml_str = yaml_str[5:]
        data = yaml.safe_load(yaml_str)
        return LearningModeOutput(**data)

    @classmethod
    def get_prompter(cls, ctx: Context) -> OpenAIChatCompletion:
        return ctx.container.force_fetch(OpenAIChatCompletion)

    def cache_command(self, command_name: str, commands: List[Dict], is_ability: bool) -> None:
        self._cached_commands.indexes[command_name] = commands.copy()
        if is_ability:
            self._cached_commands.abilities.append(command_name)
            self._cached_commands.abilities = list(set(self._cached_commands.abilities))
        self._save_cached()

    def ability_names(self) -> str:
        return "|".join(self._cached_commands.abilities)

    def invalid_order(self) -> str:
        return self.config.invalid_direction

    def parse_direction(
            self,
            ctx: Context,
            direction: str
    ) -> Tuple[List[Dict], bool]:  # 返回加工过的消息, 和 解析失败的信息.
        """
        理解一个指令, 并将它解析为 SpheroCommandMessage
        """
        try:
            commands = yaml.safe_load(direction)
            commands, ok = self.filter_commands_data(ctx, commands)
            return commands, ok
        except Exception:
            pass

        prompter = ctx.container.force_fetch(OpenAIChatCompletion)
        if self.config.use_command_cache and direction in self._cached_commands.indexes:
            command_data = self._cached_commands.indexes[direction].copy()
            return command_data, True
        else:
            stage = CtxTool.current_think_stage(ctx)
            abilities = self.ability_names()
            prompt = self.config.format_parse_command_instruction(
                commands_yaml_instruction(),
                abilities,
                stage.desc(ctx, None),
            )
            session_id = ctx.input.trace.session_id
            chat_context = [
                OpenAIChatMsg(
                    role=OpenAIChatMsg.ROLE_SYSTEM,
                    content=prompt,
                ),
                OpenAIChatMsg(
                    role=OpenAIChatMsg.ROLE_ASSISTANT,
                    name="ghost",
                    content=f"命令是: {direction}",
                ),
                OpenAIChatMsg(
                    role=OpenAIChatMsg.ROLE_ASSISTANT,
                    name="ghost",
                    content=f"yaml 输出为:",
                )
            ]
            resp = prompter.chat_completion(
                session_id,
                chat_context,
                config_name=self.config.use_llm_config,
            )
            if not resp:
                return [], False

            content = resp.as_chat_msg().content
            if content.startswith(self.config.invalid_command_mark):
                return [], False
            commands = self._unpack_commands_in_direction(content)
            result, ok = self.filter_commands_data(ctx, commands)
            if not ok:
                return [], False

            if self.config.use_command_cache:
                self._cached_commands.indexes[direction] = result.copy()
            self._save_cached()
            return result, True

    def filter_commands_data(
            self,
            ctx: Context,
            commands: List[Dict],
    ):
        result = []
        for cmd in commands:

            # loop 检查
            loop = loop_check(cmd)
            if loop is not None and loop.direction and not loop.commands:
                # 递归解析.
                commands, ok = self.parse_direction(
                    ctx,
                    loop.direction,
                )
                if not ok:
                    # todo: 可以 raise
                    return [], False
                loop.commands = commands
                result.append(loop.to_command_data())
                continue

            # ability 检查.
            ability = ability_check(cmd)
            if ability is not None and not ability.commands:
                commands = self._cached_commands.indexes.get(ability.ability_name, None)
                if commands is None:
                    return [], False
                ability.commands = commands
                result.append(ability.to_command_data())
                continue

            result.append(cmd)
        return result, True

    def nature_directions_instruction(self) -> str:
        """
        自然语言命令提示
        """
        return self.config.nl_direction_instruction.format(abilities=self.ability_names())

    def _save_cached(self):
        filename = self._cached_commands_file()
        with open(filename, 'w') as f:
            yaml.safe_dump(self._cached_commands.model_dump(), f, allow_unicode=True)

    @classmethod
    def _unpack_commands_in_direction(cls, text: str) -> List[Dict]:
        """
        解析 llm 通过 yaml 形式返回的 commands.
        """
        text = cls._unpack_yaml_in_text(text)
        command_data = yaml.safe_load(text)
        if isinstance(command_data, str):
            return [Say(content=command_data).model_dump()]
        if not isinstance(command_data, list):
            raise RuntimeError(f"invalid ghost response: {text}")
        return command_data

    @classmethod
    def _unpack_yaml_in_text(cls, text: str) -> str:
        sections = text.split("```")
        if len(sections) == 3:
            text = sections[1]
        if text.startswith("`") or text.endswith("`"):
            text.strip("`")
        return text
