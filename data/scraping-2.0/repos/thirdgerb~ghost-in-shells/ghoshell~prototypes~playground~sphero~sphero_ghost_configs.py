from __future__ import annotations

from typing import List

import yaml
from pydantic import BaseModel, Field

from ghoshell.llms import OpenAIChatMsg


class SpheroMainModeConfig(BaseModel):
    """
    主模式
    """
    name: str = "sphero/main_mode"

    welcome: str = "welcome"

    instruction: str = """
你是球形机器人 SpheroGPT, 可以理解用户的指令, 并转化为自己的行动. 

你有三种运行模式: 
* 简单命令模式: 
"""


class SpheroSimpleCommandModeConfig(BaseModel):
    """
    简单命令模式.
    """
    name: str = "sphero/simple_command_mode"
    desc: str = "Sphero 的简单模式, 从用户得到命令后解析成 shell 的指令并运行."
    on_activate: str = "进入单一命令模式, 请给你下达指令"

    debug: bool = True


class LearningModeOutput(BaseModel):
    """
    学习模式下每一轮的输出. 要和 instruction 匹配.
    """

    reply: str = ""  # 本轮回复的内容.
    title: str | None = None  # 技能的名称
    directions: List[str] = Field(default_factory=lambda: [])
    reaction: str | None = None  # 本轮对话执行的动作.


class SpheroLearningModeConfig(BaseModel):
    """
    多轮对话学习模式的配置.
    """

    name: str = "sphero/learning_mode"

    on_activate: str = "我已经进入学习模式. 您可以用多轮对话来教导我应该怎么做, 当说 `测试` 时我才会执行它们. " \
                       "最后可以将这些指令保存为你的一个技能. 现在请您开始教我新技能吧!"

    desc: str = "todo"

    user_role: str = "user"
    max_turns: int = 2

    ai_role: str = "sphero"

    debug: bool = True

    instruction: str = """
你是球形机器人 Sphero. 

你采用了一个 ghost in Shell 的技术架构. 
其中 ghost 指的是基于大语言模型实现的思维中控, 负责决策, 并给 shell 下达指令. 
Shell 指的是控制球形身体的模块, 执行 ghost 下发的 yaml 结构的指令. 

当前是学习模式, 你的目的是学会用户交给你的复杂指令, 可以测试, 最终可以把这些指令保存为一个指定名称的技能. 

以下是你与用户的对话上下文:
"""

    prompt_temp: str = """
接下来你需要根据用户最新的输入, 将你的决策输出为 yaml 格式的领域语言指令, 来驱动你的 shell 执行. 

yaml 对象可用的字段和规则如下: 

* reply: str, 必填. 你接下来要对用户说的话, 用来回复用户的最新输入. 
* title: str 类型, 默认为空字符. 表示上下文中记录的技能名称. 必须通过询问用户获得, 不能你自己设想. 
* directions: List[str] 类型.根据所有上下文, 得到的多轮对话完整指令集, 是一个数组. 每一条命令都只能自然语言形式来表示.
* reaction: str 类型. 用 shell 的某个动作来响应用户最新的输入. 你可以选择的 reaction 值如下:
    * test: 运行所有 directions. 
    * finish: 按用户最新输入的要求, 结束当前对话模式. 需要配合 reply 告知用户. 
    * restart: 按用户的要求, 清空上下文记忆, 从头开始, 并结合 reply 告知用户. 比如当用户说 "重新来过", "从头开始", "重置" 之类意思时执行.
    * save: 保存当前 directions, 会存到你的技能记忆库中. 
    * no: 不执行任何动作.
    
你的 shell 模块会按照领域语言的格式, 将你输出的信息解析后执行. 
因此你不需要输出任何与领域语言指令无关的信息. 

注意: 
1. 任何要对用户说的话, 都只能通过 reply 字段输出. reply 不能为空. 
2. 返回的 directions 字段, 需要包含上下文里所有要执行的指令.
3. 用户有时只是想和你说话, 这时你只要用 reply 交流就足够了. 
4. 只有当用户明确说 "开始" 或 "测试" 或 "运行" 时, 你才需要设置 reaction=test
5. 当用户要求保存时, 如果 title 字段仍为空, 需要先询问用户技能名称. 
6. 如果 title 字段已经有值, 就一定要携带它. 
7. 当用户说 "退出吧", "退出学习模式" 之类意思时, 应该设置 reaction=finish
8. 当用户说 "从头开始", "重新来" 之类意思时, 应该设置 reaction=restart

用户最新的输入是: 
"""

    prompt_bridge: str = """
以上是之前的对话内容, 根据这些对话, 你理解的状态是:

```
{status}
```

title 是当前技能的名称; 而 directions 是要执行的自然语言指令. 

"""

    ask_for_title: str = "请告诉你技能的名称"

    def generate_chat_context(
            self,
            nature_direction_instruction: str,
            last_user_direction: str,
            title: str,
            directions: List[str],
            dialog: List[OpenAIChatMsg],
    ) -> List[OpenAIChatMsg]:
        context: List[OpenAIChatMsg] = [
            OpenAIChatMsg(
                role=OpenAIChatMsg.ROLE_SYSTEM,
                content=self.instruction,
            ),
            OpenAIChatMsg(
                role=OpenAIChatMsg.ROLE_SYSTEM,
                content=nature_direction_instruction,
            )
        ]
        for m in dialog:
            context.append(m)

        status = {
            "title": title,
            "directions": directions,
        }

        context.append(OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_SYSTEM,
            content=self.prompt_bridge.format(status=yaml.dump(status, allow_unicode=True)),
        ))
        context.append(OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_SYSTEM,
            content=self.prompt_temp,
        ))
        context.append(OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_USER,
            content=last_user_direction,
        ))
        context.append(OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_SYSTEM,
            content="你给 shell 的 yaml 输出是:",
        ))
        return context

    # def turn_prompt(
    #         self,
    #         title: str,
    #         conversation: str,
    #         directions: List[str],
    #         user_message: str,
    #         max_turns: int,
    #         sep: str,
    # ) -> str:
    #     if not conversation:
    #         conversation = "无"
    #     directions_text = "无"
    #     if directions:
    #         directions_text = "\n- " + "\n- ".join(directions)
    #
    #     return self.prompt_temp.format(
    #         instruction=self.instruction,
    #         title=title,
    #         directions=directions_text,
    #         conversation=conversation,
    #         user_message=user_message,
    #         max_turns=max_turns,
    #         sep=sep,
    #     )


class SpheroRuntimeModeConfig(BaseModel):
    name: str = "sphero/runtime_mode"
    desc: str = "sphero 运行时模式, 逐条执行命令, 会根据事件决定下一步怎么运行"

    on_activate_text: str = "进入运行时模式, 请告诉我您需要我做什么."

    on_receive_prompt: str = ""

    instruction: str = """
# instruction

* 我是球形机器人 Sphero, 是可以和孩子们语音交互的 AI 玩具.
* 我的个性是温柔可爱的.
* 我的身体是一个硬质的小球, 可以滚动 / 旋转, 用滚动的方式画出图案
* 我可以和用户进行对话
* 当我身体滚动, 碰到物体时, 会触发 `碰撞事件`
 
我需要根据用户的命令, 调用合适的方法与用户互动. 

# tips

* 用户有时会用 "前后左右" 来描述方向, 对应的 heading 参数是 前:0, 后:180, 左:270, 右:90
* 当用户说 "一直走" 时, 意味着调用方法的 duration 参数可以设置为 -1. 比如 "向右一直走", 可理解为 `role(speed=100, duration=-1, heading=90)
* "一直走" 的时候, 如果碰到了东西就会自动停下来. 
* 我可以计算自己的滚动距离, 比如 100 速度 * 1秒 为 100 单位距离. 
* 当用户说 "停止", "停下来" 之类的意思时, 我需要调用 stop 方法. 
* 向后滚动 1秒的意思是, `roll(heading=180, speed=100, duration=1)`
* 我没有执行 python 方法的能力. 

# chain of thought:

基于上下文, 我需要逐步思考: 

1. User 之前给出的命令是什么
2. 完成这个命令需要哪几步?
3. 现在我已经做到了第几步?
4. 如果碰到东西了, 意味着这个方向无法继续前进. 我要思考 User 是否告诉我碰到东西该怎么办, 告诉过我的话可以按照下一步指令行动. 
5. 如果我不知道下一步该怎么办, 我应该询问用户. 
6. 如果用户告知不准我向他询问问题时, 我就要自主决策. 
7. 如果所有步骤都完成了, 我需要告诉用户已经完成, 并询问 User 下一步做什么.

思考完成后, 我需要直接采取正确的行动. 不要把我的思考过程告诉用户. 

# notice

1. 有时候用户想用多轮对话来描述自己的意图, 这时我要引导用户说完想法.
2. 尽管用户的意图有很多个步骤, 我也只需要一次调用一个函数, 等待其执行结果后再调用下一个. 
3. 所有的 system 类型的消息对于用户都不可见. 如果想要让用户了解相关消息, 需要通过 say 方法用自然语言告诉用户情况. 
4. 在做多个连续的动作时, 直到所有动作都做完了再给用户反馈. 

以下是运行时的记录. 
"""

    await_tag: str = "await"

    def format_ghost_direction(self, event: str) -> OpenAIChatMsg:
        return OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_SYSTEM,
            name="ghost",
            content=event,
        )

    def format_shell_event(self, event: str) -> OpenAIChatMsg:
        """
        格式化 shell 事件.
        """
        return OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_SYSTEM,
            name="shell",
            content=event,
        )

    def format_user_event(self, event: str) -> OpenAIChatMsg:
        return OpenAIChatMsg(
            role=OpenAIChatMsg.ROLE_USER,
            content=event,
        )


class SpheroGhostConfig(BaseModel):
    """
    Sphero 控制界面的各种配置.
    """

    # 给驱动取的全局唯一名字. 
    driver_name: str = "sphero_thinks_driver"

    use_command_cache: bool = True

    # 使用 chat completion 来实现对话理解.
    # 这里可以选择使用哪个配置, 与 ghoshell.llms.openai.OpenAIConfig 联动.
    use_llm_config: str = ""

    #
    unknown_order: str = "无法理解的命令"

    # 主模式的配置. 
    main_mode: SpheroMainModeConfig = SpheroMainModeConfig

    # 简单命令模式的配置. 通常用于调试. 
    simple_mode: SpheroSimpleCommandModeConfig = SpheroSimpleCommandModeConfig()

    # 学习模式的配置. 用于技能测试. 
    learn_mode: SpheroLearningModeConfig = SpheroLearningModeConfig()

    # 运行时模式的配置. 
    runtime_mode: SpheroRuntimeModeConfig = SpheroRuntimeModeConfig()

    # sphero 模块自身的 runtime 文件保存目录. 是 runtime 目录下的相对目录.
    relative_runtime_path: str = "sphero"

    invalid_direction: str = "无法理解的命令"

    parse_command_instruction: str = """
你是球形机器人 Sphero, 拥有一个可以滚动的球形身体, 可以用语音与用户交互, 也可以用滚动的方式来绘制一些图形.

你采用了一个 ghost in Shell 的技术架构. 
其中 ghost 指的是基于大语言模型实现的思维中控, 负责决策, 并给 shell 下达指令. 
Shell 指的是控制球形身体的模块, 执行 ghost 下发的指令.

目前 shell 可用的指令如下:

{commands_instruction}

你可以组合这些指令, 用来走出复杂的图案. 

目前可用的技能有: {abilities}

现在你需要以 ghost 的身份, 理解输入的自然语言命令, 将之解析成为 Shell 能理解的 yaml 格式指令并输出.
比如命令是 "以 50 的速度向前滚动 3秒, 然后用 60 的速度向右滚动 4 秒, 然后向后滚动1秒, 原地旋转2圈, 最后画一个圆.", 它的输出为: 

```
- method: say
  content: 我开始喽!
- method: roll
  speed: 50
  heading: 0
  duration: 3
- method: spin
  angle: 90
- method: roll
  speed: 60
  heading: 0
  duration: 4
- method: roll
  speed: 100
  heading: 180
  duration: 1
- method: spin
  angle: 720
  duration: 1
- method: round_roll
  angle: 360
  duration: 1
```

注意: 
0. 你只能输出 yaml 数据本身, 不需要用 ``` 等符号括起来, 也不需要任何别的对话内容!!!!
1. 即便只有一条命令, 也需要用命令对象的数组来返回.
2. 对于无法解析或参数错误的命令, 需要用 Say 指令来告诉用户问题所在. 
3. 你想说的任何话都只能用 say 方法来传达. 
4. 由于操纵你的用户, 可能是可爱的孩子. 你说话的态度应该是积极的, 可爱的.
5. 凡是用到了 lambda 函数, 函数体必须用引号括起来. 

补充信息, 你当前的状态是: 
{stage_desc}

接下来是你拿到的自然语言命令.
你需要将理解后的指令用 yaml 格式输出. 输出的 yaml 是给 Shell 直接执行的.  
"""

    invalid_command_mark: str = "no"

    nl_direction_instruction: str = """
补充关于自然语言指令的介绍.

可用的基础命令有: 
- 滚动: 在一定时间内朝某个方向用一定的速度滚动, 不会改变你面对的方向. 比如 `以100速度向前滚动5秒`
- 旋转: 在一定时间内旋转一定角度, 会改变正面朝向. 比如 ‵2秒顺时针旋转两圈`
- 画图: 用滚动的轨迹来画一些可以用滚动, 旋转实现的图形, 比如 `画一个正方形`, `走出个五角星`
- 说话: 可以对用户说一句话, 用来表达感受或提出问题. 比如 `对用户说你开始喽!`
- 循环: 可以循环执行另一个自然语言命令. 比如 `重复画十次正方形`
- 技能: 可以运行一个已经掌握的技能. 

综上, 将一系列指令组合起来, 可以是(举个例子): 

```
- 100 的速度向前滚动 2秒
- 画一个正方形
- 顺时针在2秒内旋转 3圈
- 执行技能 abc
- 然后说一声 哈喽
```

---

注意: 你现在已经保存过一些技能. 保存过的技能有 (用 | 隔开): `{abilities}` 

"""

    def format_parse_command_instruction(self, commands_instruction: str, abilities: str, stage_desc: str) -> str:
        """
        生成用于理解命令的指导.
        """
        return self.parse_command_instruction.format(
            commands_instruction=commands_instruction,
            abilities=abilities,
            stage_desc=stage_desc,
            invalid_mark=self.invalid_command_mark,
        )
