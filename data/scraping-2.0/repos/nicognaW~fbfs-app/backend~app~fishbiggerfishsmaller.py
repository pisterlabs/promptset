from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from lcserve import serving

OTHER_FEWSHOT_EXAMPLES = r"""
```example-2
奶酪越多，奶酪孔越多，奶酪孔越多，奶酪越少，所以奶酪越多，奶酪越少。
```

```example-3
雨越大，跑得越快，跑得越快，淋得越少，淋得越少，雨越小，所以雨越大，雨越小。
```

```example-4
赚钱越多，花钱越多，花钱越多，钱越少，钱越少，人越穷，所以赚钱越多，人越穷。
```

```example-5
能力越大，责任越大，责任越大，错误越多，错误越多，能力越小，所以能力越大，能力越小。
```"""

EXPLANATION = r"""```explanation
鱼越大，鱼刺越多，鱼刺越多，鱼肉越少，鱼肉越少，鱼越小，所以鱼越大，鱼越小。

以上逻辑链运用了"滑坡谬误"与"前提不一致"两种经典错误手法。

完整的论述应当是，在部分种类的鱼中存在：
    - （不同种类）鱼越大，刺越多
    - （同一个体）刺越多，肉越少
    - （同一种类）肉越少，鱼越小

使用不同前提，进行连续地断言，推导出因果完全相反的结论，令人忍俊不禁

如果将其演绎形式形式化：
if a then b, if b then c, if c then d.
因此：if a then d

这个结构属于逻辑学上的modus ponens肯定前件式，逻辑上肯定是没问题的，因为演绎过程都在肯定自己的前提条件affirming antecedent，是有效演绎。逻辑学的核心原则在于，如果逻辑结构正确的情况下且所有前提为真，那么结论必定为真，这里问题就出现在前提上。如果一个演绎逻辑正确但前提如果是错误的话，就会出现“逻辑上正确但结论为假”的演绎。

评估前提“错误之处”最为关键的一点就是语意不清的问题，某些有意或无意的替换概念。
```"""

PROMPT_TEMPLATE = r"""
```example-logic-chain
鱼越大，鱼刺越多，鱼刺越多，鱼肉越少，鱼肉越少，鱼越小，所以鱼越大，鱼越小。
```

```example-logic-chain
奶酪越多，奶酪孔越多，奶酪孔越多，奶酪越少，所以奶酪越多，奶酪越少。
```

用类似以上逻辑链的格式和逻辑表达“{fish_bigger}，{fish_smaller}”。

每一步推理都**必须**合理。
每一步推理都**必须**与上一步紧密相关且看似理所当然。
如果你无法合理推出结论，就**必须**尝试推理更多步，**禁止**出现与上一步无关或不合理的推理。
Let's work this out in a step-by-step way to make sure we have the right answer.
**必须**以"{fish_bigger}，"开头并以"所以{fish_bigger}，{fish_smaller}"结尾。
你**必须仅**提供逻辑链，**禁止**附加任何其他内容。
每一步推理之间**必须**用逗号分割且**禁止**换行。
"""

TEMPERATURE = 0.75
MODEL_NAME = "gpt-3.5-turbo"


@serving(websocket=False)
async def fbfs(fish_bigger: str,
               fish_smaller: str, ) -> str:
    llm = ChatOpenAI(temperature=TEMPERATURE, model=MODEL_NAME, max_tokens=512)
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(PROMPT_TEMPLATE)
    ).run({
        "fish_bigger": fish_bigger,
        "fish_smaller": fish_smaller,
    })


@serving(websocket=True)
async def fbfs_stream(fish_bigger: str,
                      fish_smaller: str,
                      **kwargs) -> str:
    streaming_handler = kwargs.get('streaming_handler')
    llm = ChatOpenAI(temperature=TEMPERATURE,
                     model=MODEL_NAME,
                     streaming=True,
                     callback_manager=CallbackManager([streaming_handler]),
                     presence_penalty=0.75,
                     frequency_penalty=1.25,
                     )
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(PROMPT_TEMPLATE)
    ).run({
        "fish_bigger": fish_bigger,
        "fish_smaller": fish_smaller,
    })
