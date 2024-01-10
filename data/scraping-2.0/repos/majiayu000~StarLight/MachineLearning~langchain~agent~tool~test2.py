from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

def test_call() -> None:
    """Test that the agent runs and returns output."""
    llm = ChatOpenAI(temperature=0)
    tools = load_tools(["dalle-image-generator"])

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    output = agent.run("""循环四次， 每次都在上一次的 Prompt 后面添加一句“请再漂亮一些
                       一幅写实照片， 一个日本年轻女子闭着眼， 羞涩地凑向镜头， 洋溢着青春的气息。
                       循环四次， 每次都在上一次的 Prompt 后面添加一句“请再清纯一些”
    一幅近景照片， 一个韩国年轻女子黑色长发, 温柔可爱,  羞涩地凑向镜头


循环四次， 每次都在上一次的 Prompt 后面添加一句“请再媚惑一
    一幅近景照片， 一个中国年轻女子黑色长发, 温柔可爱,  羞涩地凑向镜头""")
    print(output)
    assert output is not None

test_call()