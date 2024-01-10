from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool

llm = OpenAI(model_name="text-davinci-003")
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True,
)
agent_executor.run("""
Pytorchで単層ニューラルネットワークを作ってください。
この時、「-10~10」を定義域としたy=2xのデータを30個生成し、そのデータで100 epoch学習してください。
最後に、そのモデルを用いて、x=5に対する予測を行ってください。
""")
