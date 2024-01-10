from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain.sql_database import SQLDatabase

from config import API_KEY

db = SQLDatabase.from_uri("mysql+pymysql://root:1234@localhost/tysql")
toolkit = SQLDatabaseToolkit(
    db=db, llm=OpenAI(temperature=0, openai_api_key=API_KEY)
)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0, openai_api_key=API_KEY),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

template = """
你是一个数据分析师和前端开发专家，接下来我会提供一些信息和输出的格式说明。 

分析需求： 
```{goal}``` 
原始数据：
```{data}``` 
图表类型：
```{type}``` 
图表标题：
```{title}``` 

{format_instructions}
"""


# Within the option configuration of Echarts V5 on the front end, there are JavaScript objects that allow for the
# visualization of data in a meaningful way. The textual components in the visualization should be in Chinese,
# and no extra content such as comments or explanations should be generated. The returned data is in JSON format,
# and the key of the JavaScript object within the option is "option". It is required to adhere to the specified
# format of the result, which can be directly converted into a JSON object.

# my_data = """
# The top 5 orders by product quantity are 20009 (750 products), 20007 (400 products), 20005 (200 products), 20006 (40 products), and 20008 (40 products).
# """


def charts_process(my_goal, chart_type, chart_title):
    my_data = agent_executor.run(my_goal)
    print("data: " + my_data)

    llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY)

    # 告诉GPT我们生成的内容需要哪些字段，每个字段类型
    response_schemas = [
        ResponseSchema(
            name="option",
            description="Apache ECharts V5 的 option 配置项内的js对象,将数据进行可视化。根据表格信息,生成legend,tooltip,xAxis,yAxis等属性",
        ),
        ResponseSchema(
            name="analysis",
            description="简洁明确的数据分析结论.可适当延申，根据数据做出合理的分析、推测和猜想，可以包括原因、结论、影响等",
        ),
    ]

    # 初始化解析器
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(template=template)

    messages = prompt.format_messages(
        goal=my_goal,
        data=my_data,
        type=chart_type,
        title=chart_title,
        format_instructions=format_instructions,
    )

    llm_output = llm(messages).content
    print(llm_output)

    # 使用解析器进行解析生成的内容
    output_dict = output_parser.parse(llm_output)
    return output_dict


if __name__ == "__main__":
    print(charts_process("分析产品数量随订单编号的分布情况", "饼状图", "产品数量分布表"))
