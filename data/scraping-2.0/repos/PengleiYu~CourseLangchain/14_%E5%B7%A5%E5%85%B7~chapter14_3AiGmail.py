# 通过代理访问GmailApi
# 导入与Gmail交互所需的工具包
from langchain.agents.agent_toolkits import GmailToolkit

# 从gmail工具中导入一些有用的功能
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials

# 获取Gmail API的凭证，并指定相关的权限范围
credentials = get_gmail_credentials(
    token_file="token.json",  # Token文件路径
    # todo gmail配置的权限有问题，无法写邮件只能读邮件
    scopes=["https://mail.google.com/"],  # 具有完全的邮件访问权限
    client_secrets_file="credentials.json",  # 客户端的秘密文件路径
)
# 使用凭证构建API资源服务
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

# 获取工具
tools = toolkit.get_tools()
print(tools)

# 导入与聊天模型相关的包
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# 初始化聊天模型
llm = ChatOpenAI(
    temperature=0.7,
    # model='gpt-4',
)

# 通过指定的工具和聊天模型初始化agent
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 使用agent运行一些查询或指令
result = agent.run(
    # "最近一封邮件的作者是谁？"
    "帮我写个邮件给yupenglei@126.com，内容是: Hello world"
)

# 打印结果
print(result)
