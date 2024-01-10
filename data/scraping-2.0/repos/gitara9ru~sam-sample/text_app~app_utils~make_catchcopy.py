from langchain.memory import ConversationBufferWindowMemory
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from app_utils import lambda_app_logger

logger = lambda_app_logger.get_logger()

# TODO 設定値を環境変数から取得するようにする
openAiSettings = {"temperature": 0.2, "model_name": "gpt-3.5-turbo"}


# TODO 事例をコードから切り離す
def make_catchcopy(profile):
    system_setting = """
    システムテンプレート文
    """

    # チャットモデル
    llm = ChatOpenAI(
        temperature=openAiSettings["temperature"],
        model_name=openAiSettings["model_name"],
    )

    # チャットプロンプトテンプレート
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_setting),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    # メモリ
    memory = ConversationBufferWindowMemory(k=1, return_messages=True)

    # 会話チェーン
    conversation = ConversationChain(
        memory=memory, prompt=prompt, llm=llm, verbose=False
    )
    logger.info("Set ChatModel", extra={"settings": openAiSettings})
    logger.info("Request chat to OpenAI", extra={"userChat": input})
    result = conversation.predict(input=profile)
    logger.info("Response chat from OpenAI", extra={"aiChat": result})
    return result
