from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser


fake_action_chain = (
    PromptTemplate.from_template("Stub")
    | FakeListChatModel(responses=["Action chain isn't implemented yet."])
    | StrOutputParser()
)
