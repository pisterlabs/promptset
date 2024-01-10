"""A bot that can inspect a repo."""
import secrets
from pathlib import Path

import faiss
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from mergedbots import MergedBot
from mergedbots.experimental.sequential import SequentialMergedBotWrapper, ConversationSequence

from experiments.common.bot_manager import SLOW_GPT_MODEL, bot_manager
from experiments.common.repo_access_utils import ListRepoTool
from experiments.repo_inspector.autogpt.obsolete_agent import AutoGPT, MergedBotsHumanInputRun


@SequentialMergedBotWrapper(bot_manager.create_bot(handle="RepoInspector"))
async def repo_inspector(bot: MergedBot, conv_sequence: ConversationSequence) -> None:
    """A bot that can inspect a repo."""
    # the user just started talking to us - we need to create the agent
    root_dir = (Path(__file__).parents[3] / "mergedbots").as_posix()
    tools = [
        ListRepoTool(root_dir=root_dir),
        ReadFileTool(root_dir=root_dir),
    ]

    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    model_name = SLOW_GPT_MODEL

    message = await conv_sequence.wait_for_incoming()
    await conv_sequence.yield_outgoing(await message.service_followup_for_user(bot, f"`{model_name}`"))

    chat_llm = PromptLayerChatOpenAI(
        model_name=model_name,
        model_kwargs={
            "user": str(message.originator.uuid),
        },
        pl_tags=["mb_auto_gpt", secrets.token_hex(4)],
    )
    autogpt_agent = AutoGPT.from_llm_and_tools(
        ai_name="RepoInspector",
        ai_role="Source code researcher",
        tools=tools,
        llm=chat_llm,
        memory=vectorstore.as_retriever(),
        human_feedback_tool=MergedBotsHumanInputRun(
            conv_sequence=conv_sequence,
            current_inbound_msg=message,
            bot=bot,
        ),
    )
    # Set verbose to be true
    autogpt_agent.chain.verbose = True

    # run the agent asynchronously
    await autogpt_agent.arun([message.content])
