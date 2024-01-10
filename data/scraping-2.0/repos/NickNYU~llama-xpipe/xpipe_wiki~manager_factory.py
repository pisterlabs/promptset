import enum
import os

from core.helper import LifecycleHelper
from xpipe_wiki.robot_manager import XPipeWikiRobotManager, AzureXPipeWikiRobotManager

from multiprocessing import Lock

lock = Lock()


class XPipeRobotRevision(enum.Enum):
    SIMPLE_OPENAI_VERSION_0 = 1
    HUGGINGFACE_VERSION_0 = 2


class XPipeRobotManagerFactory:
    """
    CAPABLE: Dict[XPipeRobotRevision, XPipeWikiRobotManager] =
    {XPipeRobotRevision.SIMPLE_OPENAI_VERSION_0: XPipeWikiRobotManager()}
    """

    CAPABLE = dict()  # type: dict[XPipeRobotRevision, XPipeWikiRobotManager]

    @classmethod
    def get_or_create(cls, revision: XPipeRobotRevision) -> XPipeWikiRobotManager:
        with lock:
            if cls.CAPABLE.get(revision) is not None:
                return cls.CAPABLE[revision]
            if revision == XPipeRobotRevision.SIMPLE_OPENAI_VERSION_0:
                manager = cls.create_simple_openai_version_0()
            elif revision == XPipeRobotRevision.HUGGINGFACE_VERSION_0:
                manager = cls.create_huggingface_version_0()
            cls.CAPABLE[revision] = manager
            return manager

    @classmethod
    def create_simple_openai_version_0(cls) -> AzureXPipeWikiRobotManager:
        from llama.service_context import AzureServiceContextManager
        from langchain_manager.manager import LangChainAzureManager

        service_context_manager = AzureServiceContextManager(
            lc_manager=LangChainAzureManager()
        )
        from llama.storage_context import LocalStorageContextManager

        dataset_path = os.getenv("XPIPE_WIKI_DATASET_PATH", "./dataset")
        storage_context_manager = LocalStorageContextManager(
            dataset_path=dataset_path, service_context_manager=service_context_manager
        )

        robot_manager = AzureXPipeWikiRobotManager(
            service_context_manager=service_context_manager,
            storage_context_manager=storage_context_manager,
        )
        LifecycleHelper.initialize_if_possible(robot_manager)
        LifecycleHelper.start_if_possible(robot_manager)
        return robot_manager

    @classmethod
    def create_huggingface_version_0(cls) -> AzureXPipeWikiRobotManager:
        from llama.service_context import HuggingFaceChineseOptServiceContextManager
        from langchain_manager.manager import LangChainAzureManager

        service_context_manager = HuggingFaceChineseOptServiceContextManager(
            lc_manager=LangChainAzureManager()
        )

        from llama.storage_context import LocalStorageContextManager

        dataset_path = os.getenv("XPIPE_WIKI_DATASET_PATH", "./dataset")
        storage_context_manager = LocalStorageContextManager(
            dataset_path=dataset_path, service_context_manager=service_context_manager
        )

        robot_manager = AzureXPipeWikiRobotManager(
            service_context_manager=service_context_manager,
            storage_context_manager=storage_context_manager,
        )
        LifecycleHelper.initialize_if_possible(robot_manager)
        LifecycleHelper.start_if_possible(robot_manager)
        return robot_manager
