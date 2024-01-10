from langchain.schema import BaseMessage

from dreamsboard.engine.data_structs.data_structs import IndexStruct, IndexDict
from dreamsboard.engine.schema import BaseNode
from dreamsboard.engine.storage.storage_context import StorageContext
from dreamsboard.engine.generate.code_executor import CodeExecutor
from dreamsboard.engine.generate.code_generate import CodeGenerator
from dreamsboard.engine.generate.run_generate import CodeGeneratorHandler, CodeGeneratorChain
from typing import Any, Dict, List, Generic, TypeVar, Sequence, Type, Optional
from abc import ABC, abstractmethod

from dreamsboard.engine.storage.template_store.types import BaseTemplateStore

IS = TypeVar("IS", bound=IndexStruct)
EngineBuilderType = TypeVar("EngineBuilderType", bound="BaseEngineBuilder")


class BaseEngineBuilder(Generic[IS], ABC):

    index_struct_cls: Type[IS]
    _index_struct: Optional[IS] = None

    def __init__(
            self,
            nodes: Optional[Sequence[CodeGenerator]] = None,
            storage_context: Optional[StorageContext] = None,
            index_struct: Optional[IS] = None,
            show_progress: bool = False,
            **kwargs: Any,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and nodes is None:
            raise ValueError("One of nodes or index_struct must be provided.")
        if index_struct is not None and nodes is not None:
            raise ValueError("Only one of nodes or index_struct can be provided.")
        # This is to explicitly make sure that the old UX is not used
        if nodes is not None and len(nodes) >= 1 and not isinstance(nodes[0], CodeGenerator):
            raise ValueError("nodes must be a list of CodeGenerator objects.")
        self._storage_context = storage_context or StorageContext.from_defaults()
        self._template_store = self._storage_context.template_store
        self._show_progress = show_progress

        if index_struct is None:
            assert nodes is not None
            for node in nodes:
                self._template_store.set_template_hash(node.node_id, node.hash)
            self._index_struct = self.build_index_from_nodes(nodes)

        elif storage_context is not None and nodes is None and index_struct is not None:
            for node_id, node in storage_context.template_store.templates.items():
                if node is not None:
                    self._template_store.set_template_hash(node_id, node.hash)

            self._index_struct = index_struct
            self.build_index_from_nodes(storage_context.template_store.templates.values())

        self._storage_context.index_store.add_index_struct(self._index_struct)

    @classmethod
    def from_template(
            cls: Type[EngineBuilderType],
            nodes: Optional[Sequence[CodeGenerator]] = None,
            storage_context: Optional[StorageContext] = None,
            show_progress: bool = False,
            **kwargs: Any,
    ) -> EngineBuilderType:
        """Create index from documents.

        Args:

        """
        storage_context = storage_context or StorageContext.from_defaults()
        return cls(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    @property
    def index_id(self) -> str:
        """Get the index struct."""
        return self._index_struct.index_id

    def set_index_id(self, index_id: str) -> None:
        """Set the index id.

        NOTE: if you decide to set the index_id on the index_struct manually,
        you will need to explicitly call `add_index_struct` on the `index_store`
        to update the index store.

        .. code-block:: python
            index.index_struct.index_id = index_id
            index.storage_context.index_store.add_index_struct(index.index_struct)

        Args:
            index_id (str): Index id to set.

        """
        # delete the old index struct
        old_id = self._index_struct.index_id
        self._storage_context.index_store.delete_index_struct(old_id)
        # add the new index struct
        self._index_struct.index_id = index_id
        self._storage_context.index_store.add_index_struct(self._index_struct)

    @property
    def template_store(self) -> BaseTemplateStore:
        """Get the _template_store corresponding to the index."""
        return self._template_store

    @property
    def storage_context(self) -> StorageContext:
        return self._storage_context

    @property
    def summary(self) -> str:
        return str(self._index_struct.summary)

    @summary.setter
    def summary(self, new_summary: str) -> None:
        self._index_struct.summary = new_summary
        self._storage_context.index_store.add_index_struct(self._index_struct)

    @abstractmethod
    def _build_index_from_nodes(self, nodes: Sequence[CodeGenerator]) -> IS:
        """Build the index from nodes."""

    def build_index_from_nodes(self, nodes: Sequence[CodeGenerator]) -> IS:
        """Build the index from nodes."""
        self._template_store.add_templates(nodes, allow_update=True)
        return self._build_index_from_nodes(nodes)

    @abstractmethod
    def _add_generator(self, nodes: Sequence[CodeGenerator], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""

    def add_generators(self, nodes: Sequence[CodeGenerator], **insert_kwargs: Any) -> None:
        self._template_store.add_templates(nodes, allow_update=True)
        self._add_generator(nodes, **insert_kwargs)
        self._storage_context.index_store.add_index_struct(self._index_struct)

    def add_generator(self, node: CodeGenerator, **insert_kwargs: Any) -> None:
        """Insert a document."""

        self.add_generators([node], **insert_kwargs)

        self._template_store.set_template_hash(node.node_id, node.hash)


# Create a code generator builder
class CodeGeneratorBuilder(BaseEngineBuilder[IndexDict]):
    """
    BaseEngineBuilder[IndexDict]
    BaseEngineBuilder: 构建器基类，用于构建代码生成器的构建器.
    IndexDict: 用于存储索引的数据结构,支持链式调用、序列化和反序列化
    code_gen_chain: 代码链接器，用于存储代码生成器
    用于构建代码生成器的构建器，支持链式调用、序列化和反序列化
    被链接起来的代码生成器会按照添加的顺序依次执行，生成最终的代码
    连接器可被序列化，用于保存和加载
    """
    index_struct_cls = IndexDict
    _code_gen_chain: CodeGeneratorChain = CodeGeneratorChain()

    def _build_index_from_nodes(self, nodes: Sequence[CodeGenerator]) -> IndexDict:
        """Build index from nodes."""
        if self._index_struct is None:
            index_struct = self.index_struct_cls()
        else:
            index_struct = self._index_struct
        self._add_nodes_to_index(
            index_struct, nodes, show_progress=self._show_progress
        )

        return index_struct

    def _add_generator(self, nodes: Sequence[CodeGenerator], **insert_kwargs: Any) -> None:

        self._add_nodes_to_index(self._index_struct, nodes, **insert_kwargs)

    def _add_nodes_to_index(
            self,
            index_struct: IndexDict,
            nodes: Sequence[CodeGenerator],
            show_progress: bool = False,
    ) -> None:
        """
        添加节点到索引中，节点的
        :param index_struct:
        :param nodes:
        :param show_progress:
        :return:
        """
        if not nodes:
            return

        for node in nodes:

            self._code_gen_chain.add_generator(node)
            # NOTE: remove embedding from node to avoid duplication
            node_without_copy = node.copy()

            index_struct.add_node(node_without_copy)

    def remove_last_generator(self):
        # 先删除缓存中的代码生成器
        self._index_struct.delete(doc_id=self._code_gen_chain.chain_tail.generator.node_id)
        self._template_store.delete_template(self._code_gen_chain.chain_tail.generator.node_id, raise_error=False)
        self._code_gen_chain.remove_last_generator()

    def build_executor(self, render_data: dict = {}) -> CodeExecutor:
        if self._code_gen_chain.chain_head is None:
            raise RuntimeError("chain_head is None.")

        executor_code = self._code_gen_chain.generate(render_data)
        self.summary = executor_code
        executor = CodeExecutor(executor_code=executor_code)
        return executor
