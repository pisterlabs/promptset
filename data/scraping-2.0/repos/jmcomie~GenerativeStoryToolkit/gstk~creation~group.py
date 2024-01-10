"""
CreationGroup

CreationGroups are a container for
"""

import asyncio
import json
from asyncio import Semaphore
from enum import StrEnum
from typing import Callable, Iterator, Optional

from pydantic import BaseModel
from tqdm import tqdm

from gstk.creation.formatters import (
    _format_instance_for_vectorization,
    format_instance_for_vectorization,
    format_node_for_vectorization,
)
from gstk.creation.graph_registry import (
    CreationEdge,
    CreationNode,
    GroupCollectionData,
    GroupProperties,
    GroupSummaryData,
    Message,
    Role,
    SelectionData,
)
from gstk.graph.interface.graph.graph import Node
from gstk.graph.registry import NodeRegistry, NodeTypeData
from gstk.graph.registry_context_manager import current_node_registry
from gstk.graph.system_graph_registry import SystemEdgeType
from gstk.llmlib.async_openai import get_chat_completion_response, get_function_tool, get_openai_vectorization


class ConflictStrategy(StrEnum):
    clone = "clone"
    move = "move"
    raise_exception = "raise"


async def get_chat_completion_object_response(
    node_type: str, messages: list[Message], generate_vector: bool = False
) -> (BaseModel, list[float]):
    """
    Sends the message list to OpenAI and returns the response as an instance of the
    configured type for the node_type. This serves as the primary bridge between
    the GFTK creation and graph pattern and OpenAI chat completion.
    """
    node_registry: NodeRegistry = current_node_registry()
    node_type_data: NodeTypeData = node_registry.get_node_type_data(node_type)
    open_ai_normalized_node_type: str = node_type.replace(".", "_")
    response = await get_chat_completion_response(
        messages, tools=[get_function_tool(open_ai_normalized_node_type, node_type_data.model)]
    )
    instance: BaseModel = node_type_data.model(
        **json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    )
    vector: Optional[list[float]] = None
    if generate_vector:
        result = await get_openai_vectorization(format_instance_for_vectorization(node_type, instance))
        vector = result.data[0].embedding
    return instance, vector


class CreationGroup:
    """
    A group of created nodes. This is the interface around a node type of CreationNode.group.
    """

    def __init__(self, node: Node):
        if node.node_type != CreationNode.group:
            raise ValueError(f"group_node must be a group node, not {node.node_type}")
        self._node = node

    @property
    def node(self) -> Node:
        return self._node

    def find(
        self,
        query: str,
        node_type_filter: list[str],
        count: int = 10,
        recursive: bool = False,
        similarity_threshold: Optional[float] = None,
    ) -> list[tuple[int, float]]:
        """
        Find nodes in the group that are similar to the query in vector space.

        Args:
            query (str): the query to search for. This is passed to OpenAI for vectorization.
            node_type_filter (list[str]): return only nodes of these types.
            count (int, optional): Limit the number of results. Defaults to 10.
            recursive (bool, optional): descend into groups. Defaults to False.
            similarity_threshold (Optional[float], optional): Only return results with
                    similarity above this threshold. Defaults to None.

        Returns:
            list[tuple[int, float]]: list of tulpes of node id and similarity score.
        """
        # Determine if we need custom formatters for each node's vectorization or per-field vectorization.
        result = asyncio.run(get_openai_vectorization(query))
        query_vector: list[float] = result.data[0].embedding

        return self.node.find(
            query_vector,
            count=count,
            node_type_filter=node_type_filter,
            descend_into_types=[CreationNode.group] if recursive else [],
            descend_unexported_edges=True,
            similarity_threshold=similarity_threshold,
        )

    async def ensure_vectors(self, node_type_list: Optional[list[str]] = None, concurrent_count: int = 10):
        """
        Ensure that child nodes have vectors. If node_type_list is provided, vectorizes only
        those nodes of a type present in list.

        Args:
            node_type_list (Optional[list[str]], optional): Only vectorize nodes of these types. Defaults to None.
            concurrent_count (int, optional): Concurrent requests to OpenAI. Defaults to 10.

        XXX: This needs to be checked for correctness around semaphore acquisition.
        """

        async def process_node(node, pbar):
            if not node.vector:
                result = await get_openai_vectorization(_format_instance_for_vectorization(node.node_type, node.data))
                node.vector = result.data[0].embedding
                node.save()
                pbar.update(1)

        with self._node.graph.create_session() as session:
            async with Semaphore(concurrent_count) as sem:  # noqa
                group_node: Node = self.node.graph.get_node(session, self.node.id)
                tasks = []
                with tqdm(desc="Processing") as pbar:
                    for node in group_node.walk_tree(yield_node_types=node_type_list):
                        if not node.vector:
                            # Create a coroutine for each node that needs processing and pass the pbar
                            task = asyncio.create_task(process_node(node, pbar))
                            tasks.append(task)
                    await asyncio.gather(*tasks)
                session.commit()

    async def select(self, prompt: str, node_type_filter: Optional[list[str]] = None) -> SelectionData:
        """
        Select among the contained nodes using the given prompt. If node_type_filter is given, only nodes of those
        types will be evaluated.

        Args:
            prompt (str): the prompt to use for selection
            node_type_filter (Optional[list[str]], optional): return only nodes of these types. Defaults to None.
        Returns:
            SelectionData: the selected node identifiers and textual reason for selection
        """
        prompt_str = (
            "Fulfill this criteria to the function call, from the node id and data fields in the context, "
            f"using this prompt: {prompt}"
        )
        messages_iter: Iterator[Message] = self.list_messages_for_chat_completion(
            CreationNode.selection,
            prompt_str,
            formatter=format_node_for_vectorization,
            node_type_filter=node_type_filter,
            edge_type_filter=[SystemEdgeType.contains],
        )
        instance, _vector = await get_chat_completion_object_response(
            CreationNode.selection, messages_iter, generate_vector=False
        )
        return instance

    def set_node_referenced(self, node: int | Node):
        # Add a reference to node if the type is valid for this group.
        with self._node.graph.create_session() as session:
            if isinstance(node, int):
                node: Node = self._node.graph.get_node(self._node.session, node)
            else:
                node.session = session
            group_node: Node = self._node.graph.get_node(session, self._node.id)
            _edge, node = group_node.add_out_edge(SystemEdgeType.references, node)
            session.commit()

    def set_node_contained(
        self, node: int | Node, conflict_strategy: ConflictStrategy = ConflictStrategy.raise_exception
    ):
        # Add containment to node if the type is valid for this group.
        # Clone it if necessary. If node has no container, assign it to this group
        # as contained.
        with self._node.graph.create_session() as session:
            if isinstance(node, int):
                node: Node = self._node.graph.get_node(self._node.session, node)
            else:
                node.session = session
            for edge, node in node.get_in_nodes(edge_type_filter=[SystemEdgeType.contains]):
                if node.id == self._node.id:
                    return
                if conflict_strategy == ConflictStrategy.clone:
                    self.add_custom_node(node.node_type, node.data)
                    return
                elif conflict_strategy == ConflictStrategy.move:
                    self._node.graph.delete_edge(session, edge.id)
                    self._node.add_out_edge(SystemEdgeType.contains, node)
                    return
                elif conflict_strategy == ConflictStrategy.raise_exception:
                    raise ValueError(
                        f"Node {node.id} is already contained by group {self._node.id} and conflict strategy is "
                        f"{conflict_strategy}"
                    )
            self._node.add_out_edge(SystemEdgeType.contains, node)
            session.commit()

    def add_new_node(self, node_type: str, data: BaseModel):
        """Add a node contained by this group of the given type and data.

        Args:
            node_type (str): _description_
            data (BaseModel): _description_
        """
        # Add a node of the given type and data to this group.
        with self._node.graph.create_session() as session:
            group_node: Node = self._node.graph.get_node(session, self._node.id)
            _edge, _node = group_node.add_out_node(node_type, data, SystemEdgeType.contains)
            session.commit()

    def list_nodes_as_messages(
        self,
        formatter: Callable[[node], str] = format_instance_for_vectorization,
        node_type_filter: Optional[list[str]] = None,
        edge_type_filter: Optional[list[str]] = None,
        include_all_prompts: bool = False,
    ) -> Iterator[Message]:
        """List nodes pointed to by this group as messages that can be sent to chat completion.

        Args:
            formatter (Callable[[node], str], optional): _description_. Defaults to format_instance_for_vectorization.
            node_type_filter (Optional[list[str]], optional): _description_. Defaults to None.
            edge_type_filter (Optional[list[str]], optional): _description_. Defaults to None.
            include_all_prompts (bool, optional): _description_. Defaults to False.

        Yields:
            Iterator[Message]: _description_
        """
        self._node.refresh()
        for _edge, node in self._node.get_out_nodes(
            node_type_filter=node_type_filter, edge_type_filter=edge_type_filter
        ):
            assert isinstance(node, Node)
            if include_all_prompts:
                for _edge, prompt_node in node.get_out_nodes(
                    node_type_filter=[CreationNode.message], edge_type_filter=[CreationEdge.created_by]
                ):
                    assert isinstance(prompt_node, Node)
                    yield prompt_node.data

            if node.node_type == CreationNode.message:
                yield node.data
            else:
                yield Message(role=Role.assistant, content=formatter(node))

    def list_messages_for_chat_completion(self, node_type: str, prompt_str: str, **kwargs) -> Iterator[Message]:
        """
        Helper function for retrieving group node data, plus a user-specified prompt,
        for chat completion..
        """
        node_type_data: NodeTypeData = current_node_registry().get_node_type_data(node_type)
        if node_type_data.system_directive:
            yield Message(role=Role.system, content=node_type_data.system_directive)
        yield from self.list_nodes_as_messages(**kwargs)
        yield Message(role=Role.user, content=prompt_str)

    async def create(self, node_type: str, prompt: str, supplemental_messages: Optional[list[Message]] = None):
        """Use the given prompt to create a new node of the given type.

        Args:
            node_type (str): the node type to create
            prompt (str): the prompt to use for creation
            supplemental_messages (Optional[list[Message]], optional): _description_. Defaults to None.
        """
        # Supplemental messages are added just before the prompt and are not serialized.
        with self._node.graph.create_session() as session:
            prompt = (
                "Fulfill this criteria to the function call, avoiding duplicating what's in the context, "
                + f"using this prompt: {prompt}"
            )
            messages_iter: Iterator[Message] = self.list_messages_for_chat_completion(
                node_type,
                prompt,
                edge_type_filter=[SystemEdgeType.contains, SystemEdgeType.references],
                include_all_prompts=True,
            )

            if supplemental_messages:
                messages: list[Message] = list(messages_iter)
                # Insert supplemental messages just before the prompt.
                messages_iter = iter(messages[:-1] + supplemental_messages + [messages[-1]])

            instance, vector = await get_chat_completion_object_response(node_type, messages_iter)
            print(f"node id: {self._node.id}")
            group_node: Node = self._node.graph.get_node(session, self._node.id)
            _edge, node = group_node.add_out_node(node_type, instance, SystemEdgeType.contains, vector=vector)
            assert isinstance(node, Node)
            node.add_out_node(CreationNode.message, Message(role=Role.user, content=prompt), CreationEdge.created_by)
            session.commit()
        self._node.refresh()

    async def update(self, node: int | Node, prompt: str):
        """
        Update the data in the given node according to the prompt.
        Note that the given node must be contained by this group.

        Args:
            node (int | Node): the node instance or node id
            prompt (str): the prompt to use for updating the node

        Raises:
            ValueError: if the given node is not contained by this group
        """
        with self._node.graph.create_session() as session:
            # Reopen the group node in the new session to preserve the
            # original stored session for reading.
            group_node: Node = self._node.graph.get_node(session, self._node.id)

            if isinstance(node, Node):
                node.session = session
            if isinstance(node, int):
                node: Node = group_node.graph.get_node(session, node)
            if not group_node.graph.get_edge(session, group_node.id, node.id, SystemEdgeType.contains):
                raise ValueError(f"Node {node.id} is not contained by group {group_node.id}")

            # Load context imports.
            messages: list[Message] = list(
                self.list_nodes_as_messages(
                    formatter=format_instance_for_vectorization, edge_type_filter=[SystemEdgeType.references]
                )
            )

            # Reference node to update.
            messages.append(
                Message(
                    role=Role.assistant,
                    content="This is the node data to update:\n"
                    + _format_instance_for_vectorization(node.node_type, node.data),
                )
            )

            # Add user prompt.
            messages.append(
                Message(
                    role=Role.user,
                    content="Fulfill this criteria to the function call, as a modification to the node specified "
                    + f"for update in the context, using this prompt: {prompt}",
                )
            )

            # Get response from OpenAI.
            instance, vector = await get_chat_completion_object_response(node.node_type, messages)

            assert isinstance(instance, BaseModel)

            # Update the node.
            original_data: BaseModel = node.data
            node.data = instance
            node.vector = vector

            # Add the original node and the update prompt as messages.
            node.add_out_node(
                CreationNode.message,
                Message(role=Role.assistant, content=_format_instance_for_vectorization(node.node_type, original_data)),
                CreationEdge.created_by,
            )

            node.add_out_node(CreationNode.message, Message(role=Role.user, content=prompt), CreationEdge.created_by)

            node.save()
            assert node.session == session
            session.commit()

        # Refresh default session.
        self._node.refresh()


def new_group(
    node: Node, group_properties: GroupProperties, group_data: GroupCollectionData | GroupSummaryData
) -> Node:
    """
    Create a new group as a child of the given node with the given properties and data.
    The edge type of the group is SystemEdgeType.contains.
    """
    if not isinstance(node, Node):
        raise ValueError(f"node must be a Node, not {type(node)}")
    edge, group_node = node.add_out_node(CreationNode.group, group_properties, SystemEdgeType.contains)
    if isinstance(group_data, GroupCollectionData):
        group_node.add_out_node(CreationNode.group_collection_data, group_data, CreationEdge.metadata)
    elif isinstance(group_data, GroupSummaryData):
        group_node.add_out_node(CreationNode.group_summary_data, group_data, CreationEdge.metadata)
    else:
        raise ValueError(f"Invalid group_data type: {type(group_data)}")
    return group_node
