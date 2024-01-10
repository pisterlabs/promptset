"""Main module."""
import ast
import itertools
import re
from dataclasses import dataclass
from hashlib import md5
from typing import Iterable, List, Optional, Any, Tuple, NamedTuple, TypeVar, Type, cast, Callable
from uuid import uuid4, UUID
import tiktoken
from scipy import spatial  # type: ignore

from py_bugs_open_ai.constants import DEFAULT_MODEL, DEFAULT_IS_BUG_RE, FIND_BUGS_SYSTEM_CONTENT
from .models.base import CacheProtocol
from .models.examples import Example
from .models.open_ai import Message, Role
from .open_ai_client import OpenAiClient

AstT = TypeVar('AstT', bound=ast.AST)


def _cosine_wrapper(u: List[float], v: List[float]) -> float:
    # wrapper to correctly type spatial.distance.cosine()
    return spatial.distance.cosine(u, v)


@dataclass
class CodeChunk:
    file: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int
    code: str
    peer_group: UUID
    token_count: int
    error: Optional[str] = None
    warning: Optional[str] = None

    def get_hash(self):
        return md5(self.code.encode()).hexdigest()[:10]

    def set_exception(self, message: str, error: bool) -> 'CodeChunk':
        prefix = 'ERROR' if error else 'WARNING'
        if error:
            self.error = f"{prefix}: {message}"
        else:
            self.warning = f"{prefix}: {message}"
        return self


T = TypeVar('T')


def coalesce(*args: Optional[T]) -> T:
    for arg in args:
        if arg is not None:
            return arg
    raise TypeError('At least one argument needs to not be None')


class CodeChunker(ast.NodeVisitor):
    def __init__(self, code: str, file: str, max_chunk_size: int, model: str = DEFAULT_MODEL,
                 abs_max_chunk_size: int = -1, strict_chunk_size: bool = False):
        self.model = model
        self.max_chunk_size = max_chunk_size
        self.strict_chunk_size = strict_chunk_size
        self.file = file
        if abs_max_chunk_size < 0:
            self.abs_max_chunk_size = self.max_chunk_size
        else:
            self.abs_max_chunk_size = abs_max_chunk_size
        self._chunks_by_peer_group: List[List[CodeChunk]] = []
        self._current_peer_group = uuid4()
        self._code_lines = code.split('\n')
        self._tree = ast.parse(code)

        self.visit(self._tree)

    def get_chunks(self) -> Iterable[CodeChunk]:
        for peer_group in self._chunks_by_peer_group:
            yield from self.chunk_up_peer_group(peer_group)

    def _get_chunk_size_exception_message(self, chunk: CodeChunk) -> str:
        return f"Chunk size {chunk.token_count} bigger than max size {self.abs_max_chunk_size}"

    def chunk_up_peer_group(self, peer_group: List[CodeChunk]) -> Iterable[CodeChunk]:
        if peer_group:
            if len(peer_group) >= 2:
                total_token_count = self.combine_from_to_chunks(peer_group[0], peer_group[-1]).token_count
            else:
                total_token_count = peer_group[0].token_count

            goal_min_size = self.get_goal_min_size(
                total_token_count=total_token_count,
                max_chunk_size=self.max_chunk_size
            )
            last_chunk: Optional[CodeChunk] = None
            for chunk in peer_group:
                if last_chunk is not None:
                    concat_chunk = self.combine_from_to_chunks(last_chunk, chunk)
                else:
                    concat_chunk = chunk

                if concat_chunk.token_count >= goal_min_size:
                    if concat_chunk.token_count <= self.max_chunk_size:
                        yield concat_chunk
                        last_chunk = None
                    elif last_chunk:
                        assert last_chunk.token_count <= self.abs_max_chunk_size
                        yield last_chunk
                        last_chunk = chunk
                    else:
                        assert concat_chunk is chunk, 'These should be the same in this case'
                        if chunk.token_count <= self.abs_max_chunk_size:
                            yield chunk
                        else:
                            chunk = chunk.set_exception(
                                self._get_chunk_size_exception_message(chunk), error=self.strict_chunk_size
                            )
                            yield chunk
                else:
                    last_chunk = concat_chunk

            if last_chunk is not None:
                yield last_chunk

    @staticmethod
    def get_goal_min_size(total_token_count: int, max_chunk_size: int) -> int:
        goal_min_size = total_token_count  # will try to get each chunk up to this size
        goal_num_chunks = 1
        while goal_min_size > max_chunk_size:
            goal_num_chunks += 1
            goal_min_size = total_token_count // goal_num_chunks
        return goal_min_size

    def get_token_count(self, code: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(code))

    def chunk_from_node(self, node: ast.AST) -> Optional[CodeChunk]:
        if isinstance(node, ast.stmt):
            # and any(isinstance(node, type_) for type_ in self.NODE_TYPES_TO_CHUNK):
            return self.make_code_chunk(
                lineno=coalesce(node.lineno, 0),
                end_lineno=coalesce(node.end_lineno, 0),
                col_offset=coalesce(node.col_offset, 0),
                end_col_offset=coalesce(node.end_col_offset, 0)
            )
        else:
            return None

    def make_code_chunk(self, lineno: int, end_lineno: int, col_offset: int, end_col_offset: int,
                        token_count: Optional[int] = None, peer_group: Optional[UUID] = None) -> CodeChunk:
        lines = self._code_lines[lineno - 1:end_lineno]
        if indent_match := re.search(r'^\s+', lines[0]):
            indent_len = len(indent_match.group(0))
        else:
            indent_len = 0
        lines[-1] = lines[-1][:end_col_offset]
        lines = [
            lines[0][col_offset:],
            *(line[indent_len:] for line in lines[1:])
        ]

        code = '\n'.join(lines) + '\n'
        if token_count is None:
            token_count_ = self.get_token_count(code)
        else:
            token_count_ = token_count

        if peer_group is None:
            peer_group_ = self._current_peer_group
        else:
            peer_group_ = peer_group

        return CodeChunk(
            file=self.file,
            lineno=lineno,
            end_lineno=end_lineno,
            col_offset=col_offset,
            end_col_offset=end_col_offset,
            code=code,
            peer_group=peer_group_,
            token_count=token_count_
        )

    def _get_children(self, node: ast.AST, of_type: Type[ast.AST] = ast.AST) -> Iterable[ast.AST]:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, of_type):
                        yield item
            elif isinstance(value, of_type):
                yield value

    def _get_stmt_header(self, node: ast.stmt) -> Tuple[Optional[CodeChunk], List[ast.stmt]]:
        lineno = node.lineno
        col_offset = node.col_offset
        sub_stmts = cast(List[ast.stmt], list(self._get_children(node, of_type=ast.stmt)))
        if len(sub_stmts) > 0:
            end_lineno, end_col_offset = min((n.lineno, n.col_offset) for n in sub_stmts)
            assert (lineno, col_offset) <= (end_lineno, end_col_offset)

            return_chunk = self.make_code_chunk(
                lineno=lineno,
                col_offset=col_offset,
                end_lineno=end_lineno,
                end_col_offset=end_col_offset,
                peer_group=uuid4()
            )
        else:
            return_chunk = None

        return return_chunk, sub_stmts

    def generic_visit(self, node) -> Any:
        """
        Note: this should probably not use NodeVisitor anymore, and then also pass values like peer_group
        in a more functional manner.  TODO
        """
        chunk = self.chunk_from_node(node)
        new_peer_group: Optional[UUID]
        children_to_visit: List[ast.AST]
        if chunk is not None and chunk.token_count <= self.max_chunk_size:
            if self._chunks_by_peer_group \
                    and self._chunks_by_peer_group[-1][0].peer_group == chunk.peer_group:
                peer_group = self._chunks_by_peer_group[-1]
            else:
                peer_group = []
                self._chunks_by_peer_group.append(peer_group)
            peer_group.append(chunk)
            children_to_visit = []
            new_peer_group = uuid4()
        elif chunk and isinstance(node, ast.stmt):
            # add chunk to child peer group
            header_chunk, children_to_visit_ = self._get_stmt_header(node)
            children_to_visit = cast(List[ast.AST], children_to_visit_)
            if header_chunk is not None:
                self._chunks_by_peer_group.append([header_chunk])
                new_peer_group = header_chunk.peer_group
            else:
                # might be too big, will determine to warn or error in .chunk_up_peer_group()
                chunk.peer_group = uuid4()
                self._chunks_by_peer_group.append([chunk])
                children_to_visit = []
                new_peer_group = None
            # chunk = self.collapse_chunk(chunk)
        elif chunk:
            raise AssertionError('This shouldn\'t happen, if chunk is not None then it is a stmt')
        else:
            children_to_visit = list(self._get_children(node))
            new_peer_group = uuid4()

        if len(children_to_visit) > 0:
            assert new_peer_group is not None, 'If we get here, new_peer_group should have been set'
            old_peer_group = self._current_peer_group
            try:
                self._current_peer_group = new_peer_group
                super().generic_visit(node)
            finally:
                self._current_peer_group = old_peer_group

    def combine_from_to_chunks(self, chunk_a: CodeChunk, chunk_b: CodeChunk) -> CodeChunk:
        """
        Assumes the code chunks are consecutive.  If they aren't, it will capture the code in-between
        """
        assert chunk_a.end_lineno <= chunk_b.lineno, 'chunk_a should be before chunk_b in the code'
        # remake the chunk from the linenos and offsets so we get the spaces between
        return self.make_code_chunk(
            lineno=chunk_a.lineno,
            end_lineno=chunk_b.end_lineno,
            col_offset=chunk_a.col_offset,
            end_col_offset=chunk_b.end_col_offset
        )


class FindBugsReturn(NamedTuple):
    is_bug: bool
    description: str


class BugFinder:
    def __init__(self, open_ai_client: OpenAiClient, is_bug_re: Optional[re.Pattern] = None,
                 system_content: str = FIND_BUGS_SYSTEM_CONTENT):
        self.open_ai_client = open_ai_client
        self.is_bug_re = is_bug_re if is_bug_re is not None else re.compile(DEFAULT_IS_BUG_RE)
        self.system_content = system_content

    def get_query_messages(self, code: str) -> List[Message]:
        return [
            Message(Role.system, self.system_content),
            Message(Role.user, code),
        ]

    def find_bugs(self, code: str, refresh_cache: bool = False) -> FindBugsReturn:
        query_messages = self.get_query_messages(code)
        description = self.open_ai_client.query_messages(query_messages, refresh_cache=refresh_cache)
        is_bug = bool(self.is_bug_re.search(description))

        return FindBugsReturn(is_bug, description)


class QueryConstructor:
    def __init__(self, open_ai_client: OpenAiClient, examples: List[Example], max_tokens_to_send: int,
                 system_content: str = FIND_BUGS_SYSTEM_CONTENT, model: str = DEFAULT_MODEL):
        self.open_ai_client = open_ai_client
        self.max_tokens_to_send = max_tokens_to_send
        self.system_content = system_content
        self.model = model  # for getting token count
        self._token_count_cache: CacheProtocol[str, int] = {}
        self.examples = examples

    def get_token_count(self, code: str, refresh_cache: bool = False) -> int:
        """Return the number of tokens in a string."""
        if refresh_cache or code not in self._token_count_cache:
            encoding = tiktoken.encoding_for_model(self.model)
            self._token_count_cache[code] = len(encoding.encode(code))
        return self._token_count_cache[code]

    def _get_token_count_sum(self, messages: List[Message]) -> int:
        return sum(self.get_token_count(m.content) for m in messages)

    def _get_starting_messages(self, query: str) -> List[Message]:
        return [
            Message(role=Role.system, content=self.system_content),
            Message(role=Role.user, content=query),
        ]

    def add_examples_to_query(self, query: str) -> List[Message]:
        filter_examples = self.will_filter_examples(query)

        if filter_examples:
            return self._add_examples_filtered(query)
        else:
            return self._add_examples_all(query)

    def will_filter_examples(self, query: str) -> bool:
        starting_messages = self._get_starting_messages(query)
        token_count = self._get_token_count_sum(starting_messages)
        for example in self.examples:
            token_count += self.get_token_count(example.code)
            token_count += self.get_token_count(example.response)
            if token_count > self.max_tokens_to_send:
                filter_examples = True
                break
        else:
            filter_examples = False
        return filter_examples

    def _add_examples_all(self, query: str) -> List[Message]:
        starting_messages = self._get_starting_messages(query)
        return_messages = starting_messages[:-1]
        for example in self.examples:
            return_messages.append(Message(
                role=Role.user,
                content=example.code,
            ))
            return_messages.append(Message(
                role=Role.agent,
                content=example.response
            ))
        return_messages.append(starting_messages[-1])
        return return_messages

    @staticmethod
    def _sorted(to_sort: Iterable[T], key: Callable[[T], Any]) -> Iterable[T]:
        to_sort_keyed = map(lambda x: (key(x), x), to_sort)
        sorted_keyed = sorted(to_sort_keyed, key=lambda x: x[0])
        yield from map(lambda x: x[1], sorted_keyed)

    def _add_examples_filtered(self, query: str) -> List[Message]:
        starting_messages = self._get_starting_messages(query)
        texts_iter = itertools.chain((e.code for e in self.examples), [query])
        embeddings = self.open_ai_client.get_embeddings(texts=texts_iter)  # this should be cached
        embeddings_by_text = {text: embeddings for text, embeddings in embeddings}

        query_embeddings = embeddings_by_text[query]

        def _rank(example_: Example):
            return _cosine_wrapper(query_embeddings, embeddings_by_text[example_.code])
        sorted_examples = self._sorted(self.examples, key=_rank)

        return_messages = starting_messages
        token_count = self._get_token_count_sum(starting_messages)
        for example in sorted_examples:
            token_count += self.get_token_count(example.code) + self.get_token_count(example.response)
            if token_count > self.max_tokens_to_send:
                return return_messages  # return examples without latest
            return_messages = [
                *return_messages[:-1],
                Message(role=Role.user, content=example.code),
                Message(role=Role.agent, content=example.response),
                return_messages[-1]
            ]
        return return_messages  # we shouldn't get here, but :shrug:
