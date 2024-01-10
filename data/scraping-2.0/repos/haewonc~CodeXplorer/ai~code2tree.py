import re
import ast
import copy
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, _split_text_with_regex
from typing import (
    Any,
    List,
    Optional,
    Iterable
)

class Node:
    def __init__(self, i, node_type, name, code):
        self.i = i
        self.type = node_type
        self.name = name
        self.code = code
        self.source = None
        self.children = []
        self.context = None

    def add_child(self, idx):
        self.children.append(idx)

    def to_document(self):
        return Document(page_content=self.code, metadata={'idx': self.i,
            'type': self.type,
            'name': self.name,
            'children': self.children,
            'context': self.context})
    
    def to_dict(self):
        return {
            'idx': self.i,
            'type': self.type,
            'name': self.name,
            'code': self.code,
            'source': self.source,
            'children': self.children,
            'context': self.context
        }

class SymoblSplitter(RecursiveCharacterTextSplitter):

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_symbols(self, text: str) -> (List[Node], List[Document]):
        builder = CodeTreeBuilder()
        imports, nodes = builder.build_tree(text)
        texts = [Document(page_content='\n'.join(imports), metadata={'idx': None, 'type': 'import', 'name': 'import', 'children': [], 'context': None})] + \
            [node.to_document() for node in nodes if (node.type == 'function' and node.code != '__main__')]
        return nodes, texts

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> (List[Node], List[str]):
        nodes, texts = self._split_symbols(text)
        chunks = []
        for text in texts: 
            chunks.extend([Document(page_content=t, metadata=text.metadata) for t in self._split_text(text.page_content, self._separators)])
        return nodes, chunks

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> (List[Node], List[Document]):
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        nodes_all = []
        for i, text in enumerate(texts):
            index = -1
            nodes, chunks = self.split_text(text)
            for node in nodes:
                node.source = _metadatas[i]['source']
            nodes_all.extend(nodes)

            for chunk in chunks:
                metadata = copy.deepcopy(_metadatas[i])
                metadata['node_idx'] = chunk.metadata['idx']
                metadata['context'] = chunk.metadata['context']
                if self._add_start_index:
                    index = text.find(chunk.page_content, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk.page_content, metadata=metadata)
                documents.append(new_doc)
        index_mapping = {}
        for i, node in enumerate(nodes_all):
            index_mapping[(node.source, node.i)] = i
            node.i = i
        for node in nodes_all:
            new_children = []
            for child in node.children:
                new_children.append(index_mapping[(node.source, child)])
            node.children = new_children
            if node.context:
                node.context = index_mapping[(node.source, node.context)]
        for document in documents:
            if document.metadata['node_idx'] != None:
                document.metadata['node_idx'] = index_mapping[(document.metadata['source'], document.metadata['node_idx'])]
            else:
                document.metadata['node_idx'] = 'for imports'
            if document.metadata['context'] != None:
                document.metadata['context'] = '\nMethod of class ' + nodes_all[index_mapping[(document.metadata['source'], document.metadata['context'])]].name
            else:
                document.metadata['context'] = ''
        return [node.to_dict() for node in nodes_all], documents
    
    def split_documents(self, documents: Iterable[Document]) -> (List[Node], List[Document]):
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)
    
class CodeTreeBuilder(ast.NodeVisitor):
    def __init__(self):
        self.nodes = [Node(0, 'function', '__main__', '__main__')]
        self.current_node = self.nodes[0]
        self.current_class = None
        self.imports = []
        self.main = []
    
    def visit_Import(self, node):
        self.imports.append(ast.unparse(node))

    def visit_ClassDef(self, node):
        class_node = Node(len(self.nodes), 'class', node.name, ast.unparse(node))
        self.nodes.append(class_node)
        if self.current_node:
            self.current_node.add_child(len(self.nodes)-1)
        current_node, current_class = self.current_node, self.current_class
        self.current_node = class_node
        self.current_class = class_node
        self.generic_visit(node)
        self.current_node, self.current_class = current_node, current_class

    def visit_FunctionDef(self, node):
        func_node = Node(len(self.nodes), 'function', node.name, ast.unparse(node))
        self.nodes.append(func_node)
        if self.current_node:
            if self.current_class:
                func_node.context = self.current_class.i
            self.current_node.add_child(len(self.nodes)-1)
        current_node = self.current_node
        self.current_node = func_node
        for arg in node.args.args:
            if arg.arg == 'self':
                continue
            arg_name = arg.arg
            self.nodes.append(Node(len(self.nodes), 'variable', arg_name, arg_name))
            func_node.add_child(len(self.nodes)-1)
        self.generic_visit(node)
        self.current_node = current_node

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_node = Node(len(self.nodes), 'variable', target.id, ast.unparse(node))
                self.nodes.append(var_node)
                if self.current_node:
                    self.current_node.add_child(len(self.nodes)-1)
                    if self.current_node.i == 0:
                        self.main.append(ast.unparse(node))
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                var_node = Node(len(self.nodes), 'variable', ast.unparse(target).replace('self.',''), ast.unparse(node))
                self.nodes.append(var_node)
                assert self.current_class
                self.current_class.add_child(len(self.nodes)-1)

    def visit_Call(self, node):
        if self.current_node.i == 0:
            self.main.append(ast.unparse(node))

    def build_tree(self, source):
        tree = ast.parse(source)
        self.visit(tree)
        if len(self.main) > 0:
            self.nodes[0].code = '\n'.join(self.main)
        return self.imports, self.nodes
