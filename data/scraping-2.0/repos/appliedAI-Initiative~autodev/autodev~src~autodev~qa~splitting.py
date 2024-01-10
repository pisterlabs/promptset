import ast

from langchain.text_splitter import TextSplitter as lcTextSplitter


def get_ast_chunks(
    module_code: str,
) -> list[str]:
    module_ast = ast.parse(module_code)
    unparsed_nodes = []
    for node in module_ast.body:
        unparsed_node = ast.unparse(node)
        # fixing issues with multiline strings
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            unparsed_node = (
                unparsed_node.replace("'\\n", '"""\n')
                .replace("\\n'", '\n"""')
                .replace("\\n", "\n")
            )
        # adding newlines for functions and classes for simplified aggregation
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
            unparsed_node = "\n" + unparsed_node
        unparsed_nodes.append(unparsed_node)
    return unparsed_nodes


def split_into_semantic_chunks(
    module_code: str, num_lines: int = 350, max_lines: int = 500
) -> list[str]:
    """
    Splits a string containing module code string into a list of semantic
    chunks by aggregating adjacent AST nodes.
    Docstrings will be included, however, block comments will be excluded
    (as they are not parsed by AST).

    :param module_code: The module code to be split into semantic chunks.
    :param num_lines: The desired approximate number of lines in each chunk.
        All but the last chunk will have at least `num_lines` lines.
    :param max_lines: The maximum number of lines that a chunk can have.
        If an AST node has more than `max_lines` lines, a `ValueError` is raised.

    :returns: A list of semantic chunks produced from the input module code.
    """

    def n_lines(chunk):
        return len(chunk.split("\n"))

    module_chunks = get_ast_chunks(module_code)
    result = []

    cur_aggregated_chunk = ""
    cur_aggregates_lines = 0
    for chunk in module_chunks:
        chunk_lines = n_lines(chunk)
        if chunk_lines > max_lines:
            raise ValueError(
                f"Chunk starting with: {chunk[:100]} has too many lines: {chunk_lines}"
            )

        if cur_aggregates_lines + chunk_lines > max_lines:
            # add the current aggregated chunk to the result and start a new
            # aggregation from the current chunk
            result.append(cur_aggregated_chunk)
            cur_aggregated_chunk = chunk
            cur_aggregates_lines = chunk_lines
            continue

        if cur_aggregated_chunk:
            cur_aggregated_chunk += "\n"
            cur_aggregates_lines += 1

        cur_aggregated_chunk += chunk
        cur_aggregates_lines += chunk_lines
        if cur_aggregates_lines >= num_lines:
            result.append(cur_aggregated_chunk.rstrip("\n"))
            cur_aggregated_chunk = ""
            cur_aggregates_lines = 0
            continue

    # the last chunk
    cur_aggregated_chunk = cur_aggregated_chunk.rstrip("\n")
    if cur_aggregated_chunk:
        result.append(cur_aggregated_chunk)

    return result


class PythonAstSplitter(lcTextSplitter):
    """
    A splitter that splits a Python module into chunks by aggregating adjacent AST nodes.
    Contrary to other splitters, the chunks are not determined by the number of symbols,
    but instead by the number of lines.

    Also, **importantly**, the resulting chunks will not include block comments,
    since these are not parsed by AST. Depending on the use case, this may or may not be desirable.

    If comments are important, you can use the `langchain.text_splitter.PythonCodeTextSplitter` instead.
    """

    def __init__(self, num_lines=300, max_lines=600) -> None:
        super().__init__()
        self.num_lines = num_lines
        self.max_lines = max_lines

    def split_text(self, text: str) -> list[str]:
        return split_into_semantic_chunks(text, self.num_lines, self.max_lines)
