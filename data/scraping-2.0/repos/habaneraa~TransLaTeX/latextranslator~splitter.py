
from .tokenizer import tiktoken_length
from typing import List, Iterable, Callable


latex_macro_separators = [
    # First, try to split along Latex sections
    "\n\\chapter{",
    "\n\\section{",
    "\n\\subsection{",
    "\n\\subsubsection{",
    # split by environments
    "\n\\begin{enumerate}",
    "\n\\begin{itemize}",
    "\n\\begin{description}",
    "\n\\begin{list}",
    "\n\\begin{quote}",
    "\n\\begin{quotation}",
    "\n\\begin{verse}",
    "\n\\begin{verbatim}",
    # split by math environments
    "\n\\begin{align}",
    "$$",
    # split by the paragraphs
    "\n\n",
    # split by new lines
    "\n",
    # gg
    " ",
]

def _split(text: str, separator) -> List[str]:
    splits = text.split(separator)
    if len(splits) == 1:
        return splits
    if len(splits) > 1:
        new_split = [splits[0]]
        for split in splits[1:]:
            new_split.append(separator + split)
        splits = list(filter(lambda x: x != '', new_split))
        return splits


class LatexSourceSplitter:
    """A non-overlapping LaTeX text splitter that preserves all delimiters.
    This splitter is rewritten from LangChain.text_splitter.LatexTextSplitter """
    
    def __init__(
        self,
        chunk_size: int = 1500,
        length_function: Callable[[str], int] = tiktoken_length,
        separators: List[str] = None,
    ):
        self._chunk_size = chunk_size
        self._length_function = length_function
        self._separators = separators if separators else latex_macro_separators
    
    def merge_splits(self, splits: Iterable[str]) -> List[str]:
        # We now want to combine these smaller pieces into medium size chunks
        merged_splits = []
        current_chunk = []
        current_len = 0
        for split in splits:
            new_len = self._length_function(split)
            if new_len + current_len > self._chunk_size:
                merged_splits.append(''.join(current_chunk))
                current_chunk = [split]
                current_len = new_len
            else:
                current_chunk.append(split)
                current_len += new_len
        merged_splits.append(''.join(current_chunk))
        # assert ''.join(splits) == ''.join(merged_splits)
        return merged_splits

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks. Retain all separators."""
        final_chunks = []
        # Get appropriate separator to use
        separator = self._separators[-1]
        for _s in self._separators:
            if _s in text:
                try_split = _split(text, _s)
                if len(try_split) > 1:
                    separator = _s
                    break
        splits = try_split
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self.merge_splits(_good_splits)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                other_info = self.split_text(s)
                final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self.merge_splits(_good_splits)
            final_chunks.extend(merged_text)
        return final_chunks


def test_splitter(test_str):
    splitter = LatexSourceSplitter(chunk_size=500)
    splits = splitter.split_text(test_str)
    # should not change any text
    assert ''.join(splits) == test_str
