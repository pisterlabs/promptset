from typing import List, Callable, Any
from langchain.text_splitter import TextSplitter
from nltk.tokenize import sent_tokenize

# TODO: Fix this.
# sent_tokenize has strange behavior sometimes: 'The units could be anything (characters, words, sentences, etc.), depending on how you want to chunk your text.'
# splits into ['The units could be anything (characters, words, sentences, etc.', '), depending on how you want to chunk your text.']

StrToIntFunction = Callable[[str], int]
StrIntBoolToStrFunction = Callable[[str, int, bool], str]


def default_truncate_function(string: str, length: int, from_end: bool = False) -> str:
    return string[-length:] if from_end else string[:length]


class ParagraphSentenceUnitTextSplitter(TextSplitter):
    """A custom TextSplitter that breaks text by paragraphs, sentences, and then units (chars/words/tokens/etc).

    @param min_chunk_size: The minimum number of units in a chunk.
    @param max_chunk_size: The maximum number of units in a chunk.
    @param length_function: A function that returns the length of a string in units. Defaults to len().
    @param truncate_function: A function that truncates a string to a given unit length.
    """

    DEFAULT_MIN_CHUNK_SIZE: int = 900
    DEFAULT_MAX_CHUNK_SIZE: int = 1100
    DEFAULT_LENGTH_FUNCTION: StrToIntFunction = len
    DEFAULT_TRUNCATE_FUNCTION: StrIntBoolToStrFunction = default_truncate_function

    def __init__(
        self,
        min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        length_function: StrToIntFunction = DEFAULT_LENGTH_FUNCTION,
        truncate_function: StrIntBoolToStrFunction = DEFAULT_TRUNCATE_FUNCTION,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        self._length_function = length_function
        self._truncate_function = truncate_function

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks of length between min_chunk_size and max_chunk_size."""
        if not text:
            return []

        blocks: List[str] = []
        current_block: str = ""

        paragraphs = text.split("\n\n")
        for paragraph in paragraphs:
            current_block += "\n\n" + paragraph
            block_length = self._length_function(current_block)

            if block_length > self.max_chunk_size:
                current_block = self._handle_large_paragraph(current_block, blocks, paragraph)
            elif block_length >= self.min_chunk_size:
                blocks.append(current_block)
                current_block = ""
            else:  # current block is too small, continue appending to it
                continue

        blocks = self._handle_remaining_text(current_block, blocks)
        return [block.strip() for block in blocks]

    def _handle_large_paragraph(self, current_block: str, blocks: List[str], paragraph: str) -> str:
        # Undo adding the whole paragraph
        offset = len(paragraph) + 2  # +2 accounts for "\n\n"
        current_block = current_block[:-offset]

        sentences = sent_tokenize(paragraph)
        for sentence in sentences:
            current_block += f" {sentence}"

            block_length = self._length_function(current_block)
            if block_length < self.min_chunk_size:
                continue
            elif block_length <= self.max_chunk_size:
                blocks.append(current_block)
                current_block = ""
            else:
                current_block = self._truncate_large_block(current_block, blocks)
        return current_block

    def _truncate_large_block(self, current_block: str, blocks: List[str]) -> str:
        while self._length_function(current_block) > self.max_chunk_size:
            # Truncate current_block to max size, set remaining text as current_block
            truncated_block = self._truncate_function(current_block, self.max_chunk_size, False)
            blocks.append(truncated_block)

            current_block = current_block[len(truncated_block) :].lstrip()

        return current_block

    def _handle_remaining_text(self, last_block: str, blocks: List[str]) -> List[str]:
        if blocks == []:  # no blocks were added
            return [last_block]
        elif last_block:  # any leftover text
            len_last_block = self._length_function(last_block)
            if self.min_chunk_size - len_last_block > 0:
                # Add text from previous block to last block if last_block is too short
                part_prev_block = self._truncate_function(
                    blocks[-1], self.min_chunk_size - len_last_block, True
                )
                last_block = part_prev_block + last_block

            blocks.append(last_block)

        return blocks
