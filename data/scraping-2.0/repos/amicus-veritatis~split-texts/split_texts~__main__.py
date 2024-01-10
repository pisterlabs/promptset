import sys
from langchain.text_splitter import TextSplitter
import math
import json
import argparse
tokenizer = "cl100k_base"

# Constants
CHUNK_SIZE = 3000  # The target size of each text chunk in tokens
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = 128  # The number of embeddings to request at a time
MAX_NUM_CHUNKS = 10000000  # The maximum number of chunks to generate from a text
AVERAGE_CHARS_IN_TOKEN = 5 # in English.
DESIRED_MIN_TOKEN_FRACTION = 0.35


def min_chunk_size_chars(
    chunk_size: int,
    desired_min_token_fraction: float,
    avg_chars_per_token: int
) -> int:
    """min_chunk_size_chars Calculate the minimum number of characters in a chunk.
    
    :param chunk_size: The target size of each text chunk in tokens
    :type chunk_size: int
    :param desired_min_token_fraction: The desired minimum fraction of tokens in a chunk
    :type desired_min_token_fraction: float
    :param avg_chars_per_token: The average number of characters per token (e.g., in English)
    :type avg_chars_per_token: int
    :return: The minimum number of characters in a chunk
    :rtype: int
    """
    return math.ceil(chunk_size / desired_min_token_fraction) * avg_chars_per_token

class ImprovedTokenTextSplitter(TextSplitter):
    def __init__(
        self,
        encoding_name: str = tokenizer,
        allowed_special = set(),
        disallowed_special = "all",
        desired_min_token_fraction: float = DESIRED_MIN_TOKEN_FRACTION,
        min_chunk_length_to_embed: int = MIN_CHUNK_LENGTH_TO_EMBED,
        avg_chars_per_token: int = AVERAGE_CHARS_IN_TOKEN,
        **kwargs,
    ):
        """ImprovedTokenTextSplitter Initialize the ImprovedTokenTextSplitter with the given parameters.
        
        :param encoding_name: The tokenizer's encoding name, defaults to tokenizer
        :type encoding_name: str, optional
        :param allowed_special The set of special tokens allowed, defaults to an empty set.
        :type allowed_special: Union[Literal["all"], AbstractSet[str]], Optional
        :param disallowed_special The set of special tokens disallowed, defaults to "all".
        :type disallowed_special: Union[Literal["all"], Collection[str]], Optional
        :param desired_min_token_fraction: The desired minimum fraction of tokens in a chunk, defaults to DESIRED_MIN_TOKEN_FRACTION
        :type desired_min_token_fraction: float, optional
        :param min_chunk_length_to_embed: The minimum length of chunks to be included in the final list, defaults to MIN_CHUNK_LENGTH_TO_EMBED
        :type min_chunk_length_to_embed: int, optional
        :param avg_chars_per_token: The average number of characters per token (e.g., in English), defaults to AVERAGE_CHARS_IN_TOKEN
        :type avg_chars_per_token: int, optional
        :param kwargs: Additional keyword arguments to pass to the parent class
        :type kwargs: Any, optional
        """
        super().__init__(**kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special
        self._desired_min_token_fraction = desired_min_token_fraction
        self._min_chunk_length_to_embed = min_chunk_length_to_embed
        self._avg_chars_per_token = avg_chars_per_token
    
    def split_text(self, text:str):
        """get_text_chunks Split a text into chunks of ~CHUNK_SIZE tokens, based on punctuation and newline boundaries.
        :param text: The text to split into chunks.
        :type text: str
        :return: A list of text chunks, each of which is a string of ~CHUNK_SIZE tokens.
        :rtype: List[str]
        """
        marks = [".", "?", "!", "\n"]
        # Return an empty list if the text is empty or whitespace
        if not text or text.isspace():
            return []
        
        # Tokenize the text
        tokens = self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )
        
        # Initialize an empty list of chunks 
        chunks = []
        
        # Use self. None? not my problem
        chunk_size = self._chunk_size
        chunk_overlap = self._chunk_overlap
        
        # Calculate min_chunk_size
        chunk_min = min_chunk_size_chars(
            chunk_size=chunk_size,
            desired_min_token_fraction=self._desired_min_token_fraction,
            avg_chars_per_token=self._avg_chars_per_token
        )
        
        # Same as num_tokens in below code.
        start_idx = 0 
        
        """
        quite simple if you think.
        startidx + chunk_size => chunk_size for obvious reason.
        """
        cur_idx = min(chunk_size, len(tokens))
        
        """
        `num_chunks++` === `start_idx += self._chunk_size - self._chunk_overlap` 
        prevent inf loop
        """
        idx_step  = max(chunk_size - chunk_overlap, 1)
        
        # num_chunks => remove
        while start_idx < len(tokens):
            # chunk_ids in split_text, chunk in below
            chunk = tokens[start_idx:cur_idx]
            
            # Decode the chunk (start_idx ~ cur)
            chunk_text = self._tokenizer.decode(chunk)
            
            start_idx += idx_step
            
            cur_idx = min(start_idx + chunk_size, len(tokens))
            
            # Skip the chunk if it is empty or whitespace
            if not chunk_text or chunk_text.isspace():
                continue
            
            # Find the last period or punctuation mark in the chunk
            last_punctuation = max(
                [chunk_text.rfind(x) for x in marks]
            )

            # If there is a punctuation mark, and the last punctuation index is before MIN_CHUNK_SIZE_CHARS
            if last_punctuation != -1 and last_punctuation > chunk_min:
                # Truncate the chunk text at the punctuation mark
                chunk_text = chunk_text[: last_punctuation + 1]

            # Remove any newline characters and strip any leading or trailing whitespace
            chunk_text_to_append = chunk_text.replace("\n", " ").strip()

            if len(chunk_text_to_append) > self._min_chunk_length_to_embed:
                # Append the chunk text to the list of chunks
                chunks.append(chunk_text_to_append)
                
        return chunks

def main():
    parser = argparse.ArgumentParser(description="Split large texts into smaller chunks with JSON output for easy processing.")
    parser.add_argument("-s", "--chunk-size", type=int, default=3000, help="The target size of each text chunk in tokens.")
    parser.add_argument("-f", "--min-token-fraction", type=float, default=0.35, help="The desired minimum fraction of tokens in a chunk.")
    parser.add_argument("-c", "--avg-chars-per-token", type=int, default=5, help="The average number of characters per token (e.g., in English).")
    parser.add_argument("-l", "--min-chunk-length", type=int, default=5, help="Discard chunks shorter than this.")
    parser.add_argument("-o", "--overlap", type=int, default=0, help="The number of tokens to overlap between chunks.")
    parser.add_argument("-e", "--encoding", type=str, default="cl100k_base", help="The tokenizer's encoding name.")
    args = parser.parse_args()


    splitter = ImprovedTokenTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        desired_min_token_fraction=args.desired_min_token_fraction,
        min_chunk_length_to_embed=args.min_chunk_length_to_embed,
        avg_chars_per_token=args.avg_chars_per_token,
        encoding_name=args.encoding_name,
    )
    texts = sys.stdin.read()
    split_texts = splitter.split_text(texts)
    result = []

    for text in split_texts:
        result.append(text.strip())

    # Write the processed lines to stdout in JSON format
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

