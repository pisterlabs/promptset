# divide the subs into chunks for more accurate summarization
# TODO: divide the subs into chunks based on the topics
# summarize each chunk and add it to the markdown file
from rich.progress import track


class legacy_chunker:
    # legacy manual chunker
    def __init__(self, text):
        self.text = text

    def chunker(self, size=1000):
        words = self.text.split()
        chunks = []
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) + 1 <= size:
                current_chunk += f"{word} "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = f"{word} "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def __sizeof__(self) -> int:
        count = 0
        for _ in self.text:
            count += 1
        return count


class LangChainChunker:
    def __init__(self, text):
        self.text = text

    def chunker(self, size=1000):
        from langchain.text_splitter import CharacterTextSplitter

        # attach the duration of the video to the chunk
        # [[chunk, duration]]

        text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=size,
            chunk_overlap=0.9,
        )

        return text_splitter.split_text(self.text)

    def __sizeof__(self) -> int:
        count = 0
        for _ in self.text:
            count += 1
        return count


def ChunkByChapters(chapters: list, subs: list, size=1000):
    """Chunk the youtube video  subtitles based on the chapters

    Args:
        chapters (list): Chapters from yt api
        subs (list): subtitles from yt api
        size (int, optional): _description_. Defaults to 1000.

    Raises:
        Exception: No chapters found

    Returns:
        list : structure chunk_dict = {
              "chapter1": [
                  [chunk1, chunk2, chunk3, ...],
                  [chunk1_duration, chunk2_duration, chunk3_duration, ...]
              ],
              ...
          }
    """
    chunks = []
    chunk_dict = {}

    # format chapters for chunking
    Fchapters = [[chapter['title'], chapter['time']] for chapter in chapters]

    if len(chapters) == 0:
        raise Exception("No chapters found")
    else:

        # STEP 1:
        # chapters timestamp is set to beggining of chapter
        # to process all chapter subs instead of always checking if the sub is in the chapter
        # its easier to set the timestamp to end of chapter
        # set timestamp to last second of chapter
        for c in range(len(Fchapters) - 1):
            if c == len(Fchapters):
                break
            Fchapters[c][1] = Fchapters[c + 1][1] - 1

        # STEP 2: chunking based on chapters
        # for each chapter, chunk the subs
        # and add the chunk to the chunk_dict
        #
        #   chunk_dict = {
        #       "chapter1": [
        #           [chunk1, chunk2, chunk3, ...],
        #           [chunk1_duration, chunk2_duration, chunk3_duration, ...]
        #       ],
        #       ...
        #   }
        #

        for c in track(
            range(len(Fchapters) - 1),
            description="Chunking by chapters: "
        ):
            title = Fchapters[c][0]

            # set the start and end of the chapter
            start = 0 if c == 0 else Fchapters[c - 1][1] + 1
            end = Fchapters[c][1]

            current_chunk = ""

            # STEP 2 (a): process the subs
            # for each sub, check if it is in the chapter
            # if it is, add it to the current chunk

            for sublinedata in subs:
                cstart: int = sublinedata['start']
                subline: str = sublinedata['text']

                if cstart < start:
                    continue
                if cstart >= end:
                    break

                total_size = len(current_chunk) + len(subline)
                if total_size + 1 < size:
                    current_chunk += subline
                else:
                    chunks.append(
                        [
                            [current_chunk.strip()],
                            [cstart],
                        ]
                    )
                    current_chunk = ""

            chunk_dict.update({title: chunks})
            chunks = []

    return chunk_dict
