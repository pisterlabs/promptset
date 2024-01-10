from abc import ABC, abstractmethod

from langchain.text_splitter import TokenTextSplitter
from sentence_splitter import SentenceSplitter as SentenceSplitter_


class Splitter(ABC):
    @abstractmethod
    def split(self, text: str) -> list["str"]:
        pass


class TokenSplitter(Splitter):
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 0) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def split(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)


class SentenceSplitter(Splitter):
    def __init__(self) -> None:
        super().__init__()
        self.text_splitter = SentenceSplitter_(language="en")

    def split(self, text: str) -> list[str]:
        output = self.text_splitter.split(text)
        # drop empty string from list
        output = [x for x in output if x != ""]
        return output


class ParagarphSplitter(Splitter):
    def __init__(self) -> None:
        super().__init__()
        self.text_splitter = SentenceSplitter_(language="en")

    def split(self, text: str) -> list[str]:
        # split to sentences
        output = self.text_splitter.split(text)
        output = [x for x in output if x != ""]  # drop empty strings
        output = self.join_sentences(output)
        return output

    def join_sentences(self, sentences: list[str]) -> list[str]:
        """
        takes a list of sentence and returns a list of paragraphs
        """
        result = []
        current_list = []

        for sentence in sentences:
            current_list.append(sentence)

            if sentence.endswith("."):
                result.append(current_list)
                current_list = []

        if current_list:
            result.append(current_list)

        result = ["".join(para) for para in result]
        return result


if __name__ == "__main__":
    output = ParagarphSplitter().split(
        """
    Once upon a time, in the small town of Willowbrook, there lived a curious young girl named Emily. 
    With her bright eyes and an insatiable thirst for adventure, she was always seeking new experiences and discoveries.
    One sunny morning, Emily ventured into the dense forest that surrounded her town. 
    She had heard tales of a hidden waterfall deep within the woods, and her heart was set on finding it. 
    Armed with her trusty backpack and a sense of determination, she followed a faint trail leading her deeper into the wilderness.
    As she walked, Emily noticed the forest changing around her. The air grew cooler, 
    and beams of sunlight filtered through the thick canopy of leaves, creating a magical ambiance. 
    She could hear the soft chirping of birds and the rustling of leaves under her feet. 
    It felt as though nature itself was guiding her towards her destination.
    After hours of hiking, Emily's perseverance paid off. She stumbled upon a hidden clearing bathed in golden sunlight. 
    In the center stood a majestic waterfall, 
    cascading down from the rocks above into a crystal-clear pool below. The sight was breathtaking, 
    and Emily couldn't help but be in awe of its beauty.
    """
    )
    print(output)
