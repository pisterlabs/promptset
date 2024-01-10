from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from pathlib import Path

from typing import List
import re
from config import cfg

from log_factory import logger


def load_txt(file_path: Path) -> List[Document]:
    """
    Use the csv loader to load the CSV content as a list of documents.
    :param file_path: A CSV file path
    :return: the document list after extracting and splitting all CSV records.
    """
    loader = TextLoader(file_path=str(file_path), encoding="utf-8")
    doc_list: List[Document] = loader.load()
    logger.info(f"Length of CSV list: {len(doc_list)}")
    for doc in doc_list:
        doc.page_content = re.sub(r"\n+", "\n", doc.page_content, re.MULTILINE)

    return custom_splitter(doc_list)


def split_docs(doc_list: List[Document]) -> List[Document]:
    """
    Splits the documents in smaller chunks from a list of documents.
    :param doc_list: A list of documents.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separator=cfg.chunk_separator,
    )
    texts: List[Document] = text_splitter.split_documents(doc_list)
    logger.info(f"Length of texts: {len(texts)}")
    return texts


def custom_splitter(doc_list: List[Document], part_count: int = 3) -> List[Document]:
    """
    Splits in 'part_count' parts.
    """
    res: List[Document] = []
    for doc in doc_list:
        content = doc.page_content
        i = 0
        character_size = len(doc.page_content) // part_count - 1
        part_counter = 0
        while True:
            i += character_size
            temp = content[i - character_size : i]
            if part_counter == part_count or i > len(content):
                break
            for j in reversed(range(len(temp))):
                if temp[j] in [".", "?", "!", ":"]:
                    j = j + 1
                    end_pos = i - character_size + j
                    part_counter += 1
                    if part_counter == part_count:
                        temp = content[i - character_size :]
                        res.append(
                            Document(page_content=temp.strip(), metadata=doc.metadata)
                        )
                    else:
                        temp = content[i - character_size : end_pos]
                        res.append(
                            Document(page_content=temp.strip(), metadata=doc.metadata)
                        )
                        i = end_pos
                    break
                elif j == 0:
                    logger.error("Could not find punctuation for split")
                    break
    return res


if __name__ == "__main__":
    text = """Om shanti. Children, you should first of all understand the self: Who am I? What am I? When you say “I”, it does not refer to the body, but the soul. Where did I, the soul, come from? Whose child am I? When the soul knows that I, the soul, am a child of the Supreme Father, the Supreme Soul, there will be happiness in remembering the Father. Children experience happiness when they know their father’s occupation. When a child is young and doesn’t know his father’s occupation, there isn’t that much happiness. As the child grows older, he comes to know his father’s occupation, and so the intoxication or happiness of that increases. So, first of all, know His occupation as to who Baba is and where He lives. When you say that the soul will merge in Him, it means that the soul becomes perishable and so who would have happiness about that? You should ask new students who come to you: What are you studying here? What status will you receive through it? Those who study in worldly colleges tell you that they are becoming a doctor or an engineer. So, you would have the faith that they are truly studying that. Here, too, students say that this is the world of sorrow, which is called hell, that is, the devil world. Against that is heaven which is called the deity world, heaven. Everyone knows this, because you can understand that this is not heaven; this is hell, that is, it is the world of sorrow. It is the world of sinful souls and that is why they call out to Him: Take us to a world of charity. The children who are studying here know that Baba is now taking them to that world. New students who come here should question the children; they should learn from the children. They can tell them the Teacher’s and the Father’s occupation. The Father would not sit and praise Himself. Would a teacher praise himself? The students would say that this teacher is like this. This is why it is said: Students reveal their master. You children have studied this course and so it is your duty to explain to the new ones who come. Would a teacher who teaches students for a BA or MA teach new students the ABC? Some students are very clever and they also teach others. The mother guru has been remembered for that. That is the first mother of the deity religion who is called Jagadamba. There is a lot of praise of the mother. In Bengal, Kali, Durga, Saraswati and Lakshmi are worshipped a lot. One should know the occupation of all those four. For example, Lakshmi is the Goddess of Wealth. She ruled here and departed. However, the names Kali, Durga etc. are given to this one. If there are four mothers (goddesses), there should also be the four consorts. Narayan of Lakshmi is very well known. Who is Kali’s consort? (Shankar). However, Shankar is shown as the consort of Parvati. Parvati is not Kali. There are many who worship Kali. They remember the goddess but they don’t know about her consort. Kali must have either a consort or the Father, but no one knows about that. You have to explain that there is just this one world, which at some point in time becomes a world of sorrow, that is, hell, and that same world then becomes Paradise or heaven in the golden age. Lakshmi and Narayan used to rule this world in the golden age. There isn’t a heaven in the subtle region where there are the subtle (forms of) Lakshmi and Narayan. Their images are here and so they surely ruled here and departed. The whole play takes place in the corporeal world. History and geography are also of this corporeal world. There is no history and geography of the subtle region. However, you have to put everything aside and first of all teach new students about Alpha and beta. Alpha is God; He is the Supreme Soul. Until they have understood this fully, their love for God won’t awaken and they won’t have that happiness, because, first of all, only when they know God can they also know His occupation and have that happiness. So, there is happiness in understanding this first aspect. God is everhappy, the Blissful One. We are His children and so why should we not have that happiness? Why isn’t there the feeling of bubbling happiness? I am a son of God, I am an everhappy master god. When there isn’t that happiness, it proves that you do not consider yourself to be a son of God. God is everhappy, but I am not happy because I do not know the Father. It is a very simple matter. Instead of listening to this knowledge, some people prefer peace because there are many who won’t be able to take this knowledge. There isn’t that much time. Even if they simply understand Alpha and stay in silence, that too is good. For instance, even sannyasis go to the mountains and sit in the caves in remembrance of God. In the same way, if you stay in remembrance of the Supreme Father, the Supreme Soul, the Supreme Light, that is also good. Even sannyasis can remain viceless by having remembrance of Him. However, they would be unable to remember Him if they remained at home: their attachment to their children etc. would pull them. This is why they renounce everything. They become holy and so there is happiness in that. Sannyasis are the best of all. Adi Dev also became a sannyasi, did he not? Just opposite the Dilwala Temple is the temple to Adi Dev where he is shown doing tapasya. In the Gita too, it says: Renounce all bodily religions. When those people go away and have renunciation they become great souls. It is wrong to call a householder a great soul. God has come and inspired you to have renunciation. One has renunciation for happiness. Great souls can never be unhappy. Some kings also renounce everything, and so they throw away their crown etc. For instance, King Gopichanda renounced everything. Therefore, there is definitely happiness in that. Achcha."""
    doc: Document = Document(page_content=text, metadata={})
    doc_list: List[Document] = custom_splitter([doc, doc], 3)
    assert len(doc_list) == 6
    print(len(doc_list))
    print(doc_list[-1].page_content)
