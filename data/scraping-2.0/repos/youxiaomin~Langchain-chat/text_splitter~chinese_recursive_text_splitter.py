import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            #print(_splits)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            chunk_size: int = 100,
            chunk_overlap: int = 40,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._strip_whitespace = True

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs


    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        #print("_separator:",_separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

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
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]


if __name__ == "__main__":
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=100,
        chunk_overlap=40
    )
    ls = ["""盛阳出生于普通家庭，自小爱画画。盛阳在学生时代，因为简冰一支公益广告，对广告行业产生了极大的兴趣.凭着这份热爱，他努力吸收专业知识，进入一家广告公司担任设计师。简冰从小自强，独立，是广告导演。简冰和前夫薛义明的婚姻生活并不好，面对前夫的背叛，她选择离婚，也就是这一天，她遇见了盛阳，此时的她还不知道这个大男孩即将成为她生命中的骄阳。薛易明是业内知名的广告导演。与简冰本是业内模范夫妻。简冰因无法忍受薛易明出轨，与他离婚。梁珊珊是简冰的至交好友，也是工作上的好搭档。她关心简冰的生活，怒斥薛易明的生活作风。宋晨与盛阳毕业于同一个大学，就职于同一个广告公司，也是好朋友。宋晨重情重义，十分了解盛阳，帮他挡下潘柔的追求。盛向前是盛阳的父亲，盛向前的职业是一名环卫工人。罗美娟是盛阳的母亲，剧中罗美娟也叫罗女士，大美娟。罗美娟职业是超市的收银员。郝俊杰是一名有着音乐梦想的冰球爱好者，也是简冰的妹妹简霜的官配。郝俊杰是吴幸健饰演的角色。在剧中郝俊杰的舅舅希望他出国实现音乐梦想，但郝俊杰跟简霜一样，都有点叛逆。方沐喜欢简冰，方沐的外甥跟简冰的妹妹一起打冰球，方沐因此认识了简冰。方沐对简冰姐妹二人的初印象并不好，觉得她们不是一般人，还让外甥要远离她们。但方沐在知道简冰离婚了之后，他开始喜欢上了简冰，并成了简冰身边最忠实的追求者。潘柔是广告公司实习生，盛阳的小师妹。潘柔是敢爱敢恨的白富美，喜欢盛阳，但是盛阳对她仅仅是当做小妹妹一样照顾。反而是潘柔的好哥们宋晨，一直在背后喜欢她。"""]
    # text = """"""
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
