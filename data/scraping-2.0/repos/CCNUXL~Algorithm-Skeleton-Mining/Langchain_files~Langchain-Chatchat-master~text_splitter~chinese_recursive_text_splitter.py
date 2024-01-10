import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # text：要分割的文本字符串。
    # separator：用于分割文本的分隔符字符串。
    # keep_separator：一个布尔值，确定是否保留分隔符在分割后的结果中。
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            # 这里采用了正则表达式中的捕获分组，以便在分割后保留分隔符。
            # 通过将 _splits 列表中的奇数索引和偶数索引的元素组合在一起实现
            # 将分割后的结果重新组合，将分隔符与相邻的子字符串合并成一个字符串
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            # 如果 _splits 的长度是奇数，说明最后一个分隔符没有相应的子字符串，将其添加到结果列表中。
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
        keep_separator=True, # 保留分隔符
        is_separator_regex=True, # 分隔符是否正则表达式
        chunk_size=200,  # 每个块的最大长度
        chunk_overlap=0
    )
    ls = [
        """
        标题
        全球可再生能源趋势

        简介
        近年来，全球能源格局正在发生重大变革。随着对气候变化和化石燃料有限性的担忧不断增长，世界正在将焦点转向可再生能源。这份简要报告旨在提供当前全球可再生能源趋势的概述。

        关键点

        太阳能迅猛增长： 太阳能在过去十年中取得了显著增长。成本下降，太阳能电池板效率提高，政府激励措施都促进了这一增长。

        风能扩张： 风能是另一个有前景的领域。离岸风电场越来越普及，风力涡轮机变得更加高效和具有成本效益。

        能源储存解决方案： 鉴于可再生能源如太阳能和风能的不确定性，能源储存解决方案，如先进的电池，对于电网的稳定性和可靠性至关重要。

        新兴技术： 在潮汐和地热能源等领域的研究和开发正在为清洁能源发电开辟新的可能性。

        政府政策： 许多国家的政府正在实施促进可再生能源的政策，包括补贴、税收激励措施和减排目标。

        挑战

        间歇性： 太阳能和风能等可再生能源的不可预测性为持续供能带来了挑战。

        基础设施投资： 转向可再生能源需要大量的基础设施投资，包括电网升级和新的能源储存设施。

        公众认知： 说服公众可再生能源的益处和可行性至关重要。

        结论
        全球转向可再生能源是在应对气候变化方面的一个令人鼓舞的趋势。然而，在将可再生能源整合到现有能源基础设施方面仍然存在挑战。持续的研究、投资和公众支持对于实现可持续能源未来至关重要。
        """,
        ]
    
    # text = """"""
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        num = 1
        for chunk in chunks:
            print("chunk_num_", num, end=" ")
            print(chunk)
            num += 1
