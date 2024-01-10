#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : AI.  @by PyCharm
# @File         : text_split_with_overlap
# @Time         : 2023/4/25 15:33
# @Author       : betterme
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  : 

from meutils.pipe import *
from meutils.ai_nlp.SplitSentence import split_sentence


def merge_short_sentences(sentences, chunk_size):
    short_sentences = []
    for i, sentence in enumerate(sentences):  # 句子太长 需要再切
        short_sentences.append(sentence)
        if len(''.join(short_sentences)) > chunk_size:
            return sentences[:i], sentences[i:]
    return sentences, []


def textsplitter_with_overlap(sentences, chunk_size=512, chunk_overlap_rate=0.2):
    """

    :param sentences: sentences = split_sentence(text)
    :param chunk_size:
    :param chunk_overlap_rate:
    :return:
    """
    chunk_overlap = int(chunk_size * chunk_overlap_rate)

    result = []
    while sentences:
        merge_sentences, sentences = merge_short_sentences(sentences, chunk_size)
        # result.append(merge_sentences)
        result.append(''.join(merge_sentences).split() | xjoin)

        if not sentences:
            break

        overlap_sentences = merge_short_sentences(merge_sentences[::-1], chunk_overlap)[0][::-1]

        if len(''.join(overlap_sentences)) + len(sentences[0]) > chunk_size:  # 丢弃重叠部分
            continue

        sentences = overlap_sentences + sentences  # 新句子集合
    return result



if __name__ == '__main__':
    text = '央视新闻消息，近日，特朗普老友皮尔斯·摩根喊话特朗普：“美国人的生命比你的选举更重要。如果你继续以自己为中心，继续玩弄愚蠢的政治……如果你意识不到自己>的错误，你就做不对”。目前，特朗普已“取关”了这位老友。'
    sentences = split_sentence(text, criterion='coarse')
    print(sentences)
    print(textsplitter_with_overlap(sentences))

    from langchain.text_splitter import CharacterTextSplitter

    print(CharacterTextSplitter().split_text(text))
