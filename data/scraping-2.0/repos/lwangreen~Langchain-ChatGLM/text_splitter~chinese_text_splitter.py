from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
from configs.model_config import CHUNK_SIZE


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = CHUNK_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text1(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

    # --------------------------------------------------------------------------------------------------------
    # Luming add comments 20230620
    # Langchain-ChatGLM.local_doc_qa.load_file() initialize ChineseTextSplitter or UnstructuredFileLoader
    # call loader.load() or loader.load_and_split()
    # For loader.load(), no split_text calling
    # For loader.load_and_split(textsplitter):
    # UnstructuredFileLoader extended from UnstructuredBaseLoader, the function calling goes to BaseLoader which is the parent
    # class of UnstructuredBaseLoader.load_and_split() calls textsplitter.split_text(), which is the function below.
    #
    # For TextLoader (initialized in langchian.document_loader.text.py), it goes to BaseLoader (parent class of TextLoader) and
    # calls load_and_split(), which direct back to the function below using textsplitter.split_text()
    # ----------------------------------------------------------------------------------------------------------
    def split_text_init(self, text: str) -> List[str]:   ##此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > self.sentence_size:
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > self.sentence_size:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > self.sentence_size:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                       ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        return ls


    # 经过一些优化的文本分割逻辑。yunze 2023-07-10
    def split_text(self, text: str) -> List[str]:   ##此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        # print("split_text: ", text)
        # 保留原本文本的换行符\n，更换切割分隔符为\t。yunze 2023-07-10
        text = re.sub(r'([;；!?。！？\?])([^”’\n])', r"\1\t\2", text)  # 单字符断句符   # 去除句点分隔符
        text = re.sub(r'(\.{6})([^"’”」』\n])', r"\1\t\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』\n])', r"\1\t\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?\n])', r'\1\t\2', text)
        text = re.sub(r"(\n)", r"\1\t", text)   # 在所有 \n 之后加入 \t
        # text = text.rstrip()  # 保留段落尾部换行符，故注释。yunze 2023-07-10
        # 注意：分号、破折号、英文双引号等忽略，需要的再做些简单调整即可。
        text = re.sub(r'([\n])([^;；!?，。！？\?])', r'\1\t\2', text) # 换行符单独处理。在原本 \n 之后加入分隔符 \t 保证分段也进行分句. yunze 2023-07-10
        ls = [i for i in text.split("\t") if i and re.search("[^\s]", i)]     # 改为 \t 换行；过滤空字符串和仅含有空白符的字符串
        ls = [re.sub(r'([\s]*)([^\s])', r'\2', i) for i in ls]  # 过滤句前空白符
        for ele in ls:
            if len(ele) > self.sentence_size:
                ele1 = re.sub(r'([,，]["’”」』]{0,2})([^,，])', r'\1\t\2', ele)     # 逗号分句
                ele1_ls = ele1.split("\t")

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        ls = [sample for sample in ls if len(sample) > 3]   # 过滤 3 个字符及以下的短句，过滤无意义短句，同时代替text.rstrip()的作用
        #print("OUTOUT ls:",ls)
        return ls