from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List


class GemTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = 600, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.match(r"\n{3,}", "\n", text)
            text = re.match('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(r'\n(?=#{1,3} )')
        sent_list0 = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list0:
                sent_list0[-1] += ele
            elif ele:
                sent_list0.append(ele)
                
        h1 = ''
        h2 = ''
        h3 = ''
        sent_list1 = []
        sent_tmp = ''
        for sent in sent_list0:
            if re.match(r'^# .+', sent):
                h1 = re.match(r'^# (.+)\n', sent).group(1).strip()
                sent_tmp += sent
            elif re.match(r'^## .+', sent):
                h2 = re.match(r'^## (.+)\n', sent).group(1).strip()
                sent_tmp += f'## {h1}·{sent[3:]}'
            elif re.match(r'^### .+', sent):
                h3 = re.match(r'^### (.+)\n*(.+)', sent).group(1).strip()
                c3 = sent[len(h3) + 6:]
                sent_tmp += f'### {h1}·{h2}·{h3}\n{c3}'
                sent_list1.append(sent_tmp)
                sent_tmp = ''             
        # h3 = ''
        # h4 = ''
        # for sent in sent_list1:veduetodo
        return sent_list1
    
if __name__ == "__main__":
    with open('knowledge_base/gems/content/08-相似宝石鉴定特征单晶-多晶（GIC证书）.md', 'r', encoding='utf-8') as fr:
        text = fr.read()

    text_splitter = GemTextSplitter()
    sent_list = text_splitter.split_text(text)
    print(sent_list[0])
    print(sent_list[1])
    print(sent_list[2])
    # for sent in sent_list:
    #     print(len(sent))