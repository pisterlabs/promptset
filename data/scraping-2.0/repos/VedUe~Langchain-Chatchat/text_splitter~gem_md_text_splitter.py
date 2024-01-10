from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
import sys
sys.path.append('/code/Langchain-Chatchat')
from configs.kb_config import CHUNK_SIZE, OVERLAP_SIZE

min_size = 100

def is_only_title(text):
    if len(text.split('\n\n')) == 1:
        return True
    return False
        
def c_split(h, c, chunk_size, overlap_size, sent_tmp):
    """c_split 将过长的内容切分成块，并给每一份都附上标题

    Args:
        h (str): 标题(不包含换行符)
        c (str): 内容
        chunk_size (int): 块长（不严格的）
        overlap_size (int): 块与块的重合长度
    """
    cps = re.split(r'(?<=。)\n(?=\n+[^\u4e00-\u9fa5])',c)
    cc_list = []
    cc = ''
    flag1 = False
    flag2 = False
    for idx, cp in enumerate(cps):
        if len(cc) < chunk_size - 100:
            flag1 = True
        cc += cp + '\n\n'
        if len(cc) >= chunk_size - 100:
            flag2 = True
        if idx == len(cps):
            sent_tmp = cc + '\n\n'
        if flag1 and flag2:
            flag1 = False
            flag2 = False
            cc_list.append(f'{h}\n\n{cc}')
            if len(cp) <= overlap_size:
                cc = cp
            else:
                cc = ''
    return cc_list
      
def get_hc(sent, h, c):
    if is_only_title(sent):
        h = sent
        c = ''
    else:
        h = re.split(r'\n+', sent)[0]
        c = sent[len(h + '\n\n'):]
    return h, c
        
def into_list(sent, list, h, c, sent_tmp):
    if len(sent_tmp + sent) >= min_size and len(sent_tmp + sent) <= CHUNK_SIZE:
        list.append(f'{sent_tmp}\n\n{h}\n{c}')
        sent_tmp = ''
    elif len(sent_tmp + sent) > CHUNK_SIZE:
        if sent_tmp != '':
            h = f'{sent_tmp}\n\n{h}'
        list += c_split(h, c, CHUNK_SIZE, OVERLAP_SIZE, sent_tmp)
        sent_tmp = ''
    else:
        sent_tmp += f'{h}\n\n{sent}'
    return sent_tmp
        
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
        c1 = ''
        c2 = ''
        c3 = ''        
        sent_list1 = []
        sent_tmp = ''
        for sent in sent_list0:
            if re.match(r'^# .+', sent):
                h1, c1 = get_hc(sent, h1, c1)
                sent_tmp = into_list(sent, sent_list1, h1, c1, sent_tmp)
            elif re.match(r'^## .+', sent):
                h2, c2 = get_hc(sent, h2, c2)
                sent_tmp = into_list(sent, sent_list1, f'{h1}\n{h2}', c2, sent_tmp)
            elif re.match(r'^### .+', sent):
                h3, c3 = get_hc(sent, h3, c3)
                sent_tmp = into_list(sent, sent_list1, f'{h1}\n{h2}\n{h3}', c3, sent_tmp)
                
        # 去重复行
        sent_list2 = []
        for sent in sent_list1:
            lst = sent.split('\n')
            new_lst = []
            [new_lst.append(i) for i in lst if i not in new_lst and i != '\n' and i != '']
            sent_list2.append("\n\n".join(new_lst))
        return sent_list2
    
if __name__ == "__main__":
    with open('/code/Langchain-Chatchat/knowledge_base/gems/content/10-宝石各论（GIC证书）.md', 'r', encoding='utf-8') as fr:
        text = fr.read()
    text_splitter = GemTextSplitter()
    sent_list = text_splitter.split_text(text)
    for i in range(10, 20):
        print(sent_list[i] + '\n===')
