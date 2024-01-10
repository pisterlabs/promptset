import json
from os import PathLike


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.docstore.document import Document


def get_file_sha1(file_path: PathLike) -> str:
    import hashlib

    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def get_str_punc_and_whitespace_chars_pct(text):
    # write a python function to count the % of chars in a str that is 1) punctuation, or 2) \t, \n
    import string

    punc_and_whitespace_chars = string.punctuation + "\t\n"
    count = sum(1 for c in text if c in punc_and_whitespace_chars)
    return count / len(text)


def langchain_doc_to_chatgpt_retrieval_plugin_doc(doc: 'Document') -> dict:
    # example doc:
    #   {
    #     "page_content": "形成性评价1\n题量: 12\n 满分: 20.0\n 创建者: 陈鑫源\n 作答时间: 04-01 11:35 ⾄ 06-13 23:59\nA\n先天性⼼脏病\nB\n2型糖尿病\nC\n⽀⽓管哮喘\nD\n佝偻病\nE\n消化性溃疡\nA\n⾎钠增⾼会刺激⼼房肌细胞合成和释放ANP\nB\n⾎容量增加会刺激⼼房肌细胞合成和释放ANP\nC\nANP减少肾素分泌\nD\nANP促进醛固酮的滞钠效应\nE\nANP抑制⾎管紧张素的缩⾎管效应\nA\n肺梗死\nB\n肺⼼病\nC\n⼆尖瓣狭窄\nD\n三尖瓣狭窄\nE\n肺⽓肿\nA\n神经-肌⾁的兴奋性升⾼\nB\n少尿\nC\n胃肠道功能亢进\n⼀. 单选题（共10题，10分）\n1. (单选题, 1分) 下列哪项疾病的病因主要是免疫因素( )\n2. (单选题, 1分) 以下哪项关于⼼房钠尿肽(ANP)的说法是错误的( )\n3. (单选题, 1分) 易发⽣肺⽔肿的病因是( )\n4. (单选题, 1分) 重度低钾⾎症或缺钾的病⼈常有( )\n1\n2\n3\n4\n6\n7\n8\n9\n11\n12\n⼀. 单选题（10分）\n⼆. 简答题（10分）\n保存\n提交\n作业\n",
    #     "metadata": {
    #       "source": "/Users/tscp/Documents/sch/pathophysiology/patho-midterm-形成性评价1.pdf",
    #       "file_path": "/Users/tscp/Documents/sch/pathophysiology/patho-midterm-形成性评价1.pdf",
    #       "page_number": 1,
    #       "total_pages": 3,
    #       "format": "PDF 1.4",
    #       "title": "",
    #       "author": "",
    #       "subject": "",
    #       "keywords": "",
    #       "creator": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36",
    #       "producer": "Skia/PDF m100",
    #       "creationDate": "D:20220405013804+00'00'",
    #       "modDate": "D:20220405013804+00'00'",
    #       "trapped": ""
    #     },
    #   }

    # returned dict:
    # id = item.get("id", None)
    # text = item.get("text", None)
    # source = item.get("source", None)
    # source_id = item.get("source_id", None)
    # url = item.get("url", None)
    # created_at = item.get("created_at", None)
    # author = item.get("author", None)
    return {
        "text": doc.page_content,
        "source": doc.metadata['source']
        + f'page {doc.metadata["page"] + 1}/{doc.metadata["total_pages"]}',
        "author": doc.metadata['author'],
        "created_at": doc.metadata['creationDate'],
        "url": f'file://{doc.metadata["file_path"]}',
    }
