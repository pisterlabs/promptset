"""
@Time    : 2023/12/27 19:20
@Author  : yangzq80@gmail.com
@File    : load_document.py
"""

import time
from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
import tqdm
from langchain.vectorstores import Milvus


class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def pdf2text(filepath):
            import fitz # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            from rapidocr_onnxruntime import RapidOCR
            import numpy as np
            ocr = RapidOCR()
            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            for i, page in enumerate(doc):

                # 更新描述
                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(i))
                # 立即显示进度条更新结果
                b_unit.refresh()
                # TODO: 依据文本与图片顺序调整处理方式
                text = page.get_text("")
                resp += text + "\n"

                img_list = page.get_images()
                for img in img_list:
                    pix = fitz.Pixmap(doc, img[0])
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        resp += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    file_path="/home/ubuntu/yzq/gs-python/test/关于利州区思政课评选获奖情况的通知.pdf"
    start = time.time()
    # loader = RapidOCRPDFLoader(file_path)
    # pypdf比ocr快20倍 0.3s
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print(docs,time.time()-start)

    doc_chunks = loader.load_and_split()

    embeddings = HuggingFaceBgeEmbeddings(
        model_name='/home/ubuntu/yzq/models/bge-large-zh',
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': True}, # set True to compute cosine similarity
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )

    vector_db = Milvus.from_documents(
        doc_chunks,
        embeddings,
        collection_name="collection_2",
        connection_args={"host": "n3", "port": "19530"},
    )
    
    query = '谁获得一等奖'
    rs = vector_db.similarity_search(query)
    print("similarity_search-----",len(rs),rs[0].page_content)


