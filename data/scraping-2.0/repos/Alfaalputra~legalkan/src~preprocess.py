import json
import os
import sys
from langchain.docstore.document import Document

path_this = os.path.abspath(os.path.dirname(__file__))
path_root = os.path.join(path_this, "..")
path_data = os.path.join(path_root, "data")
sys.path.append(path_data)

class Preprocessing():

    def process(self, data):
        """
        data: json file
        """

        dataset = []
        datas = json.load(open("data/dataset.json"))
        datas = datas["data"]

        for i in datas:
            dataset.append(Document(page_content=i))

        return dataset