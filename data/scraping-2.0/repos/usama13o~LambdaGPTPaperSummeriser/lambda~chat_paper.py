import argparse
import base64
import configparser
import datetime
import json
import os
import re
from collections import namedtuple

import arxiv
import numpy as np
import openai
import requests
import tenacity
import tiktoken

import fitz, io, os
from PIL import Image
from paper import Paper
from reader import Reader




def chat_paper_main(pdf_path):

    if pdf_path:
        reader1 = Reader()
        # 开始判断是路径还是文件：
        paper_list = []
        if pdf_path.endswith(".pdf"):
            paper_list.append(Paper(path=pdf_path))
        else:
            for root, dirs, files in os.walk(pdf_path):
                print("root:", root, "dirs:", dirs, 'files:', files)  # current directory path
                for filename in files:
                    # If a PDF file is found, it is copied to the target folder.
                    if filename.endswith(".pdf"):
                        paper_list.append(Paper(path=os.path.join(root, filename)))
        print("------------------paper_num: {}------------------".format(len(paper_list)))
        [print(paper_index, paper_name.path.split('\\')[-1]) for paper_index, paper_name in enumerate(paper_list)]
        reader1.summary_with_chat(paper_list=paper_list)


# if __name__ == '__main__':
    # pdf_path = r"pdfs\Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models.pdf"
    # chat_paper_main(pdf_path=pdf_path)