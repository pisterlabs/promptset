"""
 pdf document class: using fitz to parse and extract content
"""
import json

import fitz, io, os
import re
from PIL import Image
import cfg
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import time
import requests
import configparser
import os

config = configparser.ConfigParser()
config.read('apikey.ini')
chat_api_list = config.get('OpenAI', 'OPENAI_API_KEYS')[1:-1].replace('\'', '').split(',')
os.environ["OPENAI_API_KEY"] = chat_api_list[0]

TOP_K = 20
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20
COMPRESSION = False
QUERIES = {
    'general': ["What is the company of the report?", "What sector does the company belong to?", "Where is the company located?",
                #"What climate-related issues are discussed in this report?"
          ],
    'tcfd_1': "How does the company's board oversee climate-related risks and opportunities?",
    'tcfd_2': "What is the role of management in assessing and managing climate-related risks and opportunities?",
    'tcfd_3': "What are the most relevant climate-related risks and opportunities that the organisation has identified over the short, medium, and long term? Are risks clearly associated with a horizon?",
    'tcfd_4': "How do climate-related risks and opportunities impact the organisation's businesses strategy, economic and financial performance, and financial planning?",
    'tcfd_5': "How resilient is the organisation's strategy when considering different climate-related scenarios, including a 2°C target or lower scenario? How resilient is the organisation's strategy when considering climate physical risks?",
    'tcfd_6': "What processes does the organisation use to identify and assess climate-related risks?",
    'tcfd_7': "How does the organisation manage climate-related risks?",
    'tcfd_8': "How are the processes for identifying, assessing, and managing climate-related risks integrated into the organisation's overall risk management?",
    'tcfd_9': "What metrics does the organisation use to assess climate-related risks and opportunities? How do the metrics help ensure that the performance is in line with its strategy and risk management process?",
    'tcfd_10': "Does the organisation disclose its Scope 1, Scope 2, and, if appropriate, Scope 3 greenhouse gas (GHG) emissions? What are the related risks and do they differ depending on the scope?",
    'tcfd_11': "What targets does the organisation use to understand/quantify/benchmark climate-related risks and opportunities? How is the organization performing against these targets?",
}


class Report:
    def __init__(self, path=None, url=None, title='', abs='', authers=[], store_path=None, top_k=TOP_K, db_path=None, retrieved_chunks_path=None):
        # Init the class on pdf with given path
        self.chunks = []
        self.page_idx = []
        self.path = path  # pdf path
        self.url = url # pdf url
        assert ((path is None and url is not None) or (path is not None and url is None)) # only need to pass in an url or a path
        self.store_path = store_path
        self.queries = QUERIES
        self.top_k = top_k  # retriever top-k
        self.compression = COMPRESSION
        self.section_names = []  # title
        self.section_texts = {}  # content
        self.db_path = db_path # set self.db_path
        self.retrieved_chunks_path = retrieved_chunks_path
        if title == '':
            if self.path:
                self.pdf = fitz.open(self.path)  # pdf
                #self.parse_pdf_from_url(self.url)
            else:
                self.parse_pdf_from_url(self.url)  # download from an URL
                #self.pdf = fitz.open(self.path)
            self.title = self.get_title()
            self.parse_pdf()
        else:
            self.title = title
        self.authers = authers
        self.abs = abs
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d + 1) for d in range(10)]
        self.first_image = ''

    def parse_pdf_from_url(self, url):
        response = requests.get(url)
        pdf = io.BytesIO(response.content)
        self.pdf = fitz.open(stream=pdf)

    def parse_pdf(self):
        if self.path:
            self.pdf = fitz.open(self.path)  # pdf
        else:
            self.parse_pdf_from_url(self.url) # download from an URL
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ' '.join(self.text_list)
        # self.section_page_dict = self._get_all_page_index() # paragraph and page map
        # print("section_page_dict", str(self.section_page_dict))
        # self.section_text_dict = self._get_all_page() # paragraph and content
        start_time = time.time()
        # _get_retriever load/store database from/to self.db_path
        self.retriever, self.vector_db = self._get_retriever(self.db_path)
        self.section_text_dict = self._retrieve_chunks()
        if not os.path.exists(self.retrieved_chunks_path):
            os.makedirs(self.retrieved_chunks_path)
        with open(os.path.join(self.retrieved_chunks_path, 'retrieved.json'), 'w') as f:
            to_dump = {
                key: {chunk.metadata['source']: [chunk.page_content, chunk.metadata['page']] for chunk in chunk_list}
                for key, chunk_list in self.section_text_dict.items()
            }
            json.dump(to_dump, f)

        end_time = time.time()
        print('time for retrieval:', end_time - start_time)
        self.section_text_dict.update({"title": self.title})
        # whether this is a valid pdf (use keyword to check)
        store_flag = True
        # for topic in self.sl:
        #     if len(self.section_text_dict[topic]) > 0:
        #         store_flag = True
        #         break
        if self.store_path is not None and store_flag:
            self.pdf.save(self.store_path)
        self.pdf.close()

    def get_image_path(self, image_path=''):
        """
        save first image of pdf and save to image.png，and return the path for gitee
        :param filename: image path
        :param image_path: save path
        :return:
        """
        # open file
        max_size = 0
        image_list = []
        with fitz.Document(self.path) as my_pdf_file:
            # go through all page
            for page_number in range(1, len(my_pdf_file) + 1):
                # check each
                page = my_pdf_file[page_number - 1]
                # current page
                images = page.get_images()
                # all image in the page
                for image_number, image in enumerate(page.get_images(), start=1):
                    # acquire xref
                    xref_value = image[0]
                    # get image
                    base_image = my_pdf_file.extract_image(xref_value)
                    # acquire image
                    image_bytes = base_image["image"]
                    # get the extension
                    ext = base_image["ext"]
                    # load
                    image = Image.open(io.BytesIO(image_bytes))
                    image_size = image.size[0] * image.size[1]
                    if image_size > max_size:
                        max_size = image_size
                    image_list.append(image)
        for image in image_list:
            image_size = image.size[0] * image.size[1]
            if image_size == max_size:
                image_name = f"image.{ext}"
                im_path = os.path.join(image_path, image_name)
                print("im_path:", im_path)

                max_pix = 480
                origin_min_pix = min(image.size[0], image.size[1])

                if image.size[0] > image.size[1]:
                    min_pix = int(image.size[1] * (max_pix / image.size[0]))
                    newsize = (max_pix, min_pix)
                else:
                    min_pix = int(image.size[0] * (max_pix / image.size[1]))
                    newsize = (min_pix, max_pix)
                image = image.resize(newsize)

                image.save(open(im_path, "wb"))
                return im_path, ext
        return None, None

    # recongnize the title by fontsize
    def get_chapter_names(self, ):
        # # open pdf
        doc = fitz.open(self.path)
        text_list = [page.get_text() for page in doc]
        all_text = ''
        for text in text_list:
            all_text += text
        # # create the list to store all the names
        chapter_names = []
        for line in all_text.split('\n'):
            line_list = line.split(' ')
            if '.' in line:
                point_split_list = line.split('.')
                space_split_list = line.split(' ')
                if 1 < len(space_split_list) < 5:
                    if 1 < len(point_split_list) < 5 and (
                            point_split_list[0] in self.roman_num or point_split_list[0] in self.digit_num):
                        print("line:", line)
                        chapter_names.append(line)

        return chapter_names

    def get_title(self):
        doc = self.pdf
        max_font_size = 0  # init fontsize 0
        max_string = ""  # init max string
        max_font_sizes = [0]
        for page in doc:  # go through all pages
            text = page.get_text("dict")  # acquire the info in the page
            blocks = text["blocks"]  # acquire text block
            for block in blocks:  # go through all blocks
                if block["type"] == 0 and len(block['lines']):  # if str
                    if len(block["lines"][0]["spans"]):
                        font_size = block["lines"][0]["spans"][0]["size"]  # acquire fontsize
                        max_font_sizes.append(font_size)
                        if font_size > max_font_size:
                            max_font_size = font_size  # update max fontsize
                            max_string = block["lines"][0]["spans"][0]["text"]  # update the title str
        max_font_sizes.sort()
        print("max_font_sizes", max_font_sizes[-10:])
        cur_title = ''
        for page in doc:  # go through all pages
            text = page.get_text("dict")
            blocks = text["blocks"]
            for block in blocks:
                if block["type"] == 0 and len(block['lines']):
                    if len(block["lines"][0]["spans"]):
                        cur_string = block["lines"][0]["spans"][0]["text"]
                        font_flags = block["lines"][0]["spans"][0]["flags"]
                        font_size = block["lines"][0]["spans"][0]["size"]
                        # print(font_size)
                        if abs(font_size - max_font_sizes[-1]) < 0.3 or abs(font_size - max_font_sizes[-2]) < 0.3:
                            # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)
                            if len(cur_string) > 4:
                                # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)
                                if cur_title == '':
                                    cur_title += cur_string
                                else:
                                    cur_title += ' ' + cur_string
                                    # break
        title = cur_title.replace('\n', ' ')
        return title

    # def _get_all_page_index(self):
    #     # topic search list
    #     section_list = self.sl
    #     section_page_dict = {}
    #     for page_index, page in enumerate(self.pdf):
    #         # acquire current content in the page
    #         cur_text = page.get_text()
    #         for section_name in section_list:
    #             if section_name not in section_page_dict:
    #                 section_page_dict[section_name] = []
    #             if search_page(cur_text, section_list[section_name]):
    #                 section_page_dict[section_name].append(page_index)
    #     # return topic and their pages
    #     return section_page_dict

    def _get_all_page(self):
        """
        get texts from pdf and return the content
        Returns:
            section_dict (dict): content for each topic，key is the topic name，value is the content。
        """
        text = ''
        text_list = []
        section_dict = {}
        text_list = [page.get_text() for page in self.pdf]
        for sec_index, sec_name in enumerate(self.section_page_dict):
            cur_sec_text = ''
            for page_i in self.section_page_dict[sec_name]:
                cur_sec_text += text_list[page_i]
            section_dict[sec_name] = cur_sec_text.replace('-\n', '').replace('\n', ' ')
        return section_dict

    # _get_retriever load/store database from/to self.db_path
    def _get_retriever(self, db_path):
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            # split by ["\n\n", "\n", " "].
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " "],
        )
        text_list = [page.get_text() for page in self.pdf]
        full_text = '\n\n'.join(text_list)
        for i, page in enumerate(self.pdf):
            page_chunks = text_splitter.split_text(page.get_text())
            self.page_idx.extend([i + 1] * len(page_chunks))
            self.chunks.extend(page_chunks)
        if os.path.exists(db_path):
            doc_search = FAISS.load_local(db_path, embeddings=embeddings)
        else:
            doc_search = FAISS.from_texts(self.chunks, embeddings,
                                          metadatas=[{"source": str(i), "page": str(page_idx)} for i, page_idx in
                                                     enumerate(self.page_idx)])

            doc_search.save_local(db_path)
        retriever = doc_search.as_retriever(search_kwargs={"k": self.top_k})

        return retriever, doc_search

    def _retrieve_chunks(self):
        section_text_dict = {}
        for key in self.queries.keys():
            if key == 'general':
                docs_1 = self.retriever.get_relevant_documents(self.queries[key][0])[:5]
                docs_2 = self.retriever.get_relevant_documents(self.queries[key][1])[:5]
                docs_3 = self.retriever.get_relevant_documents(self.queries[key][2])[:5]
                section_text_dict[key] = docs_1 + docs_2 + docs_3
            else:
                section_text_dict[key] = self.retriever.get_relevant_documents(self.queries[key])
        return section_text_dict


def search_page(content, search_list):
    if re.search("|".join(search_list), content.lower()):
        return True
    else:
        return False

