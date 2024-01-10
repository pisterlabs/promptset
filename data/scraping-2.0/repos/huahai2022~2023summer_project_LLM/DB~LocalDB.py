import logging
import os
import re
import time
from typing import List

import PyPDF2
import torch
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, UnstructuredPDFLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter

import vector_stroe
from config import VECTOR_SEARCH_TOP_K, CHUNK_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, embedding_model_dict, ROOT_PATH, \
	STREAME, PROMPT_TEMPLATE
from config import SENTENCE_SIZE
from models.base import BaseAnswer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import logging

from vector_stroe import MyFAISS

# 配置日志信息
logging.basicConfig(filename=f'{__name__}', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# 定义一个类，用来处理，中文的语义分割
class ChineseTextSplitter(CharacterTextSplitter):
	def __init__(self, pdf: bool = False, sentence_size: int = SENTENCE_SIZE, **kwargs):
		super().__init__(**kwargs)
		self.pdf = pdf
		self.sentence_size = sentence_size

	def split_text(self, text: str) -> List[str]:
		# 如果需要对 PDF 文件进行处理
		if self.pdf:
			text = re.sub(r"\n{3,}", r"\n", text)  # 出现三个或者三个以上换行符，将他们替换成一个换行符
			text = re.sub(r'\s', " ", text)  # 匹配任意空白字符，包括空格，制表符，换行符，替换成一个空格
			text = re.sub(r"\n\n", "", text)  # 连续出现两个换行符，替换成一个空字符串
		# 定义分句规则
		sent_sep_pattern = re.compile(r'[。！？…]+|[^\s。！？…]+')
		# 对文本进行分句
		sentences = []
		for sentence in sent_sep_pattern.findall(text):
			# 如果句子长度超过阈值，则进一步分句
			while len(sentence) > self.sentence_size:
				sub_sentence = sentence[:self.sentence_size]
				# 在子句结尾处查找分句符号
				last_sep = re.search(r'[。！？…]', sub_sentence[::-1])
				if last_sep:
					sub_sentence = sub_sentence[:-last_sep.end()]
				sentences.append(sub_sentence)
				sentence = sentence[len(sub_sentence):]
			sentences.append(sentence)

		return sentences


class ChineseTextSplitter2(CharacterTextSplitter):
	def __init__(self, pdf: bool = False, sentence_size: int = SENTENCE_SIZE, **kwargs):
		super().__init__(**kwargs)
		self.pdf = pdf
		self.sentence_size = sentence_size

	def split_text(self, text: str) -> List[str]:
		if self.pdf:
			text = re.sub(r"\n{3,}", r"\n", text)  #三个换行符替换为一个
			text = re.sub('\s', " ", text)		  #空白字符替换成为一个
			text = re.sub("\n\n", "", text)			#两个换行符替换为一个
		text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 匹配标点，不包含后面的
		text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
		text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
		text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
		text = text.rstrip()  # 段尾如果有多余的\n就去掉它
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


class ChineseTextSplitter3(CharacterTextSplitter):
	def __init__(self, pdf: bool = False, sentence_size: int = SENTENCE_SIZE, **kwargs):
		super().__init__(**kwargs)
		self.pdf = pdf
		self.sentence_size = sentence_size

	def split_text(self, text: str) -> List[str]:  ##此处需要进一步优化逻辑
		text = re.sub(r"\n{3,}", "", text)
		text = re.sub('\s', "", text)
		text = re.sub("\n\n", "", text)
		# 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
		text = text.rstrip()  # 段尾如果有多余的\n就去掉它
		sentences = []
		for i in range(100):
			start_val = str(i + 1)
			end_val = str(i + 2)
			start_index = re.search(start_val, text).start()
			end_index = re.search(end_val, text).start()
			substring = text[start_index:end_index]
			print(substring)
			sentences.append(substring)
		return sentences


def extract_text_from_pdf(filepath):
	"""
    提取 PDF 文件中的文本内容
    """
	with open(filepath, 'rb') as f:
		# 创建 PDF 文件读取器
		reader = PyPDF2.PdfReader(f)
		# 获取 PDF 文件中的页数
		num_pages = len(reader.pages)
		# 遍历每一页，提取文本内容
		text = ''
		for i in range(num_pages):
			page = reader.pages[i]
			text += page.extract_text()
	return text


# 用来处理大文件

def load_file(filepath, sentence_size=SENTENCE_SIZE):
	if filepath.lower().endswith(".md"):
		loader = UnstructuredFileLoader(filepath, mode="elements")
		# UnstructuredFileLoader类的mode参数定义了在加载文件时如何解析文本内容，具体的mode参数取值及其含义如下：
		# - `"text"`：将整个文本文件加载为一个字符串对象。
		# - `"lines"`：将文本文件按行加载为一个字符串列表，每个字符串代表一行文本。
		# - `"elements"`：将文本文件加载为一个元素对象列表，每个元素代表一个Markdown元素（如段落、标题、列表、代码块等）。
		# - `"json"`：将JSON格式的文本文件加载为一个Python对象。
		# - `"pickle"`：将pickle序列化格式的文本文件加载为一个Python对象。
		# 其中，前三种mode参数适用于Markdown格式的文本文件，后两种mode参数适用于其他格式的文本文件。在使用时，需要根据不同的文本文件类型和数据格式选择合适的mode参数。
		docs = loader.load()

	elif filepath.lower().endswith(".txt"):
		loader = TextLoader(filepath, autodetect_encoding=True)
		txtSplitter = ChineseTextSplitter2(pdf=False)  # 定义中文分割规则
		docs = loader.load_and_split(txtSplitter)  # 按照中文分割规则进行划分
	# elif filepath.lower().endswith(".pdf"):            #效果一般
	#     docs=extract_text_from_pdf(filepath)
	#     pdfSplitter=ChineseTextSplitter2(pdf=True)
	#     docs=pdfSplitter.split_text(docs)
	elif filepath.lower().endswith(".pdf"):
		loader = UnstructuredPDFLoader(file_path=filepath)
		pdfSplitter = ChineseTextSplitter2(pdf=True)
		docs = loader.load_and_split(pdfSplitter)
	else:
		loader = UnstructuredFileLoader(filepath, mode="elements")
		textsplitter = ChineseTextSplitter2(pdf=False, sentence_size=sentence_size)
		docs = loader.load_and_split(text_splitter=textsplitter)
	return docs


# 用于从文件夹里面查找文件
def find_files(filepath):
	file_paths = []  # 用于存储文件路径的列表
	for root, directories, files in os.walk(filepath):
		# root:当前目录路径
		# directories:当前目录名称
		# files:文件名称
		for filename in files:
			# 将文件的完整路径添加到列表中
			filepath = os.path.join(root, filename)
			file_paths.append(filepath)
	return file_paths


def show_docs(docs):
	for doc in docs:
		print(doc)


def generate_prompt(related_docs: List[str],
					query: str,
					prompt_template: str = PROMPT_TEMPLATE, ) -> str:
	context = "\n".join([doc.page_content for doc in related_docs])
	prompt = prompt_template.replace("{question}", query).replace("{context}", context)
	return prompt


class MyLocalDB:
	llm: BaseAnswer = None
	embaddings: object = None
	top_targets: int = VECTOR_SEARCH_TOP_K
	chunk_size: int = CHUNK_SIZE
	chunk_conent: bool = True
	score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

	def __init__(self, embedding_model: str = "text2vec",
				 embedding_device="cuda",
				 llm_model: BaseAnswer = None,
				 top_k=VECTOR_SEARCH_TOP_K
				 ):
		self.llm = llm_model
		self.embaddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
												model_kwargs={'device': embedding_device})
		self.top_targets = top_k

	def use_local_vector_stroe(self):
		db_path = os.path.join(ROOT_PATH, "vector_store")
		return db_path

	def add_file_to_vector_stroe(self, filepath, db_name):
		db_path = os.path.join(ROOT_PATH, db_name)
		vectors = MyFAISS.load_local(db_path, embeddings=self.embaddings)
		loaded_files = []
		if os.path.isfile(filepath):
			file = os.path.split(filepath)[-1]  # 取文件名
			try:
				loaded_files = load_file(filepath, sentence_size=SENTENCE_SIZE)
				print(f"{file}已经成功加载")
			except Exception as e:
				print(e)
		else:
			# 找到一个文件夹，去文件夹中找文件
			files = find_files(filepath)
			# 在分割文字的时候，可能出现文字找不到的情况，一定要加上try catch
			for file in files:
				try:
					loaded_files += load_file(file)
				except Exception as e:
					print(e)
		show_docs(loaded_files)
		print("正在将文件存储到知识库中")
		vectors.add_documents(loaded_files)
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()
		vectors.save_local(db_path)
		return db_path

	def create_new_vectors(self, db_path, filepath):
		loaded_files = []
		print("加载文件中~~")
		if os.path.isfile(filepath):
			file = os.path.split(filepath)[-1]  # 取文件名
			try:
				loaded_files = load_file(filepath, sentence_size=SENTENCE_SIZE)
				print(f"{file}已经成功加载")
			except Exception as e:
				print(e)
		else:
			# 找到一个文件夹，去文件夹中找文件
			files = find_files(filepath)
			# 在分割文字的时候，可能出现文字找不到的情况，一定要加上try catch
			for file in files:
				try:
					loaded_files += load_file(file)
				except Exception as e:
					print(e)
		show_docs(loaded_files)
		db_path = os.path.join(ROOT_PATH, db_path)
		vectors = MyFAISS.from_documents(loaded_files, embedding=self.embaddings)
		print(dir(vectors))
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()
		vectors.save_local(db_path)
		return db_path

	def init_vector_stroe(self, filepath):
		global vectors
		loaded_files = []

		if os.path.isfile(filepath):
			file = os.path.split(filepath)[-1]  # 取文件名
			try:
				loaded_files = load_file(filepath, sentence_size=SENTENCE_SIZE)
			except Exception as e:
				print(e)
		else:
			# 找到一个文件夹，去文件夹中找文件
			files = find_files(filepath)

			# 在分割文字的时候，可能出现文字找不到的情况，一定要加上try catch
			for file in files:
				try:
					loaded_files += load_file(file)
				except Exception as e:
					print(e)
		show_docs(loaded_files)
		logging.info("文件加载完毕，生成向量库中")

		db_path = os.path.join(ROOT_PATH, "vector_store")
		vectors = MyFAISS.from_documents(loaded_files, embedding=self.embaddings)
		print(dir(vectors))
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()

		if os.path.isdir(db_path) and "index.faiss" in os.listdir(db_path):
			vectors = MyFAISS.load_local(db_path, embeddings=self.embaddings)
			vectors.add_documents(loaded_files)
			torch.cuda.empty_cache()
			torch.cuda.ipc_collect()
		else:
			logging.info("什么逼玩意")
		vectors.save_local(db_path)
		return db_path, loaded_files

	# TODO:loaded_files是一个Document类型数据

	def get_answer_based_query(self,
							   query,
							   vs_path,
							   chat_history=[],
							   stream: bool = STREAME
							   ):
		vectors = MyFAISS.load_local(vs_path, embeddings=self.embaddings)
		vectors.chunk_size = self.chunk_size
		vectors.chunk_conent = self.chunk_conent
		vectors.score_threshold = self.score_threshold
		docs_with_score = vectors.similarity_search_with_score(query, k=self.top_targets)
		print(docs_with_score)
		# print(f"""相似度得分最高的是{docs_with_score.page_content},得分为{docs_with_score.metadata["score"]}""")
		# TODO:可以从这里取出值
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()
		if len(docs_with_score) > 0:
			# TODO:说明从向量库中找到了相关的消息
			prompt = generate_prompt(related_docs=docs_with_score, query=query)
		else:
			prompt = query
		for answer_reslut in self.llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=stream):
			response = answer_reslut.llm_output["answer"]
			history = answer_reslut.history
			history[-1][0] = query
			response = {"query": query,
						"prompt": prompt,
						"result": response,
						"source_documents": docs_with_score}
			yield response, history

	def list_files_from_vector(self,db_path):
		vector_stroe=MyFAISS.load_local(db_path,embeddings=self.embaddings)
		docs=vector_stroe.list_docs()
		return [os.path.split(doc)[-1] for doc in docs]
