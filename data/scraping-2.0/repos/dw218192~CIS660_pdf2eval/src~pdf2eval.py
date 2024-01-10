# todo: fix bug when user clicks process again after attach
import os
import sys
from queue import Queue

import openai
import PyPDF2
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread, pyqtSignal

import config
from prompt import Prompt


class ResultSectionType:
	SUMMARY : int = 0
	INTERESTING : int = 1
	DISLIKE : int = 2
	QUESTION : int = 3
	
def get_result_types() -> list[str]:
	ret = []
	for attr_name, attr_value in ResultSectionType.__dict__.items():
		if not callable(attr_value):
			if not attr_name.startswith('__') or not attr_name.endswith('__'):
				ret.append(attr_name)
	return ret

class WorkerResult(object):
	def __init__(self):
		super().__init__()
		self.paper_section_summary : list[str] = []
		self.results : dict[str,str] = {}
	
	def set_section(self, name : str, text : str):
		self.results[name] = text
	
	def get_section(self, name : str) -> str:
		if name not in self.results:
			raise RuntimeError('unknown section: {}'.format(name))
		return self.results[name]
	
	def write_plain(self, file):
		for _, text in self.results.items():
			file.write(text)
			file.write('\n')
			file.write('-' * 20)
			file.write('\n')

class GenerationParams(object):
	def __init__(self) -> None:
		super().__init__()
		self.summary_algorithm = config.SummaryAlgorithm.NAIVE
		self.qa_algorithm = config.QAAlgorithm.NAIVE
		self.writing_sample = ''

class PdfWorker(QThread):
	PROCESS_REQUEST = 0
	REDO_REQUEST = 1
	TERMINATE_REQUEST = 2
	
	progress_signal = pyqtSignal(str, int, int)
	result_receiver_signal = pyqtSignal(WorkerResult)

	def __init__(self, parent, pdf_name : str, params : GenerationParams):
		super().__init__(parent)
		self.pdf_name = pdf_name
		self.request_queue = Queue()
		self.params = params

	def update_prog(self, msg = ''):
		self.cur_prog = min(self.cur_prog + 1, self.total_prog)
		self.progress_signal.emit(msg, self.cur_prog, self.total_prog)

	def process_sections(self, pages) -> list[str]:
		p = Prompt()
		p.add(Prompt.SYS).add(config.SUMMARY_SYS_PROMPT)

		page_summary = []

		user = None # stores user message
		for i, page in enumerate(pages):
			# remove previous user query
			p.remove(user)

			# formulate user prompt
			user = p.add(Prompt.USER)
			(user.add_important(config.SUMMARY_USER_PROMPT + '\n')
				.add(page.extract_text()))

			cur_page_summary = p.dispatch()
			if self.params.summary_algorithm == config.SummaryAlgorithm.FULL_CONTEXT:
				# store the current summary as context
				# later pages have greater importance
				context = p.add(Prompt.ASSIST)
				(context.add_important('the summary of page {} is:\n'.format(i))
					.add(cur_page_summary, i))

			self.update_prog('processing page {}'.format(i))
			page_summary.append(cur_page_summary)
		return page_summary

	@staticmethod
	def _get_result_section_prompt(page_summary : list[str], type : int) -> Prompt:
		p = Prompt()
		if type == ResultSectionType.SUMMARY:
			# p.add(Prompt.SYS).add()
			user = p.add(Prompt.USER).add_important(config.FINAL_SUMMARY_PROMPT + '\n')
			for summary in page_summary:
				user.add(summary)
		else:
			assist = p.add(Prompt.ASSIST).add_important("the summary of the paper is: \n")
			for summary in page_summary:
				assist.add(summary)
			for spice in config.SPICE:
				assist = p.add(Prompt.ASSIST).add(spice)

			user = (p.add(Prompt.USER)
					.add_important("please answer the following question based on the summary of the academic paper provided above"))
			if type == ResultSectionType.INTERESTING:
				user.add(config.INTERESTING_PROMPT)
			elif type == ResultSectionType.DISLIKE:
				user.add(config.DISLIKE_PROMPT)
			elif type == ResultSectionType.QUESTION:
				user.add(config.QUESTION_PROMPT)
		return p
	
	def get_result(self, page_summary : list[str], type : int) -> str:
		p = PdfWorker._get_result_section_prompt(page_summary, type)
		if len(self.params.writing_sample) > 0:
			text = p.dispatch()
			p = Prompt()
			p.add(Prompt.USER).add_important(config.IMITATION_PROMPT_FMT.format(self.params.writing_sample, text))
		return p.dispatch()

	def run(self):
		result_section_types = get_result_types()

		while True:
			task_type, arg = self.request_queue.get()
			if task_type == PdfWorker.PROCESS_REQUEST:
				self.result = WorkerResult()

				self.cur_prog = 0
				reader = PyPDF2.PdfFileReader(self.pdf_name)
				self.total_prog = len(reader.pages) + len(result_section_types) + 1

				with open('out.txt', 'w', encoding="utf-8") as text_file:
					self.result.paper_section_summary = self.process_sections(reader.pages)
					all_summary = self.result.paper_section_summary
					# text_file.write(str(section_summary))
					# text_file.write('\n\n')

					if self.params.qa_algorithm == config.QAAlgorithm.FULL_CONTEXT:
						for i, type_name in enumerate(result_section_types):
							result = self.get_result(all_summary, i)
							self.result.set_section(type_name, result)
							self.update_prog('writing section {}'.format(type_name))
					else:
						# naive approach uses the total summary as context for answering questions
						type = ResultSectionType.SUMMARY
						total_summary = self.get_result(all_summary, type)
						self.result.set_section(result_section_types[type], total_summary)
						self.update_prog('writing section {}'.format(result_section_types[type]))

						for i, type_name in enumerate(result_section_types):
							if i == ResultSectionType.SUMMARY:
								continue
							self.result.set_section(type_name, 
								self.get_result([total_summary], i))
							self.update_prog('writing section {}'.format(type_name))

					self.result.write_plain(text_file)
					self.update_prog()
			elif task_type == PdfWorker.REDO_REQUEST:
				redo_type, = arg
				self.cur_prog, self.total_prog = 0, 2
				all_summary = self.result.paper_section_summary
				self.update_prog('rewriting section {}'.format(result_section_types[redo_type]))

				if self.params.qa_algorithm == config.QAAlgorithm.FULL_CONTEXT:
					text = self.get_result(all_summary, redo_type)
					self.result.set_section(result_section_types[redo_type], text)
				else:
					if redo_type == ResultSectionType.SUMMARY:
						total_summary = self.get_result(all_summary, redo_type)
						self.result.set_section(result_section_types[redo_type], total_summary)
					else:
						total_summary = self.result.get_section(result_section_types[ResultSectionType.SUMMARY])
						text = self.get_result(all_summary, redo_type)
						self.result.set_section(result_section_types[redo_type], text)
				with open('out.txt', 'w', encoding="utf-8") as text_file:
					self.result.write_plain(text_file)
				self.update_prog()

			elif task_type == PdfWorker.TERMINATE_REQUEST:
				break
			else:
				raise RuntimeError('unknown worker request: {}'.format(task_type))

class Window(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		openai.api_key = config.load_last_api_key()
		self.init_ui()
		self.pdf_file = ''
		self.worker_params = GenerationParams() # Worker readonly, Window RW
		self.worker = None

	def init_ui(self):
		uic.loadUi('window.ui', self)
		self.browseBtn.clicked.connect(self.get_pdf)
		self.processBtn.clicked.connect(self.process_pdf)
		self.processBtn.setEnabled(False)
		self.contextSummaryBtn.stateChanged.connect(self.set_summary_algorithm)
		self.contextSummaryBtn.setCheckState(0)
		self.fullContextQABtn.stateChanged.connect(self.set_qa_algorithm)
		self.fullContextQABtn.setCheckState(0)
		self.redoBtn1.clicked.connect(lambda : self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.SUMMARY))
		self.redoBtn2.clicked.connect(lambda : self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.INTERESTING))
		self.redoBtn3.clicked.connect(lambda : self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.DISLIKE))
		self.redoBtn4.clicked.connect(lambda : self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.QUESTION))
		self.paraphraseBtn.clicked.connect(self.redo_all)
		self.apiKeyText.setText(openai.api_key)
		self.apiKeyText.editingFinished.connect(self.set_api_key)
		self.sampleWritingText.setPlainText(config.get(config.WRITING_SAMPLE))
		self.sampleWritingText.textChanged.connect(self.set_writing_sample)

		self.set_pdf_dependent_btns(False)
		self.show()

	def get_pdf(self):
		fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open PDF', '', 'PDF Files (*.pdf)')
		# check if the file exists
		if os.path.exists(fname[0]):
			self.pdf_file = fname[0]
			self.processBtn.setEnabled(True)
			self.print("file attached")
		else:
			self.pdf_file = ''
			self.processBtn.setEnabled(False)
			self.print("file not found")
	
	def set_pdf_dependent_btns(self, state : bool):
		self.redoBtn1.setEnabled(state)
		self.redoBtn2.setEnabled(state)
		self.redoBtn3.setEnabled(state)
		self.redoBtn4.setEnabled(state)
		self.paraphraseBtn.setEnabled(state)

	def send_worker_request(self, type, *args):
		if self.worker is not None:
			self.worker.request_queue.put((type, args))
	def new_worker(self):
		self.worker = PdfWorker(self, self.pdf_file, self.worker_params)
		self.worker.progress_signal.connect(self.set_progress)
		self.worker.start()
	def redo_all(self):
		self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.SUMMARY)
		self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.INTERESTING)
		self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.DISLIKE)
		self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.QUESTION)
	def process_pdf(self):
		self.send_worker_request(PdfWorker.TERMINATE_REQUEST)
		self.new_worker()
		self.browseBtn.setEnabled(False)
		self.processBtn.setEnabled(False)
		self.set_pdf_dependent_btns(False)
		self.send_worker_request(PdfWorker.PROCESS_REQUEST)

	def set_api_key(self):
		openai.api_key = self.apiKeyText.text()
		config.set_api_key(self.apiKeyText.text())
	
	def set_writing_sample(self):
		self.worker_params.writing_sample = self.sampleWritingText.toPlainText()
		config.set(config.WRITING_SAMPLE, self.sampleWritingText.toPlainText())

	def set_summary_algorithm(self, state : int):
		if state > 0:
			self.worker_params.summary_algorithm = config.SummaryAlgorithm.FULL_CONTEXT
		else:
			self.worker_params.summary_algorithm = config.SummaryAlgorithm.NAIVE

	def set_qa_algorithm(self, state : int):
		if state > 0:
			self.worker_params.qa_algorithm = config.QAAlgorithm.FULL_CONTEXT
		else:
			self.worker_params.qa_algorithm = config.QAAlgorithm.NAIVE

	def print(self, text):
		self.messageLabel.setText(text)
		self.messageLabel.adjustSize()

	def set_progress(self, msg, value, total):
		# set progress bar
		if value == 0:
			self.pbar.setValue(0)
			self.print('')
		elif value == total:
			self.pbar.setValue(0)
			self.browseBtn.setEnabled(True)
			self.processBtn.setEnabled(True)
			self.set_pdf_dependent_btns(True)
			self.print('finished')
		else:
			self.pbar.setValue(int(value / total * 100))
			self.print(msg)

if __name__ == "__main__":
	try:
		app = QtWidgets.QApplication([])
		a_window = Window()
		code = app.exec_()
		if a_window.worker is not None:
			a_window.send_worker_request(PdfWorker.TERMINATE_REQUEST)
			a_window.worker.wait()
	finally:
		config.save()
	sys.exit(code)