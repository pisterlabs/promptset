import sys

from PyQt6.QtWidgets import QMainWindow, QPushButton, QApplication, QVBoxLayout, QWidget, QTextEdit
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from openai import OpenAI

class Window(QMainWindow):
	def __init__(self, db, client):
		super().__init__()

		self.db = db
		self.client = client
		self.loading = False

		# setting title
		self.setWindowTitle("FITGPT")

		# setting geometry

		# showing all the widgets
		self.show()

		self.widget = QWidget()
		self.layout_v = QVBoxLayout()
		self.widget.setLayout(self.layout_v)
		self.setCentralWidget(self.widget)

		self.input = QTextEdit()
		self.input.setPlaceholderText("Pitanje")
		self.layout_v.addWidget(self.input)
		self.input.setMinimumHeight(400)

		self.button = QPushButton("Pitaj")
		self.layout_v.addWidget(self.button)
		self.button.clicked.connect(self.button_clicked)

		self.output = QTextEdit()
		self.output.setPlaceholderText("Output")
		self.layout_v.addWidget(self.output)
		self.output.setReadOnly(True)
		self.output.setFixedWidth(500)
		self.output.setFixedHeight(400)

	def button_clicked(self):
		if self.loading:
			return
		self.loading = True
		self.input.setReadOnly(True)
		self.button.setText("Samo sekunda...")
		self.output.setText("")
		query = self.input.toPlainText()

		results = self.db.similarity_search_with_relevance_scores(query, top_k=6)
		if len(results) == 0:
			self.output.setText(f"Nista nije nadjeno za upit: {query}")
			return

		context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

		stream = self.client.chat.completions.create(
			model="potrebo-samo-za-official-openai-api",
			messages=[
				{"role": "system","content": f"Always answer in the Bosnian language. Use only the provided context. Answer short and to the point. This is the context: {context}"},
				{"role": "user", "content": query},
			],
			temperature=0.7,
			stream=True,
			max_tokens=500,
		)

		for chunk in stream:
			print(chunk.choices[0].delta.content or "", end="")
			old = self.output.toPlainText()
			self.output.setText(old + chunk.choices[0].delta.content or "")
			self.output.repaint()

		self.output.append(f"\nIzvori: {[doc[0].metadata.get('source', None) for doc in results]}")
		self.loading = False
		self.input.setReadOnly(False)
		self.input.setText("")
		self.button.setText("Pitaj")


def __main__():
	# create pyqt5 app
	App = QApplication(sys.argv)
	embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
	db = Chroma(persist_directory="./chroma_openai_compat", embedding_function=embeddings)
	client = OpenAI(base_url="http://localhost:1234/v1", api_key="potrebo-samo-za-official-openai-api")  # Potreban openai kompatibilan server

	# create the instance of our Window
	window = Window(db, client)

	# start the app
	sys.exit(App.exec())

if __name__ == "__main__":
	__main__()