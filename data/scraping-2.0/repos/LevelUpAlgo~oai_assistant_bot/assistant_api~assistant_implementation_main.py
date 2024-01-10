#!/usr/bin/env conda activate whopo
from abc import ABC, abstractmethod
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich import box, pretty, print
import os, zipfile
import os
import zipfile
import chardet
import uuid
from rich import print as rprint
from rich.console import Console, Text
from rich.text import Text
from datetime import datetime

pretty.install()

class IConsoleManager(ABC):
	@abstractmethod
	def print(self, message, style=None):
		pass

	@abstractmethod
	def add_row_to_table(self, row):
		pass

	@abstractmethod
	def print_table(self):
		pass

class ConsoleManager(IConsoleManager):
	def __init__(self):
		self.console = Console(
			width=100,
			color_system="auto",
			force_terminal=True,
			legacy_windows=False,
			record=True,
			markup=True,
			emoji=True,
			highlight=True,
			log_time_format="[%X]",
			log_path=False,
			soft_wrap=True,
			no_color=False,
			style="none",
			tab_size=4,
			_environ=os.environ,
		)
		self.table = Table(title="Assistants", box=box.DOUBLE_EDGE)
		self.table.add_column("ID", justify="right", style="cyan", no_wrap=True)
		self.table.add_column("Name", style="magenta")
		self.table.add_column("Description", style="cyan")
		self.table.add_column("Status", justify="right", style="green")
		self.table.add_column("Model", justify="right", style="yellow")
		self.table.add_column("Created At", justify="right", style="blue")

	def print(self, message, style=None):
		self.console.print(message, style=style)

	def add_row_to_table(self, row):
		self.table.add_row(*row)

	def print_table(self):
		self.console.print(self.table)

class IOpenAIManager(ABC):
	@abstractmethod
	def __init__(self):
		pass

class OpenAIManager(IOpenAIManager):
	def __init__(self):
		self.client = OpenAI()

class IAssistant(ABC):
	@abstractmethod
	def __init__(self, client, assistant_id):
		pass

	@abstractmethod
	def retrieve_assistant(self):
		pass

	@abstractmethod
	def update_assistant(self, file_ids):
		pass

class Assistant(IAssistant):
	def __init__(self, client, assistant_id):
		self.client = client
		self.assistant_id = assistant_id
		self.assistant = self.retrieve_assistant()

	def retrieve_assistant(self):
		assistant = self.client.beta.assistants.retrieve(self.assistant_id)
		return {_: my_assistant for _, my_assistant in assistant}

	def update_assistant(self, file_ids):
		return self.client.beta.assistants.update(self.assistant["id"], file_ids=file_ids)

class IFile(ABC):
	@abstractmethod
	def __init__(self, client, filepath, purpose):
		pass

	@abstractmethod
	def create_file(self):
		pass

class File(IFile):
	def __init__(self, client, filepath, purpose):
		self.client = client
		self.filepath = filepath
		self.purpose = purpose
		self.file_object = self.create_file()

	def create_file(self):
		with open(self.filepath, 'rb') as file:
			return self.client.files.create(file=file, purpose=self.purpose)

class IThread(ABC):
	@abstractmethod
	def __init__(self, client):
		pass

	@abstractmethod
	def create_thread(self):
		pass

class Thread(IThread):
	def __init__(self, client):
		self.client = client
		self.thread = self.create_thread()

	def create_thread(self):
		return self.client.beta.threads.create()

class IMessage(ABC):
	@abstractmethod
	def __init__(self, client, thread_id, file_ids, role, content):
		pass

	@abstractmethod
	def create_message(self):
		pass

	@abstractmethod
	def retrieve_message(self, message_id):
		pass

class Message(IMessage):
	def __init__(self, client, thread_id, file_ids, role, content):
		self.client = client
		self.thread_id = thread_id
		self.file_ids = file_ids
		self.role = role
		self.content = content
		self.thread_message = self.create_message()

	def create_message(self):
		return self.client.beta.threads.messages.create(
			thread_id=self.thread_id,
			file_ids=self.file_ids,
			role=self.role,
			content=self.content,
		)

	def retrieve_message(self, message_id):
		return self.client.beta.threads.messages.retrieve(
			message_id=message_id,
			thread_id=self.thread_id,
		)

class IRunStepDetailsPrinter(ABC):
	@abstractmethod
	def __init__(self, console_manager):
		pass

	@abstractmethod
	def print_run_step_details(self, run_steps):
		pass

class RunStepDetailsPrinter(IRunStepDetailsPrinter):
	def __init__(self, console_manager):
		self.console_manager = console_manager

	def print_run_step_details(self):
		run_steps =   openai_manager.client.beta.threads.runs.steps.list(
			thread_id=thread.thread.id,
			order="desc",
			run_id=run.id
		)
		for run_step in run_steps.data:
			self.console_manager.print(f"Status: {run_step.status}")

			if run_step.step_details is not None:
				self.console_manager.print("Tool calls:")
				for tool_call in run_step.step_details:
					self.console_manager.print(tool_call)
					try:
						if tool_call.code_interpreter is not None:
							self.console_manager.print(f"Input: {tool_call.code_interpreter.input}")
							for output in tool_call.code_interpreter.outputs:
								self.console_manager.print(f"Output: {output.logs}")

					except AttributeError:
						pass

class IFileDownloader(ABC):
	@abstractmethod
	def __init__(self, client):
		pass

	@abstractmethod
	def download_file(self, file_id, output_path):
		pass

class FileDownloader(IFileDownloader):
	def __init__(self, client):
		self.client = client

	def download_file(self, file_id, output_path):
		# Check if downloads directory exists, if not, create it
		if not os.path.exists('downloads'):
			os.makedirs('downloads')

		# Modify output_path to include 'downloads' directory
		output_path = os.path.join('downloads', output_path)

		info = self.client.files.content(file_id)
		info.stream_to_file(output_path)

class DirectoryManager:
		@staticmethod
		def zip_directory(directory_path, zip_file_name):
				with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
						for root, dirs, files in os.walk(directory_path):
								if 'node_modules' in dirs:
										dirs.remove('node_modules')  # don't visit node_modules directories
								for file in files:
										file_path = os.path.join(root, file)
										zipf.write(file_path, os.path.relpath(file_path, directory_path))
				return zip_file_name



class StatusPrinter:
		def __init__(self, openai_manager, console_manager, file_downloader):
				self.openai_manager = openai_manager
				self.console_manager = console_manager
				self.file_downloader = file_downloader
				self.console = Console()

		def print_file_details(self, file_name, file_id):
				table = Table(show_header=True, header_style="bold magenta")
				table.add_column("File Name", style="dim", width=50)
				table.add_column("File ID", style="dim", width=50)
				table.add_row(Text(file_name, style="green"), Text(file_id, style="blue"))
				self.console.print(table)

		def status(self, thread):
				thread_messages = self.openai_manager.client.beta.threads.messages.list(thread.thread.id, order='asc')
				for msg in thread_messages:
						for content in msg.content:
								self.console.print(Text(content.text.value))
								if hasattr(content.text, 'annotations'):
										for annotation in content.text.annotations:
												file_name = os.path.basename(annotation.text)
												file_id =  annotation.file_path.file_id
												self.print_file_details(file_name, annotation.file_path.file_id)

												self.file_downloader.download_file(file_id, file_name)
												self.console.print(Text('Downloaded file: ', style="bold green"), file_name)
								self.console.rule(
										title=Text(msg.id, style="bold red"),
										characters='*',
										style='bold green',
										align='center'
								)

		def update_status(self, thread_id):
				while True:
						runs = self.openai_manager.client.beta.threads.runs.list(thread_id=thread_id, order="desc")
						for run in runs.data:
								self.console.clear()
								table = Table(show_header=True, header_style="bold magenta")
								table.add_column("ID", style="dim", width=50)
								table.add_column("Status", style="dim", width=50)
								table.add_column("Created At", style="dim", width=50)
								table.add_column("Started At", style="dim", width=50)
								table.add_column("Expires At", style="dim", width=50)
								table.add_row(
										Text(run.id, style="green"),
										Text(run.status, style="blue"),
										Text(datetime.fromtimestamp(run.created_at).strftime('%Y-%m-%d %H:%M:%S'), style="green"),
										Text(datetime.fromtimestamp(run.started_at).strftime('%Y-%m-%d %H:%M:%S'), style="blue"),
										Text(datetime.fromtimestamp(run.expires_at).strftime('%Y-%m-%d %H:%M:%S'), style="green")
								)
								self.console.print(table)
						time.sleep(2)

		def log_thread(self, thread_id):
				while True:
						messages = self.openai_manager.client.beta.threads.messages.list(thread_id=thread_id)
						for message in messages.data:
								self.console.clear()
								table = Table(show_header=True, header_style="bold magenta")
								table.add_column("Message ID", style="dim", width=50)
								table.add_column("Role", style="dim", width=50)
								table.add_column("Content", style="dim", width=50)
								table.add_column("Created At", style="dim", width=50)
								table.add_row(
										Text(message.id, style="green"),
										Text(message.role, style="blue"),
										Text(message.content[0].text.value, style="green"),
										Text(datetime.fromtimestamp(message.created_at).strftime('%Y-%m-%d %H:%M:%S'), style="blue")
								)
								self.console.print(table)
						time.sleep(2)



# Usage

console_manager = ConsoleManager()
openai_manager = OpenAIManager()
run_step_details_printer = RunStepDetailsPrinter(console_manager)
file_downloader = FileDownloader(openai_manager.client)
status_printer = StatusPrinter(openai_manager, console_manager, file_downloader)
directory_manager = DirectoryManager()

# Zip the directory
home = os.environ['HOME']
zip_file_name = directory_manager.zip_directory(f'{home}/Desktop/oai_docs/gentlement_club_nyc' , f'{home}/Desktop/oai_docs/gentlement_club_nyc.zip')
file = File(openai_manager.client, "/Users/clockcoin/Desktop/oai_docs/downloads/nyc_limo_service_spa_updated.zip", 'assistants')

assistant = openai_manager.client.beta.assistants.create(
		name="Fullstack engineer (Web dev)",
		instructions="You are a personal math tutor. Write and run code to answer math questions.",
		tools=[{"type": "code_interpreter"} , {"type": "retrieval"}],
		model="gpt-4-1106-preview",
		file_ids=[file.file_object.id, ]
)
assistant.update_assistant([file.file_object.id, ])
assistant = Assistant(openai_manager.client,  assistant.id)
thread2 = Thread(openai_manager.client)
message = Message(openai_manager.client, thread2.thread.id, [file.file_object.id], "user", "[your in flow on a 30mg addy and a redbull, your code is detailed and excellent]\n\
Ok add animations and make this the most modern and sleek vue app. (the attachmen t is a zip file)\n")

run = openai_manager.client.beta.threads.runs.create(
	thread_id=thread2.thread.id,
	instructions="1. Add a reactive scroll bar to the landing page\n2.Add a chat bubble and messaging\3. Add a schedule a ride with a calandar picker.\4. Use modern font and design guidelines",
	assistant_id=assistant.assistant_id
)
status_printer.log_thread(thread2.thread.id)
status_printer.update_status(thread.thread2.id)
status_printer.status(thread2)
# console_manager.add_row_to_table([
# 	assistant.assistant["id"],
# 	assistant.assistant["name"],
# 	assistant.assistant.get("description", "No description"),
# 	"Active" if assistant.assistant["tools"] else "Inactive",
# 	assistant.assistant["model"],
# 	str(assistant.assistant["created_at"]),
# ])
# console_manager.print_table()

# # Zip the directory

# def zip_directory(directory_path, zip_file_name):
# 	with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
# 		for root, dirs, files in os.walk(directory_path):
# 			if 'node_modules' in dirs:
# 				dirs.remove('node_modules')  # don't visit node_modules directories
# 			for file in files:
# 				file_path = os.path.join(root, file)
# 				zipf.write(file_path, os.path.relpath(file_path, directory_path))
# 	return zip_file_name

# #  Add the zipped directory as a file
# home = os.environ['HOME']
# zip_file_name = zip_directory(f'{home}/Desktop/oai_docs/assistant_api' , f'{home}/Desktop/oai_docs/assistant_api1.zip')
# file = File(openai_manager.client, zip_file_name, 'assistants')

# # file2 = File(openai_manager.client,"/Users/clockcoin/Desktop/oai_docs/assistant_implementation.py", 'assistants')
# print(file.file_object)

# assistant.update_assistant([file.file_object.id, ])

# thread = Thread(openai_manager.client)
# message = Message(openai_manager.client, thread.thread.id, [file.file_object.id], "user", "[your in flow on a 30mg addy and a redbull, your code is detailed and excellent\n$5,000 tip for exellent work]\n \
# 	yes, implement all missing functionality from the documentation and make sure the code is excellent. \n ")
# console_manager.print(message.thread_message)

# # run = openai_manager.client.beta.threads.runs.create(
# run = openai_manager.client.beta.threads.runs.create(
# 	thread_id=thread.thread.id,
# 	assistant_id='asst_M8rgFTKZWASS1T40IplYycHb'
# )



# status  =   openai_manager.client.beta.threads.runs.list(
# 	 thread_id=thread.thread.id,
# 	 order="desc",
# )
# def print_file_details(file_name, file_id):
# 	table = Table(show_header=True, header_style="bold magenta")
# 	table.add_column("File Name", style="dim", width=50)
# 	table.add_column("File ID", style="dim", width=50)
# 	table.add_row(Text(file_name, style="green"), Text(file_id, style="blue"))
# 	console = Console()
# 	console.print(table)

# def status():
# 	console = Console()
# 	thread_messages = openai_manager.client.beta.threads.messages.list(thread.thread.id, order='asc')
# 	for msg in thread_messages:
# 		for content in msg.content:
# 			console.print(Text(content.text.value))
# 			if hasattr(content.text, 'annotations'):
# 				for annotation in content.text.annotations:
# 					file_name = os.path.basename(annotation.text)
# 					file_id =  annotation.file_path.file_id
# 					print_file_details(file_name, annotation.file_path.file_id)

# 					file_downloader.download_file(file_id, file_name)
# 					console.print(Text('Downloaded file: ', style="bold green"), file_name)
# 			console.rule(
# 				title=Text(msg.id, style="bold red"),
# 				characters='*',
# 				style='bold green',
# 				align='center'
# 			)
# status()

# run_step_details_printer.print_run_step_details()
# openai_manager.client.beta.threads.messages.files.list(
# 	thread_id=thread.thread.id,
# 	message_id=msg.id
# )


# # # message_files = openai_manager.client.beta.threads.messages.files.retrieve(
# # # 	thread_id=thread.thread.id,
# # # 	message_id='msg_2swWDtcN0zr81T5pcEsoyKYB',
# # # 	file_id="file-OgnBzHZeJyy5M5j6RYC5iGsr"
# # # )

# # file_list = openai_manager.client.files.list()
# # file_downloader.download_file('file-V69PEIkYkXClnO3pda9MUSFC', "pinadh2e.zip")
