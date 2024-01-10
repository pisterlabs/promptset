from openai import OpenAI
import time
import sys
import wx

client = OpenAI()

class MyFrame(wx.Frame):    
	def __init__(self):
		super().__init__(parent=None, title='OpenAI Assistant Creator', size = wx.Size(600, 375))
		panel = wx.Panel(self)        
		my_sizer = wx.BoxSizer(wx.VERTICAL)

		self.text_ctrl = wx.TextCtrl(panel)
		my_sizer.Add(self.text_ctrl, 0, wx.ALL | wx.EXPAND, 15)

		self.text_reply = wx.TextCtrl(panel, id=-1, size=(50,75), style=wx.TE_READONLY | wx.TE_MULTILINE | wx.TE_RICH) 
		my_sizer.Add(self.text_reply, 0, wx.ALL | wx.EXPAND, 30)

		my_btn = wx.Button(panel, label='Press Me')
		my_btn.Bind(wx.EVT_BUTTON, self.on_press)
		my_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 5) 

		self.text_status = wx.TextCtrl(panel, id=-1, size=(250,-1), style=wx.TE_READONLY) 
		my_sizer.Add(self.text_status, 0, wx.ALL | wx.LEFT, 40)   
		       
		panel.SetSizer(my_sizer)  
		self.Show()

	def set_status_text(self, text):
		wx.CallAfter(self.text_status.SetValue, text)

	def set_reply_text(self, text):
		wx.CallAfter(self.text_reply.SetValue, text)

	def on_press(self, event):
		value = self.text_ctrl.GetValue()
		if not value:
			self.set_status_text("You didn't enter anything!  Ya big dummy!!!  :)")
		else:
			#print(f'You typed: "{value}"')
			#self.set_status_text(value)
			thread = client.beta.threads.create()

			input_message = client.beta.threads.messages.create(
				thread_id = thread.id,
				role = "user",
				content = value
			)
			if input_message is None:
				self.set_status_text("Failed to create input message")
			else:
				run = client.beta.threads.runs.create(
					thread_id = thread.id,
					assistant_id = assistant.id
				)
			if run is None:
				self.set_status_text("Failed to create run for message")


			#Per the OpenAI architecture get a thread and message going here to prepare for user interaction

			#while run.status != "completed":
			#	# Be nice to the API
			#	time.sleep(0.5)
			#	run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

			#try:
			#	messages = client.beta.threads.messages.list(
			#			thread_id = thread.id
			#		)
			#except:
			#	print("message list failed")
			#	exit()

			run_done = False

			while not run_done:
				
				#Give the API a break
				time.sleep(0.5)

				#retrieve the runs
				run = client.beta.threads.runs.retrieve(
					thread_id = thread.id,
					run_id = run.id
				)

				if run.status in ["queued", "in_progress", "cancelling"]:
					run_done = False
				elif run.status in ["cancelled", "failed", "completed", "expired"]:
					run_done = True
				elif run.status in ["requires_action"]:
					self.set_status_text("Required Action...need to do more coding!!")
					run_done = False
				else:
					self.set_status_text("chat: unrecognised run status" + run.status)
					run_done = True

				# send status to status callbackxs
				self.set_status_text(run.status)

			# retrieve messages on the thread
			reply_messages = client.beta.threads.messages.list(thread_id=thread.id, order = "asc", after=input_message.id)
			if reply_messages is None:
				self.set_status_text("chat: failed to retrieve messages")

			# concatenate all messages into a single reply skipping the first which is our question
			reply = ""
			need_newline = False
			for message in reply_messages.data:
				reply = reply + message.content[0].text.value
				if need_newline:
					reply = reply + "\n"
				need_newline = True

			if reply is None or reply == "":
				self.set_status_text("chat: failed to retrieve latest reply")
			else:
				self.set_reply_text(reply)


assistant_name=None
module_name=None
assistant_created=False

#https://www.tutorialspoint.com/python/index.html

#TODO: Provide for running script without assistant name provided
if assistant_name is None:
	assistant_name = str(sys.argv[1])

if module_name is None:
	module_name = "gpt-4-1106-preview"

# if assistant exists, use it
assistants_list = client.beta.assistants.list()
for existing_assistant in assistants_list.data:
	if existing_assistant.name == assistant_name:
		print("setup_assistant: using existing assistant: " + existing_assistant.name + " id:" + existing_assistant.id)
		assistant = client.beta.assistants.retrieve(existing_assistant.id)
		assistant_created = True

if assistant_created == False:
	assistant = client.beta.assistants.create(
		name = assistant_name,
		instructions = "You are a math tutor. Write and run code to answer math questions",
		tools = [{"type": "code_interpreter"}],
		model = "gpt-4-1106-preview"
	)
	print("Assistant " + assistant_name + " was created")

#run = client.beta.threads.runs.create(
#		thread_id = thread.id,
#		assistant_id = assistant.id
#		)

if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()