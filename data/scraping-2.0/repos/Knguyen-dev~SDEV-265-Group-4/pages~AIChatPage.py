import time
import customtkinter as ctk
import sys, os
sys.path.append("..")
from classes.models import Message
from tkinter import messagebox
from customtkinter import CTkCanvas
from PIL import Image 

'''
+ AIChatPage: Frame that represents the page where the user and AI send chat messages to each other in order to 
	write their story.

Attributes/Variables:
- master (App): 'App' class instance from 'Main.py' 
- innerPageFrame (CTkFrame): Page frame that contains all of the widgets for the page and is used to center it
- header (CTkFrame): Header of the page frame
- heading (CTkLabel): Heading message
- storyStateMessage (CTkLabel): Label that tells the user what kind of story they're writing, whether they're remixing, writing 
	a new story, continuing a saved story, etc.
- pageStatusMessage (CTkLabel): Indicates status of the page like when the user is currently waiting on the AI for a response
	or whether an occur has occurred.
- chatBox (CTkTextbox): Textbox that shows the messages of the user and AI.
- chatInputSection (CTkFrame): Section with all of the input related widgets
- chatEntry (CTkEntry): Input text box where user types in their message
- openSaveStoryBtn (CTkButton): Button that redirects the user to the saveStoryPage
- sendChatBtn (CTkButton): Button that sends the chat to the AI.

Methods:
- processUserChat(self): Sends a user chat message to the AI and gets its response.
- renderChat(self, messageObj): Renders message text onto the screen given a messgae object.
'''
class AIChatPage(ctk.CTkFrame):
	def __init__(self, master):
		self.master = master
		super().__init__(self.master, fg_color=self.master.theme["main_clr"], corner_radius=0)

		self.chatEntry_height=20
		self.max_chatEntry_height = 400 # 4 line max view space
		# This logic prevents the dynamically resizing msgbox from overexpanding - Powered by Nuke The Dev
		self.msgbox_height=30
		self.max_msgbox_height = 1200 # 12 line max view space

		self.setup_ui()

		'''
		- Cases for the initial state:
		1. User is continuing a saved story
		2. Using is currently writing a remixed story, it's unsaved. If storyGenObj is detected with the constructor's logic then 
			we're rendering the AI's first response to a user's remixed story, which would be the first message of the chat.
		3. Using is continuing an unsaved story that isn't a remix.
		'''
		# If there have been any unsaved messages, render them 
		if self.master.unsavedStoryMessages:
			for messageObj in self.master.unsavedStoryMessages:
				self.renderChatMessageObj(messageObj)

		# if storyGenObj exists, then we have to process a generator that the AI returned
		# NOTE: In this case, when storyGenObj exists here, that means it was set by the remixStoryPage, 
		# and this generator contains a response for remixing a story
		if self.master.storyGenObj:
			self.processAIChat()

		# call the function once to start the periodic check
		self.check_length_and_resize()
	
	def setup_ui(self):
		innerPageFrame = ctk.CTkFrame(self, fg_color=self.master.theme["sub_clr"])
		innerPageFrame.pack(expand=True)
		header = ctk.CTkFrame(innerPageFrame, fg_color="transparent")
		heading = ctk.CTkLabel(header, text="Write Your Story!", font=("Helvetica", 32), text_color=self.master.theme["label_clr"])
		storyStateMessage = ctk.CTkLabel(header, text="", text_color=self.master.theme["label_clr"])
		self.pageStatusMessage = ctk.CTkLabel(header, text="StoryBot is currently waiting for your input.", font=("Helvetica", 24), text_color=self.master.theme["label_clr"])

		# This is where we view the messages sent from the AI and the User
		self.chatBox = ctk.CTkScrollableFrame(innerPageFrame, fg_color=self.master.theme["main_clr"], width=800, height=400)

		# Section with all of the input options the user has for the AIChatPage
		chatInputSection = ctk.CTkFrame(innerPageFrame, fg_color="transparent")
		self.chatEntry = ctk.CTkTextbox(chatInputSection, height=50, width=600, fg_color=self.master.theme["entry_clr"], text_color=self.master.theme["entry_text_clr"], font=("Helvetica", 16), wrap="word", activate_scrollbars=True)
  
		try:
			openSaveStoryBtn_image = ctk.CTkImage(Image.open(os.path.join(self.master.image_path, 'glass_save_btn.png')), size=(50, 50))
		except IOError as e:
			messagebox.showerror("Error", f"Failed to load image: {e}")
			return

		self.openSaveStoryBtn = ctk.CTkButton(chatInputSection, image=openSaveStoryBtn_image, height=10, width=20, text="Save Story", font=("Helvetica", 16, "bold"), text_color=self.master.theme["btn_text_clr"], fg_color='transparent', hover_color=self.master.theme["hover_clr"], command=lambda: self.master.openPage("saveStoryPage"))
		
		sendChatBtn_image = ctk.CTkImage(Image.open(os.path.join(self.master.image_path, 'glass_send_btn.png')),
				size=(50, 50))
		self.sendChatBtn = ctk.CTkButton(chatInputSection, corner_radius=0, image=sendChatBtn_image, height=10, width=20, text="Send", font=("Helvetica", 16, "bold"), text_color=self.master.theme["btn_text_clr"], fg_color='transparent', hover_color=self.master.theme["hover_clr"], hover=True, anchor="e", command=self.processUserChat)
	
		# Structure and style widgets accordingly
		header.grid(row=0, column=0, pady=10)
		heading.grid(row=0, column=0)
		storyStateMessage.grid(row=1, column=0)
		self.pageStatusMessage.grid(row=2, column=0)
		self.chatBox.grid(row=1, column=0, pady=10)
		chatInputSection.grid(row=2, column=0, pady=20)
		self.chatEntry.grid(row=0, column=0, padx=10, pady=5)
		self.sendChatBtn.grid(row=0, column=1, padx=5)
		self.openSaveStoryBtn.grid(row=0, column=2, padx=5)
  
		if self.master.isSavedStory: 
			# Render saved messages associated with the current story
			for messageObj in self.master.currentStory.messages:
				self.renderChatMessageObj(messageObj)
			storyStateMessage.configure(text=f"Currently continuing '{self.master.currentStory.storyTitle}'!")
		elif self.master.isRemixedStory: 
			storyStateMessage.configure(text=f"Currently writing a remix based on '{self.master.currentStory.storyTitle}'!")
		else:
			storyStateMessage.configure(text=f"Currently continuing writing a new story!")
 
	def renderChatMessageObj(self, messageObj):
		'''
		- Renders messageObj as chat messages on the screen
		- NOTE: Only good for rendering saved, unsaved, and user chats because those are easily in message object form.
			For rendering AI's response, it's a generator so use processAIChat(self).
		'''
		# Chat window is read and write now
		messageText = messageObj.text

		# If it's an AI message, else it was a message sent by the user
		if messageObj.isAISender:
			messageText = "StoryBot: " + messageText
		else:
			messageText = f"{self.master.loggedInUser.username}: " + messageText 

		# access the last msgbox to print to
		msgbox = self.drawMsgBox()
		msgbox.insert("1.0", messageText)

		# Calculate the required height of the message
		height = self.expandEntryBox(msgLength=messageText)
		# Now we use the calculated `height` parameter to set the height of the msgbox
		print('height=', height)
		msgbox.configure(height=height)

    # Check for the length of text in the entry field and adjust entry field height accordingly
	def check_length_and_resize(self):
		# get the text in the CTkTextbox
		if self.chatEntry_height <= self.max_chatEntry_height:
			new_height=self.expandEntryBox(self.chatEntry.get('1.0', 'end'))
			self.chatEntry.configure(height=new_height)

		# schedule the next check in 5000 milliseconds (1 second)
		self.after(2000, self.check_length_and_resize)
		
	def processUserChat(self):
		'''
		- Sends the user chat message to the ai, for the ai to respond, then goes to render both of those chat messages
		1. userMessage (Message): Message object containing text that the user sent
		2. AIResponse (Generator): Generator object containing text that the AI generated in response to the user
		'''
		# Check if user actually sent something
		if (self.chatEntry.get('1.0', 'end').strip() == ""):
			messagebox.showwarning('Empty Message!', 'Please enter a valid message!')
			return

		# Process and render the user's message
		# The .strip() method ensures that a user cannot type whitespaces 
		# before the message content which has been known to cause an openAI api exception
		userMessage = Message(text=self.chatEntry.get('1.0', 'end').strip(), isAISender=False)
		self.renderChatMessageObj(userMessage)
		self.master.unsavedStoryMessages.append(userMessage) 	
		
		# Clear entry widget when user sends a message
		self.chatEntry.delete(1.0, "end")
			
		AIResponse = self.master.storyGPT.sendStoryPrompt(userMessage.text) 
		self.master.storyGenObj = AIResponse # type: ignore 

		# Process and render AI's message
		self.processAIChat()

	def drawMsgBox(self):
		msgbox = ctk.CTkTextbox(self.chatBox, fg_color=self.master.theme["entry_clr"], font=("Helvetica", 16), width=750, height=20, wrap="word", activate_scrollbars=False)
		msgbox.grid(row=len(self.master.msgboxes), column=0, padx=5, pady=5, sticky="nsew")
		self.master.msgboxes.append(msgbox)
		return msgbox

	def expandEntryBox(self, msgLength):
		num_chars = len(msgLength) # The number of characters in the text box at hand
  
		# Calculate the number of lines at 100 characters per line of text onscreen
		num_lines = num_chars // 100 # Use integer division to get the number of full lines
		if num_chars % 100 > 0: # If there are any remaining characters, they will form an additional line
			num_lines += 1
   
		# Calculate the height
		height = num_lines * 30 # Each line is 30 units high
  
		# Now you can use `height` to set the height of your widget
		return height

	def processAIChat(self):
		'''
		- Handles the proces of processing the AI's generated chat.
		1. Enable and disable certain parts of the UI, preventing the user from sending another 
			message to the AI until the first one is finished. Also prevent the user from being 
			able to redirect themselves to other pages, so that they don't lose their AI generated message.
		2. Renders chunks of the messages as they're being obtained from openai. 
		3. Save the ai's generated message to unsavedStoryMessages so that we can keep track of it
		'''

		# Access the current messagebox at it's index
		# Disable send chat button as user can't send another chat until the ai is finished
		self.sendChatBtn.configure(state="disabled")		

		# Ensure user can't navigate to other pages while AI is generating message
		self.master.sidebar.disableSidebarButtons()
		self.openSaveStoryBtn.configure(state="disabled")

		# Update page status message to indicate that AI is currently generating a message 
		self.pageStatusMessage.configure(text="Please wait here until StoryBot is finished!")
		
		# Message object that contains the text from the generator
		messageObj = Message(text="", isAISender=True) 
		chunkIndex = 0

		
		# Create a new real-time dynamically resizing msg bubble to display AI response in
		msgbox = self.drawMsgBox()
		# Make the chat box writable
		msgbox.configure(state="normal")

		msgbox.insert('end', 'Story Bot: ')
		for chunk in self.master.storyGenObj: 
			if any(chunk.endswith(char) for char in ['.', '?', '!']):
				punct_marks = ['.', '?', '!']
				for mark in punct_marks:
					if chunk.endswith(f'{mark}'):
						msgbox.insert('end', f"{mark}" + " ")
			else:
				msgbox.insert('end', chunk)
			
			# Enables smooth real time typing
			if (self.msgbox_height <= self.max_msgbox_height):
					new_height=self.expandEntryBox(msgbox.get('1.0', 'end'))
					self.msgbox_height = new_height

			# Dynamically resize the height of the current msgbox
			msgbox.update()
			msgbox.configure(height=self.msgbox_height)
			self.update_idletasks()
			# add the chunk onto the message object's text since we want to keep track of this message; then increment chunkIndex
			messageObj.text += chunk
			chunkIndex += 1

		#reset the msgbox height after each message
		self.msgbox_height=30
			
		# AI response processing is done, so append message object and variables related to processing a message
		self.master.unsavedStoryMessages.append(messageObj) 
		self.master.storyGenObj = None 

		# Scroll to bottom and make chatbox read only
		# This allows the user to view the latest text
		msgbox.see("end-1c")
		msgbox.configure(state="disabled")

		# Allow the user to send another message and navigate to other pages
		self.openSaveStoryBtn.configure(state="normal")
		self.sendChatBtn.configure(state="normal")
		self.master.sidebar.updateSidebar() 

		# Update the page status message to indicate the ai is done
		self.pageStatusMessage.configure(text="StoryBot is currently waiting for your input.")