import datetime
import json
import os
import re
import sys
import threading
import winsound
import gui
import wx
from enum import Enum

import addonHandler
import api
import config
import queueHandler
import speech
import tones
import ui
from logHandler import log
from .consts import (
	ADDON_DIR, DATA_DIR,
	MODELS, MODEL_VISION,
	TOP_P_MIN, TOP_P_MAX,
	N_MIN, N_MAX,
	DEFAULT_SYSTEM_PROMPT
)
from .imagehelper import (
	describeFromImageFileList,
	encode_image,
	get_image_dimensions,
	resize_image,
)
from .recordthread import RecordThread
from .resultevent import ResultEvent, EVT_RESULT_ID

additionalLibsPath = os.path.join(ADDON_DIR, "lib")
sys.path.insert(0, additionalLibsPath)
import openai
import markdown2
sys.path.remove(additionalLibsPath)

addonHandler.initTranslation()

TTS_FILE_NAME = os.path.join(DATA_DIR, "tts.wav")
DATA_JSON_FP = os.path.join(DATA_DIR, "data.json")
URL_PATTERN = re.compile(r"^(?:http)s?://(?:[A-Z0-9-]+\.)+[A-Z]{2,6}(?::\d+)?(?:/?|[/?]\S+)$", re.IGNORECASE)

def EVT_RESULT(win, func):
	win.Connect(-1, -1, EVT_RESULT_ID, func)


def copyToClipAsHTML(html_content):
	html_data_object = wx.HTMLDataObject()
	html_data_object.SetHTML(html_content)
	if wx.TheClipboard.Open():
		wx.TheClipboard.Clear()
		wx.TheClipboard.SetData(html_data_object)
		wx.TheClipboard.Close()
	else:
		raise RuntimeError("Unable to open the clipboard")


def get_display_size(size):
	if size < 1024:
		return f"{size} B"
	if size < 1024 * 1024:
		return f"{size / 1024:.2f} KB"
	return f"{size / 1024 / 1024:.2f} MB"


class ImageFileTypes(Enum):

	UNKNOWN = 0
	IMAGE_LOCAL = 1
	IMAGE_URL = 2


class ImageFile:

	def __init__(
		self,
		path: str,
		name: str=None,
		description: str=None,
		size: int=-1,
		dimensions: tuple=None
	):
		if not isinstance(path, str):
			raise TypeError("path must be a string")
		self.path = path
		self.type = self._get_type()
		self.name = name or self._get_name()
		self.description = description
		if size and size > 0:
			self.size = get_display_size(size)
		else:
			self.size = self._get_size()
		self.dimensions = dimensions or self._get_dimensions()

	def _get_type(self):
		if os.path.exists(self.path):
			return ImageFileTypes.IMAGE_LOCAL
		if re.match(
			URL_PATTERN,
			self.path
		):
			return ImageFileTypes.IMAGE_URL
		return ImageFileTypes.UNKNOWN

	def _get_name(self):
		if self.type == ImageFileTypes.IMAGE_LOCAL:
			return os.path.basename(self.path)
		if self.type == ImageFileTypes.IMAGE_URL:
			return self.path.split("/")[-1]
		return "N/A"

	def _get_size(self):
		if self.type == ImageFileTypes.IMAGE_LOCAL:
			size = os.path.getsize(self.path)
			return get_display_size(size)
		return "N/A"

	def _get_dimensions(self):
		if self.type == ImageFileTypes.IMAGE_LOCAL:
			return get_image_dimensions(self.path)
		return None

	def __str__(self):
		return f"{self.name} ({self.path}, {self.size}, {self.dimensions}, {self.description})"


class CompletionThread(threading.Thread):

	def __init__(self, notifyWindow):
		threading.Thread.__init__(self)
		self._notifyWindow = notifyWindow
		self._wantAbort = 0

	def run(self):
		wnd = self._notifyWindow
		client = wnd.client
		conf = wnd.conf
		data = wnd.data
		block = HistoryBlock()
		system = wnd.systemText.GetValue().strip()
		block.system = system
		prompt = wnd.promptText.GetValue().strip()
		block.prompt = prompt
		model = wnd.getCurrentModel()
		block.model = model.name
		conf["model"] = model.name
		stream = conf["stream"]
		debug = conf["debug"]
		maxTokens = wnd.maxTokens.GetValue()
		block.maxTokens = maxTokens
		key_maxTokens = "maxTokens_%s" % model.name
		data[key_maxTokens] = maxTokens
		temperature = 1
		topP = 1
		if conf["advancedMode"]:
			temperature = wnd.temperature.GetValue() / 100
			key_temperature = "temperature_%s" % model.name
			data[key_temperature] = wnd.temperature.GetValue()

			topP = wnd.topP.GetValue() / 100
			conf["topP"] = wnd.topP.GetValue()

			debug = wnd.debugModeCheckBox.IsChecked()
			conf["debug"] = debug

			stream = wnd.streamModeCheckBox.IsChecked()
			conf["stream"] = stream

		block.temperature = temperature
		block.topP = topP
		conversationMode = conf["conversationMode"]
		conf["conversationMode"] = wnd.conversationCheckBox.IsChecked()
		block.pathList = wnd.pathList.copy()

		if not 0 <= temperature <= model.maxTemperature * 100:
			wx.PostEvent(self._notifyWindow, ResultEvent(_("Invalid temperature")))
			return
		if not TOP_P_MIN <= topP <= TOP_P_MAX:
			wx.PostEvent(self._notifyWindow, ResultEvent(_("Invalid top P")))
			return
		messages = []
		if system:
			messages.append({"role": "system", "content": system})
		wnd.getMessages(messages)
		if wnd.pathList:
			images = wnd.getImages()
			if images:
				messages.append({"role": "user", "content": images})
		if prompt:
			messages.append({"role": "user", "content": prompt})
		nbImages = 0
		for message in messages:
			if (
				message["role"] == "user"
				and not isinstance(message["content"], str)
			):
				for content in message["content"]:
					if content["type"] == "image_url":
						nbImages += 1
		msg = ""
		if nbImages == 1:
			# Translators: This is a message displayed when uploading one image to the API.
			msg = _("Uploading one image, please wait...")
		elif nbImages > 1:
			# Translators: This is a message displayed when uploading multiple images to the API.
			msg = _("Uploading %d images, please wait...") % nbImages
		else:
			# Translators: This is a message displayed when sending a request to the API.
			msg = _("Processing, please wait...")
		wnd.message(msg)
		if debug and nbImages:
			log.info(f"{nbImages} images")
			log.info(f"{json.dumps(messages, indent=2, ensure_ascii=False)}")
		params = {
			"model": model.name,
			"messages": messages,
			"temperature": temperature,
			"max_tokens": maxTokens,
			"top_p": topP,
			"stream": stream
		}
		try:
			response = client.chat.completions.create(**params)
		except BaseException as err:
			wx.PostEvent(self._notifyWindow, ResultEvent(repr(err)))
			return
		if wnd.lastBlock is None:
			wnd.firstBlock = wnd.lastBlock = block
		else:
			wnd.lastBlock.next = block
			block.previous = wnd.lastBlock
			wnd.lastBlock = block
		wnd.previousPrompt = wnd.promptText.GetValue()
		wnd.promptText.Clear()

		if stream:
			self._responseWithStream(response, block, debug)
		else:
			self._responseWithoutStream(response, block, debug)
		wnd.pathList.clear()
		wx.PostEvent(self._notifyWindow, ResultEvent())
		wnd.message(_("Ready"))

	def abort(self):
		self._wantAbort = True

	def _responseWithStream(self, response, block, debug=False):
		wnd = self._notifyWindow
		text = ""
		for i, event in enumerate(response):
			if wnd.stopRequest.is_set():
				break
			delta = event.choices[0].delta
			finish = event.choices[0].finish_reason
			text = ""
			if delta.content:
				text = delta.content
			block.responseText += text
		block.responseTerminated = True

	def _responseWithoutStream(self, response, block, debug=False):
		wnd = self._notifyWindow
		text = ""
		for i, choice in enumerate(response.choices):
			if self._wantAbort: break
			text += choice.message.content
		block.responseText += text
		block.responseTerminated = True


class TextToSpeechThread(threading.Thread):

	def __init__(self, notifyWindow, text):
		threading.Thread.__init__(self)
		self._notifyWindow = notifyWindow
		self._text = text
		self._voice = notifyWindow.conf["TTSVoice"]
		self._model = notifyWindow.conf["TTSModel"]
		self._wantAbort = 0

	def run(self):
		wnd = self._notifyWindow
		client = wnd.client
		try:
			if os.path.exists(TTS_FILE_NAME):
				os.remove(TTS_FILE_NAME)
			response = client.audio.speech.create(
				model=self._model,
				voice=self._voice,
				input=self._text,
				response_format="mp3"
			)
			response.stream_to_file(
				TTS_FILE_NAME,
			)
		except BaseException as err:
			wx.PostEvent(self._notifyWindow, ResultEvent(repr(err)))
			return
		wx.PostEvent(self._notifyWindow, ResultEvent(response))

	def abort(self):
		self._wantAbort = True
		self._recording = False


class TextSegment:

	previous= None
	next= None
	originalText = ""
	start = 0
	end = 0
	owner = None

	def __init__(self, control, text, owner):
		self.control = control
		self.originalText = text
		self.owner = owner

		# Management of segment chain in Control
		if not hasattr (control, "lastSegment") or control.lastSegment is None:
			# The linked list is not yet initialized
			control.firstSegment = self
			control.lastSegment = self
		else:
			# Add the segment to the end of the list
			control.lastSegment.next = self
			self.previous = control.lastSegment
			control.lastSegment  = self

		# Save current position
		p = control.GetInsertionPoint()

		# Add the text at the end of the control
		control.SetInsertionPointEnd()
		self.start = control.GetInsertionPoint()
		control.AppendText(text)
		self.end = control.GetInsertionPoint()

		# Restore the previously saved position
		control.SetInsertionPoint(p)

	def appendText (self, text):
		# Save current position
		p = self.control.GetInsertionPoint()

		# Add the text at the end of the segment
		self.control.AppendText(text)
		self.end = self.control.GetInsertionPoint()

		# Restore the previously saved position
		self.control.SetInsertionPoint(p)

	def getText (self):
		return self.control.GetRange (self.start, self.end)

	@staticmethod
	def getCurrentSegment(control):
		# Get the current position
		p = control.GetInsertionPoint()

		if not hasattr (control, "firstSegment"):
			return None
		segment = control.firstSegment

		# Iterate through the segments
		while segment != None:
			if segment.start <= p and segment.end > p:
				# The current segment is found
				return segment
			segment = segment.next

		# No segment found, probably the insertion point is at the end of the text.
		return control.lastSegment

	def delete(self):
		# Remove the text from the control
		self.control.Remove (self.start, self.end)

		# Remove the segment from the linked list
		if self.previous != None:
			self.previous.next = self.next
		else:
			# This is the first segment
			self.control.firstSegment = self.next

		if self.next != None:
			self.next.previous = self.previous
		else:
			# This is the last segment
			self.control.lastSegment = self.previous

		# Update the start and end positions of all following segments
		segment = self.next
		while segment != None:
			segment.start -= (self.end - self.start)
			segment.end -= (self.end - self.start)
			segment = segment.next


class HistoryBlock():
	previous = None
	next = None
	contextPromp = ""
	prompt = ""
	response = {}
	responseText = ""
	segmentBreakLine = None
	segmentPromptLabel = None
	segmentPrompt = None
	segmentResponseLabel = None
	segmentResponse = None
	lastLen = 0
	model = ""
	temperature = 0
	topP = 0
	displayHeader = True
	focused = False
	responseTerminated = False
	pathList = None


class OpenAIDlg(wx.Dialog):

	def __init__(
		self,
		parent,
		client,
		conf,
		title=None,
		pathList=None
	):
		if not client or not conf:
			return
		self.client = client
		self.conf = conf
		self.data = self.loadData()
		self._orig_data = self.data.copy() if isinstance(self.data, dict) else None
		self._historyPath = None
		self.blocks = []
		self.pathList = []
		if pathList:
			for path in pathList:
				if isinstance(path, ImageFile):
					self.pathList.append(path)
				elif isinstance(path, str):
					if not os.path.exists(path):
						continue
					self.pathList.append(ImageFile(path))
				elif isinstance(path, tuple) and len(path) == 2:
					location, name = path
					if not os.path.exists(location):
						continue
					self.pathList.append(
						ImageFile(
							location,
							name=name
						)
					)
				else:
					raise ValueError(f"Invalid path: {path}")
		self.previousPrompt = None
		self._lastSystem = None
		self._model_names = [model.name for model in MODELS]
		if self.conf["saveSystem"]:
			# If the user has chosen to save the system prompt, use the last system prompt used by the user as the default value, otherwise use the default system prompt.
			if "system" in self.data:
				self._lastSystem = self.data["system"]
			else:
				self._lastSystem = DEFAULT_SYSTEM_PROMPT
		else:
			# removes the system entry from data so that the last system prompt is not remembered when the user unchecks the save system prompt checkbox.
			self.data.pop("system", None)
		if not title:
			title = "Open AI - %s" % (
				_("organization") if conf["use_org"] else _("personal")
			)
		super().__init__(parent, title=title)

		self.conversationCheckBox = wx.CheckBox(
			parent=self,
			label=_("Conversati&on mode")
		)
		self.conversationCheckBox.SetValue(conf["conversationMode"])
		systemLabel = wx.StaticText(
			parent=self,
			label=_("S&ystem:")
		)
		self.systemText = wx.TextCtrl(
			parent=self,
			size=(550, -1),
			style=wx.TE_MULTILINE,
		)
		# Adds event handler to reset the system prompt to the default value when the user opens the context menu on the system prompt.
		self.systemText.Bind(wx.EVT_CONTEXT_MENU, self.onSystemContextMenu)
		# If the system prompt has been defined by the user, use it as the default value, otherwise use the default system prompt.
		if conf["saveSystem"]:
			self.systemText.SetValue(self._lastSystem)
		else:
			self.systemText.SetValue(DEFAULT_SYSTEM_PROMPT)

		historyLabel = wx.StaticText(
			parent=self,
			label=_("&History:")
		)
		self.historyText = wx.TextCtrl(
			parent=self,
			style=wx.TE_MULTILINE|wx.TE_READONLY,
			size=(550, -1)
		)
		self.historyText.Bind(wx.EVT_CONTEXT_MENU, self.onHistoryContextMenu)

		promptLabel = wx.StaticText(
			parent=self,
			label=_("&Prompt:")
		)
		self.promptText = wx.TextCtrl(
			parent=self,
			size=(550, -1),
			style=wx.TE_MULTILINE,
		)
		self.promptText.Bind(wx.EVT_CONTEXT_MENU, self.onPromptContextMenu)

		self.imageListLabel = wx.StaticText(
			parent=self,
			# Translators: This is a label for a list of images attached to the prompt.
			label=_("Attached ima&ges:")
		)
		self.imageListCtrl = wx.ListCtrl(
			parent=self,
			style=wx.LC_REPORT | wx.LC_HRULES | wx.LC_VRULES
		)
		self.imageListCtrl.InsertColumn(0, _("name"))
		self.imageListCtrl.InsertColumn(1, _("path"))
		self.imageListCtrl.InsertColumn(2, _("size"))
		self.imageListCtrl.InsertColumn(3, _("Dimensions"))
		self.imageListCtrl.InsertColumn(4, _("description"))
		self.imageListCtrl.SetColumnWidth(0, 100)
		self.imageListCtrl.SetColumnWidth(1, 200)
		self.imageListCtrl.SetColumnWidth(2, 100)
		self.imageListCtrl.SetColumnWidth(3, 100)
		self.imageListCtrl.SetColumnWidth(4, 200)
		self.imageListCtrl.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self.onImageListContextMenu)
		self.imageListCtrl.Bind(wx.EVT_KEY_DOWN, self.onImageListKeyDown)
		self.imageListCtrl.Bind(wx.EVT_CONTEXT_MENU, self.onImageListContextMenu)
		self.imageListCtrl.Bind(wx.EVT_RIGHT_UP, self.onImageListContextMenu)

		if self.pathList:
			self.promptText.SetValue(
				self.getDefaultImageDescriptionsPrompt()
			)
		self.updateImageList()
		modelsLabel = wx.StaticText(
			parent=self,
			label=_("&Model:")
		)
		models = [str(model) for model in MODELS]
		self.modelListBox = wx.ListBox(
			parent=self,
			choices=models,
			style=wx.LB_SINGLE | wx.LB_HSCROLL | wx.LB_NEEDED_SB
		)
		model = MODEL_VISION if self.pathList else conf["model"]
		idx = list(self._model_names).index(model) if model in self._model_names else 0
		self.modelListBox.SetSelection(idx)
		self.modelListBox.Bind(wx.EVT_LISTBOX, self.onModelChange)

		maxTokensLabel = wx.StaticText(
			parent=self,
			label=_("Maximum to&kens for the completion:")
		)
		self.maxTokens = wx.SpinCtrl(
			parent=self,
			min=0
		)

		if conf["advancedMode"]:
			temperatureLabel = wx.StaticText(
				parent=self,
				label=_("&Temperature:")
			)
			self.temperature = wx.SpinCtrl(
				parent=self
			)

			topPLabel = wx.StaticText(
				parent=self,
				label=_("Probability &mass (Top P):")
			)
			self.topP = wx.SpinCtrl(
				parent=self,
				min=TOP_P_MIN,
				max=TOP_P_MAX,
				initial=conf["topP"]
			)

			self.streamModeCheckBox = wx.CheckBox(
				parent=self,
				label=_("Stream mode")
			)
			self.streamModeCheckBox.SetValue(conf["stream"])

			self.debugModeCheckBox = wx.CheckBox(
				parent=self,
				label=_("&Debug mode")
			)
			self.debugModeCheckBox.SetValue(conf["debug"])

		self.onModelChange(None)
		sizer1 = wx.BoxSizer(wx.VERTICAL)
		sizer1.Add(self.conversationCheckBox, 0, wx.ALL, 5)
		sizer1.Add(systemLabel, 0, wx.ALL, 5)
		sizer1.Add(self.systemText, 0, wx.ALL, 5)
		sizer1.Add(historyLabel, 0, wx.ALL, 5)
		sizer1.Add(self.historyText, 0, wx.ALL, 5)
		sizer1.Add(promptLabel, 0, wx.ALL, 5)
		sizer1.Add(self.promptText, 0, wx.ALL, 5)
		sizer1.Add(self.imageListLabel, 0, wx.ALL, 5)
		sizer1.Add(self.imageListCtrl, 0, wx.ALL, 5)
		sizer1.Add(modelsLabel, 0, wx.ALL, 5)
		sizer1.Add(self.modelListBox, 0, wx.ALL, 5)
		sizer1.Add(maxTokensLabel, 0, wx.ALL, 5)
		sizer1.Add(self.maxTokens, 0, wx.ALL, 5)
		if conf["advancedMode"]:
			sizer1.Add(temperatureLabel, 0, wx.ALL, 5)
			sizer1.Add(self.temperature, 0, wx.ALL, 5)
			sizer1.Add(topPLabel, 0, wx.ALL, 5)
			sizer1.Add(self.topP, 0, wx.ALL, 5)
			sizer1.Add(self.streamModeCheckBox, 0, wx.ALL, 5)
			sizer1.Add(self.debugModeCheckBox, 0, wx.ALL, 5)

		self.recordBtn = wx.Button(
			parent=self,
			label=_("Start &recording") + " (Ctrl+R)"
		)
		self.recordBtn.Bind(wx.EVT_BUTTON, self.onRecord)
		self.recordBtn.SetToolTip(_("Record audio from microphone"))

		self.transcribeFromFileBtn = wx.Button(
			parent=self,
			label=_("Transcribe from &audio file") + " (Ctrl+Shift+R)"
		)
		self.transcribeFromFileBtn.Bind(wx.EVT_BUTTON, self.onRecordFromFilePath)
		self.transcribeFromFileBtn.SetToolTip(_("Transcribe audio from a file path"))

		self.imageDescriptionBtn = wx.Button(
			parent=self,
			label=_("&Image description")
		)
		self.imageDescriptionBtn.Bind(wx.EVT_BUTTON, self.onImageDescription)
		self.imageDescriptionBtn.SetToolTip(_("Describe an image from a file path or an URL"))

		self.TTSBtn = wx.Button(
			parent=self,
			label=_("&Vocalize the prompt") + " (Ctrl+T)"
		)
		self.TTSBtn.Bind(wx.EVT_BUTTON, self.onTextToSpeech)

		sizer2 = wx.BoxSizer(wx.HORIZONTAL)
		sizer2.Add(self.recordBtn, 0, wx.ALL, 5)
		sizer2.Add(self.imageDescriptionBtn, 0, wx.ALL, 5)
		sizer2.Add(self.transcribeFromFileBtn, 0, wx.ALL, 5)
		sizer2.Add(self.TTSBtn, 0, wx.ALL, 5)

		self.okBtn = wx.Button(
			parent=self,
			id=wx.ID_OK
		)
		self.okBtn.Bind(wx.EVT_BUTTON, self.onOk)
		self.okBtn.SetDefault()

		self.cancelBtn = wx.Button(
			parent=self,
			id=wx.ID_CANCEL
		)
		self.cancelBtn.Bind(wx.EVT_BUTTON, self.onCancel)

		sizer3 = wx.BoxSizer(wx.HORIZONTAL)
		sizer3.Add(self.okBtn, 0, wx.ALL, 5)
		sizer3.Add(self.cancelBtn, 0, wx.ALL, 5)

		sizer4 = wx.BoxSizer(wx.VERTICAL)
		sizer4.Add(sizer1, 0, wx.ALL, 5)
		sizer4.Add(sizer2, 0, wx.ALL, 5)
		sizer4.Add(sizer3, 0, wx.ALL, 5)

		self.SetSizer(sizer4)
		self.SetAutoLayout(True)
		sizer4.Fit(self)
		self.Layout()
		self.Center()
		self.SetSize((600, 600))
		self.SetMinSize((600, 600))

		self.addShortcuts()
		self.promptText.SetFocus()
		EVT_RESULT(self, self.OnResult)
		self.worker = None
		self.firstBlock = None
		self.lastBlock = None
		self.timer = wx.Timer(self)
		self.Bind(wx.EVT_TIMER, self.onTimer, self.timer)
		self.timer.Start (100)
		self.Bind(wx.EVT_CHAR_HOOK, self.onCharHook)
		self.Bind(wx.EVT_CLOSE, self.onCancel)

	def getDefaultImageDescriptionsPrompt(self):
		if self.conf["images"]["useCustomPrompt"]:
			return self.conf["images"]["customPromptText"]
		return _("Describe the images in as much detail as possible.")

	def loadData(self):
		if not os.path.exists(DATA_JSON_FP):
			return {}
		try:
			with open(DATA_JSON_FP, 'r') as f :
				return json.loads(f.read())
		except BaseException as err:
			log.error(err)

	def saveData(self, force=False):
		if not force and self.data == self._orig_data:
			return
		with open(DATA_JSON_FP, "w") as f:
			f.write(json.dumps(self.data))

	def getCurrentModel(self):
		return MODELS[self.modelListBox.GetSelection()]

	def onResetSystemPrompt(self, event):
		self.systemText.SetValue(DEFAULT_SYSTEM_PROMPT)
	def onDelete(self, event):
		self.systemText.SetValue('')

	def addStandardMenuOptions(self, menu):
		menu.Append(wx.ID_UNDO)
		menu.Append(wx.ID_REDO)
		menu.AppendSeparator()
		menu.Append(wx.ID_CUT)
		menu.Append(wx.ID_COPY)
		menu.Append(wx.ID_PASTE)
		menu.Append(wx.ID_DELETE)
		menu.AppendSeparator()
		menu.Append(wx.ID_SELECTALL)
		self.Bind(wx.EVT_MENU, self.onDelete, id=wx.ID_DELETE)

	def onModelChange(self, evt):
		model = self.getCurrentModel()
		self.maxTokens.SetRange(
			0,
			model.maxOutputToken if model.maxOutputToken > 1 else model.contextWindow
		)
		defaultMaxOutputToken = 512
		key_maxTokens = "maxTokens_%s" % model.name
		if (
			key_maxTokens in self.data
			and isinstance(self.data[key_maxTokens], int)
			and self.data[key_maxTokens] > 0
		):
			defaultMaxOutputToken = self.data[key_maxTokens]
		else:
			defaultMaxOutputToken = model.maxOutputToken // 2
			if defaultMaxOutputToken < 1:
				defaultMaxOutputToken  = model.contextWindow // 2
		if defaultMaxOutputToken < 1:
			defaultMaxOutputToken = 1024
		self.maxTokens.SetValue(defaultMaxOutputToken)
		if self.conf["advancedMode"]:
			self.temperature.SetRange(0, model.maxTemperature * 100)
			key_temperature = "temperature_%s" % model.name
			if key_temperature in self.data:
				self.temperature.SetValue(self.data[key_temperature])
			else:
				self.temperature.SetValue(model.defaultTemperature * 100) 

	def onOk(self, evt):
		if not self.promptText.GetValue().strip() and not self.pathList:
			self.promptText.SetFocus()
			return
		if self.worker:
			return
		model = self.getCurrentModel()
		if not model:
			gui.messageBox(
				_("Please select a model."),
				"Open AI",
				wx.OK | wx.ICON_ERROR
			)
			return
		if (
			model.name == MODEL_VISION
			and not self.conversationCheckBox.IsChecked()
			and not self.pathList
		):
			gui.messageBox(
				_("No image provided. Please use the Image Description button and select one or more images. Otherwise, please select another model."),
				"Open AI",
				wx.OK | wx.ICON_ERROR
			)
			return
		if model.name != MODEL_VISION and self.pathList:
			gui.messageBox(
				_("This model does not support image description. Please select the %s model.") % MODEL_VISION,
				"Open AI",
				wx.OK | wx.ICON_ERROR
			)
			return
		if (
			model.name == MODEL_VISION
			and not self.conf["images"]["resize"]
			and not self.conf["images"]["resizeInfoDisplayed"]
		):
			msg = _("Be aware that the add-on may auto-resize images before API submission to lower request sizes and costs. Adjust this feature in the Open AI settings if needed. This message won't show again.")
			gui.messageBox(
				msg,
				"Open AI",
				wx.OK | wx.ICON_INFORMATION
			)
			self.conf["images"]["resizeInfoDisplayed"] = True
		system = self.systemText.GetValue().strip()
		if self.conf["saveSystem"] and system != self._lastSystem:
			self.data["system"] = system
			self._lastSystem = system
		winsound.PlaySound(f"{ADDON_DIR}/sounds/progress.wav", winsound.SND_ASYNC|winsound.SND_LOOP)
		self.disableButtons()
		self.historyText.SetFocus()
		self.stopRequest = threading.Event()
		self.worker = CompletionThread(self)
		self.worker.start()

	def onCancel(self, evt):
		self.saveData()
		if self.worker:
			self.worker.abort()
			self.worker = None
			winsound.PlaySound(None, winsound.SND_ASYNC)
		self.timer.Stop()
		self.Destroy()

	def OnResult(self, event):
		winsound.PlaySound(None, winsound.SND_ASYNC)
		self.enableButtons()
		self.worker = None
		if not event.data:
			return

		if isinstance(event.data, openai.types.chat.chat_completion.Choice):
			historyBlock = HistoryBlock()
			historyBlock.system = self.systemText.GetValue().strip()
			historyBlock.prompt = self.promptText.GetValue().strip()
			model = self.getCurrentModel()
			historyBlock.model = model.name
			if self.conf["advancedMode"]:
				historyBlock.temperature = self.temperature.GetValue() / 100
				historyBlock.topP = self.topP.GetValue() / 100
			else:
				historyBlock.temperature = model.defaultTemperature
				historyBlock.topP = self.conf["topP"] / 100
			historyBlock.maxTokens = self.maxTokens.GetValue()
			historyBlock.n = 1 # self.n.GetValue()
			historyBlock.response = event.data
			historyBlock.responseText = event.data.message.content
			historyBlock.responseTerminated = True
			if self.lastBlock is None:
				self.firstBlock = self.lastBlock = historyBlock
			else:
				self.lastBlock.next = historyBlock
				historyBlock.previous = self.lastBlock
				self.lastBlock = historyBlock
			self.previousPrompt = self.promptText.GetValue()
			self.promptText.Clear()
			return

		if isinstance(event.data, openai.types.audio.transcription.Transcription):
			self.promptText.AppendText(event.data.text)
			self.promptText.SetFocus()
			self.promptText.SetInsertionPointEnd()
			self.message(
				_("Insertion of: %s") % event.data.text,
				True
			)
			return

		if isinstance(event.data, openai._base_client.HttpxBinaryResponseContent):
			if os.path.exists(TTS_FILE_NAME):
				os.startfile(TTS_FILE_NAME)
			return
		errMsg = repr(event.data)
		gui.messageBox(
			errMsg,
			_("Open AI error"),
			wx.OK | wx.ICON_ERROR
		)
		if "model's maximum context length is " in errMsg:
			self.modelListBox.SetFocus()
		else:
			self.promptText.SetFocus()

	def onCharHook(self, evt):
		if self.conf["blockEscapeKey"] and evt.GetKeyCode() == wx.WXK_ESCAPE:
			self.message(_("Press Alt+F4 to close the dialog"))
		else:
			evt.Skip()

	def onTimer(self, event):
		if self.lastBlock is not None:
			block = self.lastBlock
			if block.displayHeader:
				if block != self.firstBlock:
					block.previous.segmentBreakLine = TextSegment(self.historyText, "\n", block)
				block.segmentPromptLabel = TextSegment(self.historyText, _("User:") + ' ', block)
				block.segmentPrompt = TextSegment(self.historyText, block.prompt + "\n", block)
				block.segmentResponseLabel = TextSegment(self.historyText, _("Assistant:") + ' ', block)
				block.displayHeader = False
			l = len(block.responseText)
			if block.lastLen == 0 and l > 0:
				self.historyText.SetInsertionPointEnd()
				block.responseText = block.responseText.lstrip()
				l = len(block.responseText)
			if l > block.lastLen:
				newText = block.responseText[block.lastLen:]
				block.lastLen = l
				if block.segmentResponse is None:
					block.segmentResponse = TextSegment(self.historyText, newText, block)
				else:
					block.segmentResponse.appendText (newText)
			if not block.focused and (block.responseTerminated or "\n" in block.responseText or len (block.responseText) > 180):
				self.historyText.SetFocus ()
				block.focused = True

	def addEntry(self, accelEntries, modifiers, key, func):
		id_ = wx.Window.NewControlId()
		self.Bind(wx.EVT_MENU, func, id=id_)
		accelEntries.append ( (modifiers, key, id_))

	def addShortcuts(self):
		self.historyText.Bind(wx.EVT_TEXT_COPY, self.onCopyMessage)

		accelEntries  = []
		self.addEntry(accelEntries, wx.ACCEL_NORMAL, ord("M"), self.onCurrentMessage)
		self.addEntry(accelEntries, wx.ACCEL_NORMAL, ord("J"), self.onPreviousMessage)
		self.addEntry(accelEntries, wx.ACCEL_CTRL + wx.ACCEL_SHIFT, wx.WXK_UP, self.onPreviousMessage)
		self.addEntry(accelEntries, wx.ACCEL_NORMAL, ord("K"), self.onNextMessage)
		self.addEntry(accelEntries, wx.ACCEL_CTRL + wx.ACCEL_SHIFT, wx.WXK_DOWN, self.onNextMessage)
		self.addEntry(accelEntries, wx.ACCEL_CTRL + wx.ACCEL_SHIFT, ord("C"), lambda evt: self.onCopyMessage(evt, True))
		self.addEntry(accelEntries, wx.ACCEL_CTRL, ord("E"), self.onEditBlock)
		self.addEntry(accelEntries, wx.ACCEL_CTRL, ord("D"), self.onDeleteBlock)
		self.addEntry(accelEntries, wx.ACCEL_CTRL, ord("S"), self.onSaveHistory)
		self.addEntry(accelEntries, wx.ACCEL_NORMAL, wx.WXK_SPACE, lambda evt: self.onWebviewMessage(evt, True))
		self.addEntry(accelEntries, wx.ACCEL_SHIFT, wx.WXK_SPACE, lambda evt: self.onWebviewMessage(evt, False))
		self.addEntry(accelEntries, wx.ACCEL_ALT, wx.WXK_LEFT, self.onCopyResponseToSystem)
		self.addEntry(accelEntries, wx.ACCEL_ALT, wx.WXK_RIGHT, self.onCopyPromptToPrompt)
		accelTable = wx.AcceleratorTable(accelEntries)
		self.historyText.SetAcceleratorTable(accelTable)

		accelEntries  = []
		self.addEntry (accelEntries, wx.ACCEL_CTRL, wx.WXK_UP, self.onPreviousPrompt)
		accelTable = wx.AcceleratorTable(accelEntries)
		self.promptText.SetAcceleratorTable(accelTable)

		accelEntries  = []
		self.addEntry(accelEntries, wx.ACCEL_CTRL, ord("r"), self.onRecord)
		self.addEntry(accelEntries, wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord("r"), self.onRecordFromFilePath)
		self.addEntry(accelEntries, wx.ACCEL_CTRL, ord("i"), self.onImageDescriptionFromFilePath)
		self.addEntry(accelEntries, wx.ACCEL_CTRL, ord("u"), self.onImageDescriptionFromURL)
		self.addEntry(accelEntries, wx.ACCEL_CTRL, ord("t"), self.onTextToSpeech)
		accelTable = wx.AcceleratorTable(accelEntries)
		self.SetAcceleratorTable(accelTable)

	def getImages(
		self,
		pathList: list = None
	) -> list:
		conf = self.conf
		if not pathList:
			pathList = self.pathList
		images = []
		for imageFile in pathList:
			path = imageFile.path
			log.debug(f"Processing {path}")
			if imageFile.type == ImageFileTypes.IMAGE_URL:
				images.append({"type": "image_url", "image_url": {"url": path}})
			elif imageFile.type == ImageFileTypes.IMAGE_LOCAL:
				if conf["images"]["resize"]:
					path_resized_image = os.path.join(DATA_DIR, "last_resized.jpg")
					resize_image(
						path,
						max_width=conf["images"]["maxWidth"],
						max_height=conf["images"]["maxHeight"],
						quality=conf["images"]["quality"],
						target=path_resized_image
					)
					path = path_resized_image
				base64_image = encode_image(path)
				format = path.split(".")[-1]
				mime_type = f"image/{format}"
				images.append({
					"type": "image_url",
					"image_url": {
						"url": f"data:{mime_type};base64,{base64_image}"
					}
				})
			else:
				raise ValueError(f"Invalid image type for {path}")
				break
		return images

	def getMessages(
		self,
		messages: list
	) -> list:
		model = self.getCurrentModel()
		if not self.conversationCheckBox.IsChecked():
			return messages
		block = self.firstBlock
		while block is not None:
			content = []
			if block.pathList:
				content.extend(self.getImages(block.pathList))
			if block.prompt:
				content.append({
					"type": "text",
					"text": block.prompt
				})
			if content:
				messages.append({
					"role": "user",
					"content": content
				})
			if block.responseText:
				messages.append({"role": "system", "content": block.responseText})
			block = block.next

	def onPreviousPrompt(self, event):
		value = self.previousPrompt
		if value:
			self.promptText.SetValue(value)

	def onPreviousMessage(self, evt):
		segment = TextSegment.getCurrentSegment(self.historyText)
		if segment is None:
			return
		block = segment.owner
		if segment == block.segmentPromptLabel or segment == block.segmentPrompt:
			prev = block.previous
			if prev is None:
				self.message(_("Begin"))
				return
			start = prev.segmentResponse.start
			text = prev.segmentResponse.getText ()
			label = prev.segmentResponseLabel.getText ()
		elif segment == block.segmentResponseLabel or segment == block.segmentResponse or segment == block.segmentBreakLine:
			start = block.segmentPrompt.start
			text = block.segmentPrompt.getText ()
			label = block.segmentPromptLabel.getText ()
		self.historyText.SetInsertionPoint(start)
		self.message(label + text)

	def onNextMessage(self, evt):
		segment = TextSegment.getCurrentSegment (self.historyText)
		if segment is None:
			return
		block = segment.owner
		if segment == block.segmentResponseLabel or segment == block.segmentResponse or segment == block.segmentBreakLine:
			next = block.next
			if next is None:
				self.message(_("End"))
				return
			start = next.segmentPrompt.start
			text = next.segmentPrompt.getText ()
			label = next.segmentPromptLabel.getText ()
		elif segment == block.segmentPromptLabel or segment == block.segmentPrompt:
			start = block.segmentResponse.start
			text = block.segmentResponse.getText ()
			label = block.segmentResponseLabel.getText ()
		self.historyText.SetInsertionPoint(start)
		self.message(label + text)

	def onCurrentMessage(self, evt):
		"""Say the current message"""
		segment = TextSegment.getCurrentSegment (self.historyText)
		if segment is None:
			return
		block = segment.owner
		if segment == block.segmentPromptLabel or segment == block.segmentPrompt:
			text = block.segmentPrompt.getText ()
		elif segment == block.segmentResponseLabel or segment == block.segmentResponse:
			text = block.segmentResponse.getText ()
		self.message(text)


	def onEditBlock (self, evt):
		segment = TextSegment.getCurrentSegment (self.historyText)
		if segment is None:
			return
		block = segment.owner
		self.systemText.SetValue(block.system)
		self.promptText.SetValue (block.userPrompt)
		self.promptText.SetFocus ()

	def onCopyResponseToSystem (self, evt):
		segment = TextSegment.getCurrentSegment(self.historyText)
		if segment is None:
			return
		block = segment.owner
		text = block.segmentResponse.getText ()
		self.systemText.SetValue(text)
		self.message(_("Response copied to system: %s") % text)

	def onCopyPromptToPrompt(self, evt):
		segment = TextSegment.getCurrentSegment(self.historyText)
		if segment is None:
			return
		block = segment.owner
		self.promptText.SetValue (block.segmentPrompt.getText ())
		self.promptText.SetFocus ()
		self.message(_("Compied to prompt"))

	def onCopyMessage(self, evt, isHtml=False):
		text = self.historyText.GetStringSelection()
		msg = _("Copy")
		if not text:
			segment = TextSegment.getCurrentSegment (self.historyText)
			if segment is None:
				return
			block = segment.owner
			if segment == block.segmentPromptLabel or segment == block.segmentPrompt:
				text = block.segmentPrompt.getText ()
				msg = _("Copy prompt")
			elif segment == block.segmentResponseLabel or segment == block.segmentResponse:
				text = block.segmentResponse.getText()
				msg = _("Copy response")
		if isHtml:
			text = markdown2.markdown(
				text,
				extras=["fenced-code-blocks", "footnotes", "header-ids", "spoiler", "strike", "tables", "task_list", "underline", "wiki-tables"]
			)
			copyToClipAsHTML(text)
			msg += ' ' + _("as formatted HTML")
		else:
			api.copyToClip(text)
		self.message(msg)

	def onDeleteBlock(self, evt):
		segment = TextSegment.getCurrentSegment (self.historyText)
		if segment is None:
			return
		block = segment.owner

		if block.segmentBreakLine  is not None:
			block.segmentBreakLine.delete()
		block.segmentPromptLabel.delete()
		block.segmentPrompt.delete()
		block.segmentResponseLabel.delete()
		block.segmentResponse.delete()

		if block.previous is not None:
			block.previous.next = block.next
		else:
			self.firstBlock = block.next
		if block.next is not None:
			block.next.previous = block.previous
		else:
			self.lastBlock = block.previous
		self.message(_("Block deleted"))

	def onWebviewMessage(self, evt, isHtml=False):
		segment = TextSegment.getCurrentSegment (self.historyText)
		if segment is None:
			return
		block = segment.owner
		if segment == block.segmentPromptLabel or segment == block.segmentPrompt:
			text = block.segmentPrompt.getText ()
		elif segment == block.segmentResponseLabel or segment == block.segmentResponse:
			text = block.segmentResponse.getText ()
		ui.browseableMessage(
			markdown2.markdown(
				text,
				extras=["fenced-code-blocks", "footnotes", "header-ids", "spoiler", "strike", "tables", "task_list", "underline", "wiki-tables"]
			),
			title="Open AI",
			isHtml=isHtml
		)

	def onSaveHistory(self, evt):
		"""
		Save the history to a file.
		"""
		path = None
		if self._historyPath and os.path.exists(self._historyPath):
			path = self._historyPath
		else:
			now = datetime.datetime.now()
			now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
			defaultFile = "openai_history_%s.txt" % now_str
			dlg = wx.FileDialog(
				None,
				message=_("Save history"),
				defaultFile=defaultFile,
				wildcard=_("Text file") + " (*.txt)|*.txt",
				style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
			)
			if dlg.ShowModal() != wx.ID_OK:
				return
			path = dlg.GetPath()
		if not path:
			return
		self._historyPath = path
		with open(path, "w", encoding="utf-8") as f:
			f.write(self.historyText.GetValue())
		self.message(_("History saved"))

	def onSystemContextMenu(self, event):
		menu = wx.Menu()
		item_id = wx.NewIdRef()
		resetItem = menu.Append(item_id, _("Reset to default"))
		self.Bind(wx.EVT_MENU, self.onResetSystemPrompt, id=item_id)
		menu.AppendSeparator()
		self.addStandardMenuOptions(menu)
		self.systemText.PopupMenu(menu)
		menu.Destroy()

	def onHistoryContextMenu(self, evt):
		menu = wx.Menu()
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Show message in web view as formatted HTML") + " (Space)")
		self.Bind(wx.EVT_MENU, lambda evt: self.onWebviewMessage(evt, True), id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Show message in web view as HTML source") + " (Shift+Space)")
		self.Bind(wx.EVT_MENU, lambda evt: self.onWebviewMessage(evt, False), id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Copy message as plain text") + " (Ctrl+C)")
		self.Bind(wx.EVT_MENU, lambda evt: self.onCopyMessage(evt, False), id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Copy message as formatted HTML") + " (Ctrl+Shift+C)")
		self.Bind(wx.EVT_MENU, lambda evt: self.onCopyMessage(evt, True), id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Copy response to system") + " (Alt+Left)")
		self.Bind(wx.EVT_MENU, self.onCopyResponseToSystem, id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Copy prompt to prompt") + " (Alt+Right)")
		self.Bind(wx.EVT_MENU, self.onCopyPromptToPrompt, id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Delete block") + " (Ctrl+D)")
		self.Bind(wx.EVT_MENU, self.onDeleteBlock, id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Save history as text file") + " (Ctrl+S)")
		self.Bind(wx.EVT_MENU, self.onSaveHistory, id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Say message") + " (M)")
		self.Bind(wx.EVT_MENU, self.onCurrentMessage, id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Move to previous message") + " (j)")
		self.Bind(wx.EVT_MENU, self.onPreviousMessage, id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Move to next message") + " (k)")
		self.Bind(wx.EVT_MENU, self.onNextMessage, id=item_id)
		menu.AppendSeparator()
		self.addStandardMenuOptions(menu)
		self.historyText.PopupMenu(menu)
		menu.Destroy()

	def onPromptContextMenu(self, evt):
		menu = wx.Menu()
		if self.previousPrompt:
			item_id = wx.NewIdRef()
			menu.Append(item_id, _("Insert previous prompt") + " (Ctrl+Up)")
			self.Bind(wx.EVT_MENU, self.onPreviousPrompt, id=item_id)
			menu.AppendSeparator()
		self.addStandardMenuOptions(menu)
		self.promptText.PopupMenu(menu)
		menu.Destroy()

	def message(self, msg, onlySpeech=False):
		func = ui.message if not onlySpeech else speech.speakMessage
		queueHandler.queueFunction(queueHandler.eventQueue, func, msg)

	def onImageDescription(self, evt):
		"""
		Display a menu to select the source of the image.
		"""
		menu = wx.Menu()
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("From f&ile path...") + " (Ctrl+I)")
		self.Bind(wx.EVT_MENU, self.onImageDescriptionFromFilePath, id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("From &URL...") + " (Ctrl+U)")
		self.Bind(wx.EVT_MENU, self.onImageDescriptionFromURL, id=item_id)
		self.PopupMenu(menu)
		menu.Destroy()

	def onImageListKeyDown(self, evt):
		key_code = evt.GetKeyCode()
		if key_code == wx.WXK_DELETE:
			self.onRemoveSelectedImages(evt)
		elif key_code == ord('A') and evt.ControlDown():
			self.onImageListSelectAll(evt)
		evt.Skip()

	def onImageListContextMenu(self, evt):
		"""
		Display a menu to manage the image list.
		"""
		menu = wx.Menu()
		if self.pathList:
			if self.imageListCtrl.GetItemCount() > 0 and self.imageListCtrl.GetSelectedItemCount() > 0:
				item_id = wx.NewIdRef()
				menu.Append(item_id, _("&Remove selected images") + " (Del)")
				self.Bind(wx.EVT_MENU, self.onRemoveSelectedImages, id=item_id)
			item_id = wx.NewIdRef()
			menu.Append(item_id, _("Remove &all images"))
			self.Bind(wx.EVT_MENU, self.onRemoveAllImages, id=item_id)
			menu.AppendSeparator()
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Add from f&ile path...") + " (Ctrl+I)")
		self.Bind(wx.EVT_MENU, self.onImageDescriptionFromFilePath, id=item_id)
		item_id = wx.NewIdRef()
		menu.Append(item_id, _("Add from &URL...") + " (Ctrl+U)")
		self.Bind(wx.EVT_MENU, self.onImageDescriptionFromURL, id=item_id)
		self.PopupMenu(menu)
		menu.Destroy()

	def onImageListSelectAll(self, evt):
		for i in range(self.imageListCtrl.GetItemCount()):
			self.imageListCtrl.Select(i)

	def onImageListChange(self, evt):
		"""
		Select the model for image description.
		"""
		self.modelListBox.SetSelection(
			self._model_names.index(MODEL_VISION)
		)
		self.imageListCtrl.SetSelection(evt.GetSelection())
		evt.Skip()

	def onRemoveSelectedImages(self, evt):
		if not self.pathList:
			return
		focused_item = self.imageListCtrl.GetFocusedItem()
		items_to_remove = []
		selectedItem = self.imageListCtrl.GetFirstSelected()
		while selectedItem != wx.NOT_FOUND:
			items_to_remove.append(selectedItem)
			selectedItem = self.imageListCtrl.GetNextSelected(selectedItem)

		if not items_to_remove:
			return
		self.pathList = [
			path for i, path in enumerate(self.pathList)
			if i not in items_to_remove
		]
		self.updateImageList()
		if focused_item == wx.NOT_FOUND:
			return
		if focused_item > self.imageListCtrl.GetItemCount() - 1:
			focused_item -= 1
		self.imageListCtrl.Focus(focused_item)
		self.imageListCtrl.Select(focused_item)

	def onRemoveAllImages(self, evt):
		self.pathList.clear()
		self.updateImageList()

	def imageExists(self, path, pathList=None):
		if not pathList:
			pathList = self.pathList
		for imageFile in pathList:
			if imageFile.path.lower() == path.lower():
				return True
		block = self.firstBlock
		while block is not None:
			if block.pathList:
				for imageFile in block.pathList:
					if imageFile.path.lower() == path.lower():
						return True
			block = block.next
		return False

	def updateImageList(self, focusPrompt=True):
		self.imageListCtrl.DeleteAllItems()
		if not self.pathList:
			self.imageListCtrl.Hide()
			self.imageListLabel.Hide()
			self.Layout()
			if focusPrompt:
				self.promptText.SetFocus()
			return
		for path in self.pathList:
			self.imageListCtrl.Append([
				path.name,
				path.path,
				path.size,
				f"{path.dimensions[0]}x{path.dimensions[1]}" if isinstance(path.dimensions, tuple) else "N/A",
				path.description or "N/A"
			])
		self.imageListLabel.Show()
		self.imageListCtrl.SetItemState(
			0,
			wx.LIST_STATE_FOCUSED,
			wx.LIST_STATE_FOCUSED
		)
		self.imageListCtrl.Show()
		self.Layout()

	def onImageDescriptionFromFilePath(self, evt):
		"""
		Open a file dialog to select one or more images.
		"""
		if not self.pathList:
			self.pathList = []
		dlg = wx.FileDialog(
			None,
			# Translators: This is a message displayed in a dialog to select one or more images.
			message=_("Select image files"),
			defaultFile="",
			wildcard=_("Image files") + " (*.png;*.jpeg;*.jpg;*.gif)|*.png;*.jpeg;*.jpg;*.gif",
			style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE
		)
		if dlg.ShowModal() != wx.ID_OK:
			return
		paths = dlg.GetPaths()
		if not paths:
			return
		for path in paths:
			if not self.imageExists(path):
				self.pathList.append(
					ImageFile(path)
				)
			else:
				gui.messageBox(
					# Translators: This message is displayed when the image has already been added.
					_("The following image has already been added and will be ignored:\n%s") % path,
					"Open AI",
					wx.OK | wx.ICON_ERROR
				)
		model_name = self.getCurrentModel().name
		if model_name != MODEL_VISION:
			self.modelListBox.SetSelection(
				self._model_names.index(MODEL_VISION)
			)
		if not self.promptText.GetValue().strip():
			self.promptText.SetValue(
				self.getDefaultImageDescriptionsPrompt()
			)
		self.updateImageList()

	def onImageDescriptionFromURL(self, evt):
		"""
		Open a dialog to enter an image URL.
		"""
		dlg = wx.TextEntryDialog(
			None,
			# Translators: This is a message displayed in a dialog to enter an image URL.
			message=_("Enter image URL"),
			caption="Open AI",
			style=wx.OK | wx.CANCEL
		)
		if dlg.ShowModal() != wx.ID_OK:
			return
		url = dlg.GetValue().strip()
		if not url:
			return
		url_pattern = re.compile(
			URL_PATTERN
		)
		if re.match(url_pattern, url) is None:
			gui.messageBox(
				_("Invalid URL, bad format."),
				"Open AI",
				wx.OK | wx.ICON_ERROR
			)
			return
		try:
			import urllib.request
			r = urllib.request.urlopen(url)
		except urllib.error.HTTPError as err:
			gui.messageBox(
				# Translators: This message is displayed when the image URL returns an HTTP error.
				_("HTTP error %s.") % err,
				"Open AI",
				wx.OK | wx.ICON_ERROR
			)
			return
		if not r.headers.get_content_type().startswith("image/"):
			gui.messageBox(
				# Translators: This message is displayed when the image URL does not point to an image.
				_("The URL does not point to an image."),
				"Open AI",
				wx.OK | wx.ICON_ERROR
			)
			return
		if not self.pathList:
			self.pathList = []
		description = []
		content_type = r.headers.get_content_type()
		if content_type:
			description.append(content_type)
		size = r.headers.get("Content-Length")
		if size and size.isdigit():
			size = int(size)
		if description:
			description = ", ".join(description)
		try:
			dimensions = get_image_dimensions(r)
		except BaseException as err:
			log.error(err)
			dimensions = None
			gui.messageBox(
				# Translators: This message is displayed when the add-on fails to get the image dimensions.
				_("Failed to get image dimensions. %s") % err,
				"Open AI",
				wx.OK | wx.ICON_ERROR
			)
			return
		self.pathList.append(
			ImageFile(
				url,
				description=description,
				size=size or -1,
				dimensions=dimensions
			)
		)
		self.modelListBox.SetSelection(
			self._model_names.index(MODEL_VISION)
		)
		if not self.promptText.GetValue().strip():
			self.promptText.SetValue(
				self.getDefaultImageDescriptionsPrompt()
			)
		self.updateImageList()

	def onRecord(self, evt):
		if self.worker:
			self.onStopRecord(evt)
			return
		self.disableButtons()
		self.recordBtn.SetLabel(_("Stop &recording") + " (Ctrl+R)")
		self.recordBtn.Bind(wx.EVT_BUTTON, self.onStopRecord)
		self.recordBtn.Enable()
		self.worker = RecordThread(
			self.client,
			self,
			conf=self.conf["audio"]
		)
		self.worker.start()

	def onRecordFromFilePath(self, evt):
		dlg = wx.FileDialog(
			None,
			message=_("Select audio file"),
			defaultFile="",
			wildcard=_("Audio files (*.mp3;*.mp4;*.mpeg;*.mpga;*.m4a;*.wav;*.webm)|*.mp3;*.mp4;*.mpeg;*.mpga;*.m4a;*.wav;*.webm"),
			style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
		)
		if dlg.ShowModal() != wx.ID_OK:
			return
		fileName = dlg.GetPath()
		self.message(_("Processing, please wait..."))
		winsound.PlaySound(f"{ADDON_DIR}/sounds/progress.wav", winsound.SND_ASYNC|winsound.SND_LOOP)
		self.disableButtons()
		self.historyText.SetFocus()
		self.worker = RecordThread(
			self.client,
			self,
			fileName
		)
		self.worker.start()

	def onTextToSpeech(self, evt):
		if not self.promptText.GetValue().strip():
			gui.messageBox(
				_("Please enter some text in the prompt field first."),
				"Open AI",
				wx.OK | wx.ICON_ERROR
			)
			self.promptText.SetFocus()
			return
		self.message(_("Processing, please wait..."))
		winsound.PlaySound(f"{ADDON_DIR}/sounds/progress.wav", winsound.SND_ASYNC|winsound.SND_LOOP)
		self.disableButtons()
		self.promptText.SetFocus()
		self.worker = TextToSpeechThread(self, self.promptText.GetValue())
		self.worker.start()

	def onStopRecord(self, evt):
		if self.worker:
			self.worker.stop()
			self.worker = None
			winsound.PlaySound(None, winsound.SND_ASYNC)
		self.recordBtn.SetLabel(
			_("Start &recording") + " (Ctrl+R)"
		)
		self.recordBtn.Bind(wx.EVT_BUTTON, self.onRecord)
		self.enableButtons()

	def disableButtons(self):
		self.okBtn.Disable()
		self.cancelBtn.Disable()
		self.recordBtn.Disable()
		self.transcribeFromFileBtn.Disable()
		self.imageDescriptionBtn.Disable()
		self.TTSBtn.Disable()
		self.modelListBox.Disable()
		self.maxTokens.Disable()
		self.conversationCheckBox.Disable()
		self.promptText.SetEditable(False)
		self.systemText.SetEditable(False)
		self.imageListCtrl.Disable()
		if self.conf["advancedMode"]:
			self.temperature.Disable()
			self.topP.Disable()
			self.streamModeCheckBox.Disable()
			self.debugModeCheckBox.Disable()

	def enableButtons(self):
		self.okBtn.Enable()
		self.cancelBtn.Enable()
		self.recordBtn.Enable()
		self.transcribeFromFileBtn.Enable()
		self.imageDescriptionBtn.Enable()
		self.TTSBtn.Enable()
		self.modelListBox.Enable()
		self.maxTokens.Enable()
		self.conversationCheckBox.Enable()
		self.promptText.SetEditable(True)
		self.systemText.SetEditable(True)
		self.imageListCtrl.Enable()
		if self.conf["advancedMode"]:
			self.temperature.Enable()
			self.topP.Enable()
			self.streamModeCheckBox.Enable()
			self.debugModeCheckBox.Enable()
		self.updateImageList(False)
