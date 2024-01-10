import datetime
from redact_text_w_openAI import OpenAIRedactor
import util
from baseview import ViewInterface, BaseView
import os
from dotenv import set_key
import npyscreen
import ui_const

MAX_TOKEN_LENGTHS = {
    "gpt-4": 8192, #8192 is the proper value, but this is unusable in the afternoons in CEST timezone, so we have a 
    "gpt-3.5-turbo": 4096
}     
SYSTEM_PROMPT_DEF = "You are a silent AI tool helping to format the long texts that were transcribed from speeches. " \
    "You format the text follows: you break up the text into paragraphs where new topic is discussed or where that would otherwise help easier reading, and correct the text for spelling errors, delete any repetitions. " \
    "You may receive the text in multiple batches. " \
    "Do not include your own text in the response, and use the original language of the text."

REQUEST_TIMEOUT = 600


MSG_REDACTIONCOMPLETE_EN = "Redaction complete!"
MSG_FILENOTEXIST_EN = "The file {} does not exist."
MSG_REDACTIONSTARTED_EN = "Redaction started ... A response could take up to " + str(round(REQUEST_TIMEOUT/60)) + " minutes per chunk. Number of chunks: {}"
MSG_CHUNKSTARTED_EN = "{}. chunk, started: {}. Config: {}"
MSG_REDACTEDCHUNKRESULT_EN = "[Redacted chunk: [{}]"
MSG_CHUNKPROCESSINGTIME_EN = "{}. chunk processing time: {}"

ratio_of_total_max_prompt = 0.5

class RedactorView(BaseView, ViewInterface):
    def __init__(self, form, api_key):
        super().__init__(form)
        self.form = form
        self.api_key = api_key 
        self.presenter = RedactorPresenter(self)

    def create(self):
        from npyscreen import ButtonPress
        from ui_const import NAME_REDACTBUTTON_EN
        self.form.redact_button = self.form.add(ButtonPress, name=NAME_REDACTBUTTON_EN, hidden=True, rely=7, relx=2, max_height=1)
        self.form.redact_button.whenPressed = self.on_redact
        self.redact_prompt = ""
        self.redact_prompt_button = RedactPromptButton(self.form)
        self.redact_prompt_button.create()
        self.redact_prompt_setter = SetRedactPrompt(self.form)
        self.redact_prompt_setter.create()
        
    
    def on_redact(self):
        self.presenter.handle_redaction()
        
    def update_visibility(self, visible: bool=None):
        if visible:
            self.form.redact_button.hidden = not visible
        else:            
            if self.form.output_file:
                self.form.redact_button.hidden = False
            else:
                self.form.redact_button.hidden = True
        self.form.redact_button.update()
        self.form.display()

class RedactorModel:
    def __init__(self, view: RedactorView):
        self.view = view
        
    def redact(self, output_file: str, apikey: str, current_model_config: str, token_max_length: int, ratio_of_total_max_prompt: float) -> str:
        if not os.path.exists(output_file):
            self.view.display_message_confirm(MSG_FILENOTEXIST_EN.format(output_file))
            return ''
        text = ''
        with open(output_file, 'r', encoding="utf-8", errors="ignore") as file:
            text = file.read()
        
        chunks = util.create_chunks(text, int(ratio_of_total_max_prompt * token_max_length))
        self.view.display_message_queue(MSG_REDACTIONSTARTED_EN.format(len(chunks)))
        redacted_text_list = []

        for i, chunk in enumerate(chunks, start=1):            
            start_time = datetime.datetime.now()
            self.view.display_message_queue(MSG_CHUNKSTARTED_EN.format(i, start_time.strftime("%H:%M:%S"), current_model_config))
            redact_with_OpenAI=OpenAIRedactor(apikey)            
            try:
                redacted_chunk = redact_with_OpenAI.call_openAi_redact(user_input=chunk, system_prompt = SYSTEM_PROMPT_DEF, model_config=current_model_config, max_completion_length = int((1 - ratio_of_total_max_prompt) * token_max_length) - 1, timeout=REQUEST_TIMEOUT)
            except Exception as e:
                self.view.display_message_queue(str(e))
            end_time = datetime.datetime.now()
            time_difference = end_time - start_time
            if redacted_chunk is None:
                redacted_chunk = ""
            redacted_text_list.append(redacted_chunk)
            self.view.display_message_queue(MSG_REDACTEDCHUNKRESULT_EN.format({redacted_chunk[:100]}))
            self.view.display_message_queue(MSG_CHUNKPROCESSINGTIME_EN.format(i, time_difference))

        redacted_text = " ".join(redacted_text_list)
        return redacted_text
    
class RedactorPresenter:
    def __init__(self, view: RedactorView):
        self.view = view
        self.model = RedactorModel(view)
        
    def handle_redaction(self): 
        output_file = self.view.form.output_file_display.value
        token_max_length = os.getenv("MAXTOKENLENGTH")
        if token_max_length is None or not token_max_length.isdigit:
            token_max_length = SetGptMaxTokenLength.get_default_token_length_number(self.view.form.current_model_config)
        else:
            token_max_length = int(token_max_length)
        redacted_text = self.model.redact(output_file, self.view.api_key, self.view.form.current_model_config, token_max_length, ratio_of_total_max_prompt) 
        if redacted_text == "":
            return
        self._write_redacted_file(output_file=output_file, redacted_text=redacted_text)
        self.view.display_message_queue(MSG_REDACTIONCOMPLETE_EN)
       
    def _write_redacted_file(self, output_file: str, redacted_text: str):
        output_dir, output_filename = os.path.split(output_file)
        redacted_filename = "redacted_" + output_filename
        redacted_output_file = os.path.join(output_dir, redacted_filename)

        with open(redacted_output_file, "w", encoding="utf-8", errors="ignore") as file:
            file.write(redacted_text)

class RedactPromptButton:
    def __init__(self, form):
        self.form = form
    def create(self):
        selected_value = ui_const.NAME_REDACTPROMPT_EN
        if os.getenv('REDACT_PROMPT') is not None:
            selected_value = ui_const.NAME_REDACTPROMPT_EN + "*"
        self.form.transcription_prompt_button = self.form.add(npyscreen.ButtonPress, name=selected_value, rely=7, relx=25)
        self.form.transcription_prompt_button.whenPressed = self.switch_to_redact_prompt_form
    def switch_to_redact_prompt_form(self):
        self.form.parentApp.switchForm('REDACT_PROMPT')        

class SetRedactPrompt(npyscreen.ActionPopup):
    
    def create(self):
        from npyscreen_overrides import EuMultiLineEdit
        multiline = self.add(EuMultiLineEdit, name = ui_const.NAME_REDACTPROMPT_EN, begin_entry_at=0, value=self.get_redact_prompt(), help = ui_const.HELP_SETINITIALPROMPT_EN, autowrap=True)
        multiline.reformat_preserve_nl()

    def get_redact_prompt(self) -> str:
        prompt = os.getenv('REDACT_PROMPT')
        if not prompt:
            prompt = SYSTEM_PROMPT_DEF
        return prompt.strip()
    
    def on_ok(self):
        set_key('.env', "REDACT_PROMPT", util.get_prompt_value_formatted(self.get_widget(0).value))
        self.parentApp.switchFormPrevious()
    
    def on_cancel(self):
        self.redact_prompt  = ""
        self.parentApp.switchFormPrevious()

class Gpt4CheckBox:
    def __init__(self, form):
        self.form = form    
    def create(self):
        checkbox_value_bool = self.get_cb_gpt_4_value_from_env()
        self.form.cb_gpt4 = self.form.add(npyscreen.Checkbox, name=ui_const.NAME_GPT4CBOX_EN, value=checkbox_value_bool, help= ui_const.HELP_GPT4CBOX_EN, rely=8, relx=30, max_width=20, max_height=1)
        self.form.cb_gpt4.whenToggled = self.update_cb_gpt4    
    def update_cb_gpt4(self):
        if self.form.cb_gpt4.value:
            set_key(".env", "GPT4", "True")
            self.form.current_model_config = "gpt-4"
        else:
            set_key(".env", "GPT4", "False")
            self.form.current_model_config = "gpt-3.5-turbo"
    
    def get_cb_gpt_4_value_from_env(self) -> bool:
        checkbox_value_str = os.getenv("GPT4")
        if checkbox_value_str == "True":
            return True
        else:
            return False
    
    @staticmethod
    def get_model_from_gpt_4_env() -> str:
        value_str = os.getenv("GPT4")
        if value_str == "True":
            return "gpt-4"
        else:
            return "gpt-3.5-turbo"
        
class GptMaxTokenLengthButton:
    def __init__(self, form):
        self.form = form
    def create(self):
        self.form.gpt_max_button = self.form.add(npyscreen.ButtonPress, name=ui_const.NAME_GPTMAXBUTTON_EN, rely=8, relx=50)
        self.form.gpt_max_button.whenPressed = self.switch_to_set_gpt_max_token_length
    def switch_to_set_gpt_max_token_length(self):
        self.form.parentApp.switchForm('MAXTOKENLENGTH')
        
class SetGptMaxTokenLength(npyscreen.ActionPopup):
    def create(self):        
        import ui_const
        self.add(npyscreen.Textfield, name = ui_const.NAME_GPTMAXLENGTHINPUT_EN, begin_entry_at=0, value=str(self.get_gpt_max_token_length_from_env()), max_width=6, max_height=1)        
    
    def on_ok(self):
        current_model = self.parentApp.getForm("MAIN").current_model_config
        input_value = self.get_widget(0).value
        if input_value.isdigit():
            input_value = SetGptMaxTokenLength.get_max_token_length_number(value=input_value, current_model=current_model)
            set_key('.env', "MAXTOKENLENGTH", str(input_value))
            self.parentApp.switchFormPrevious()
                
    def on_cancel(self):
        self.parentApp.switchFormPrevious()
    
    def get_gpt_max_token_length_from_env(self) -> int:
        current_model = self.parentApp.getForm("MAIN").current_model_config
        length = os.getenv("MAXTOKENLENGTH")
        if length.isdigit:
                SetGptMaxTokenLength.get_max_token_length_number(length, current_model=current_model)
                return length
        return SetGptMaxTokenLength.get_default_token_length_number(current_model)

    @staticmethod
    def get_max_token_length_number(value, current_model) -> int:
        value = int(value)
        if value > MAX_TOKEN_LENGTHS[current_model]:
            value = MAX_TOKEN_LENGTHS[current_model]
        elif value < 0:
            value = 500
        return value
    
    @staticmethod
    def get_default_token_length_number(current_model) -> int:
        if current_model=="gpt-4":
            return 3000
        return int(MAX_TOKEN_LENGTHS[current_model]*3/4)
