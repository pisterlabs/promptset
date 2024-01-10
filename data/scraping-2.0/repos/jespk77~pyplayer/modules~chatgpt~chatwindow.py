import json

from ui.qt import pywindow, pyelement, pydialog
from core import modules
module = modules.Module(__package__)

import openai, tiktoken
openai.enable_telemetry = False
from . import parameters, requestworker

class ChatGPTWindow(pywindow.PyWindow):
    main_window_id = "chatgpt_window"
    request_update_task_id = "chatgpt_request_update"
    request_complete_task_id = "chatgpt_request_complete"
    request_error_task_id = "chatgpt_request_error"
    start_text = "--- Ready for your question ---"

    def __init__(self, parent):
        pywindow.PyWindow.__init__(self, parent, window_id=self.main_window_id)
        self.layout.row(0, weight=1).column(0, weight=1)
        self.title = "AI Assistant: ChatGPT"
        self.icon = "assets/icon_album"
        self._dialog = None

        self._set_parameters()
        self._token_encoder = tiktoken.encoding_for_model(self.model)
        self._messages = []
        self._input_changed = False
        self._request_worker = None

        self.add_task(self.request_update_task_id, self._add_response_delta)
        self.add_task(self.request_complete_task_id, self._complete_response)
        self.add_task(self.request_error_task_id, self._error_response)

    def _set_parameters(self):
        input_frame, output_frame = self["input"], self["output"]
        api_key, organization_id = module.configuration.get("api_key"), module.configuration.get("organization_id")
        error = True

        if not organization_id: output_frame["content"].text = "No organization ID set, check the settings"
        elif not api_key: output_frame["content"].text = "No API key set, check the settings"
        else:
            openai.organization, openai.api_key = organization_id, api_key
            output_frame["content"].text = self.start_text
            error = False

        self._set_input_enabled(not error)

    def create_widgets(self):
        output_frame = self.add_element("output", element_class=pyelement.PyLabelFrame)
        output_frame.layout.row(0, weight=1).column(1, weight=1).margins(3)
        output_frame.label = "Output"

        output = output_frame.add_element("content", element_class=pyelement.PyTextField, columnspan=5)
        output.accept_input = False
        output.font_size = 12
        cost_label = output_frame.add_element("status", element_class=pyelement.PyTextLabel, row=1)
        cost_label.set_alignment("centerV")
        cost_label.text = "Ask your question below"

        load = output_frame.add_element("load", element_class=pyelement.PyButton, row=1, column=2)
        load.text = "Load"
        @load.events.EventInteract
        def _load():
            if self._dialog is not None: self._dialog.close()
            self._dialog = pydialog.PyFileDialog(self, "existing", text="Load session", directory=module.configuration.get("$default_save_folder", ""))
            self._dialog.load = True
            self._dialog.events.EventSubmit(self._load_output)
            self._dialog.events.EventCancel(lambda : self._load_output(None))
            self._dialog.filter = "*.gpt"
            self._dialog.open()

        save = output_frame.add_element("save", element_class=pyelement.PyButton, row=1, column=3)
        save.text = "Save"
        @save.events.EventInteract
        def _save():
            if self._dialog is not None: self._dialog.close()
            self._dialog = pydialog.PyFileDialog(self, "any", text="Save session", directory=module.configuration.get("$default_save_folder", ""))
            self._dialog.save = True
            self._dialog.events.EventSubmit(self._save_layout)
            self._dialog.events.EventCancel(lambda : self._save_layout(None))
            self._dialog.filter = "*.gpt"
            self._dialog.open()

        reset = output_frame.add_element("reset", element_class=pyelement.PyButton, row=1, column=4)
        reset.text = "Reset"
        @reset.events.EventInteract
        def _reset_all():
            self._messages.clear()
            self["output"]["content"].text = self.start_text

        self.add_element("separator", element_class=pyelement.PySeparator, row=1).drag_resize = True
        input_frame = self.add_element("input", element_class=pyelement.PyLabelFrame, row=2)
        input_frame.layout.row(6, weight=1).column(1, weight=1).margins(3)
        input_frame.label = "Input"

        input_frame.add_element("mdlbl", element_class=pyelement.PyTextLabel, column=0).text = "Model:"
        model = input_frame.add_element("model", element_class=pyelement.PyTextInput, column=1, columnspan=2)
        model.text = self.model
        @model.events.EventKeyDown("all")
        def _set_model(element): self.model = element.text

        input_frame.add_element("tlbl", element_class=pyelement.PyTextLabel, row=1).text = "Temperature:"
        temperature = input_frame.add_element(element=pyelement.PyNumberInput(input_frame, "temperature", double=True), row=1, column=1)
        temperature.tooltip = parameters.temperature.description
        temperature.value, temperature.min, temperature.max, temperature.step = self.temperature, 0, 2, 0.1
        @temperature.events.EventInteract
        def _update_temperature(): self.temperature = temperature.value
        temperature_default = input_frame.add_element("temperature_default", element_class=pyelement.PyButton, row=1, column=2)
        temperature_default.text = "Reset"
        @temperature.events.EventInteract
        def _reset_temperature(): self.temperature = temperature.value = parameters.temperature.default_value

        input_frame.add_element("tplbl", element_class=pyelement.PyTextLabel, row=2).text = "Top_P:"
        top_p = input_frame.add_element(element=pyelement.PyNumberInput(input_frame, "top_p", double=True), row=2, column=1)
        top_p.tooltip = parameters.top_p.description
        top_p.value, top_p.min, top_p.max, top_p.step = self.top_p, 0, 1, 0.1
        @top_p.events.EventInteract
        def _update_top_p(): self.top_p = top_p.value
        top_p_default = input_frame.add_element("top_p_default", element_class=pyelement.PyButton, row=2, column=2)
        top_p_default.text = "Reset"
        @top_p_default.events.EventInteract
        def _reset_top_p(): self.top_p = top_p.value = parameters.top_p.default_value

        input_frame.add_element("tklbl", element_class=pyelement.PyTextLabel, row=3).text = "Token limit:"
        token_limit = input_frame.add_element("max_tokens", element_class=pyelement.PyNumberInput, row=3, column=1)
        token_limit.tooltip = parameters.max_tokens.description
        token_limit.value, token_limit.min, token_limit.max, token_limit.step = self.max_tokens, 0, 4096, 10
        @token_limit.events.EventInteract
        def _update_max_tokens(): self.max_tokens = token_limit.value
        token_limit_default = input_frame.add_element("max_tokens_default", element_class=pyelement.PyButton, row=3, column=2)
        token_limit_default.text = "Reset"
        @token_limit_default.events.EventInteract
        def _reset_max_token(): self.max_tokens = token_limit.value = parameters.max_tokens.default_value

        input_frame.add_element("pplbl", element_class=pyelement.PyTextLabel, row=4).text = "Presence penalty:"
        presence_penalty = input_frame.add_element(element=pyelement.PyNumberInput(input_frame, "presence_penalty", double=True), row=4, column=1)
        presence_penalty.tooltip = parameters.presence_penalty.description
        presence_penalty.value, presence_penalty.min, presence_penalty.max, presence_penalty.step = self.presence_penalty, -2, 2, 0.1
        @presence_penalty.events.EventInteract
        def _update_presence_penalty(): self.presence_penalty = presence_penalty.value
        presence_penalty_default = input_frame.add_element("presence_penalty_default", element_class=pyelement.PyButton, row=4, column=2)
        presence_penalty_default.text = "Reset"
        @presence_penalty_default.events.EventInteract
        def _reset_presence_penalty(): self.presence_penalty = presence_penalty.value = parameters.presence_penalty.default_value

        input_frame.add_element("fplbl", element_class=pyelement.PyTextLabel, row=5).text = "Frequency penalty:"
        freq_penalty = input_frame.add_element(element=pyelement.PyNumberInput(input_frame, "frequence_penalty", double=True), row=5, column=1)
        freq_penalty.tooltip = parameters.frequency_penalty.description
        freq_penalty.value, freq_penalty.min, freq_penalty.max, freq_penalty.step = self.frequency_penalty, -2, 2, 0.1
        @freq_penalty.events.EventInteract
        def _update_freq_penalty(): self.frequency_penalty = freq_penalty.value
        freq_penalty_default = input_frame.add_element("frequency_penalty_default", element_class=pyelement.PyButton, row=5, column=2)
        freq_penalty_default.text = "Reset"
        @freq_penalty_default.events.EventInteract
        def _reset_freq_penalty(): self.frequency_penalty = freq_penalty.value = parameters.frequency_penalty.default_value

        input = input_frame.add_element("content", element_class=pyelement.PyTextField, row=6, columnspan=3)
        input.font_size = 12
        @input.events.EventKeyDown("Return")
        def _send(modifiers):
            if "shift" not in modifiers:
                self._send_message()
                return input.events.block_action

        @input.events.EventKeyDown("all")
        def _check_tokens(element): self._check_token_count(element)

        send = input_frame.add_element("send", element_class=pyelement.PyButton, row=7, column=2)
        send.text = "Send"
        send.events.EventInteract(self._send_message)
        token_text = input_frame.add_element("token", element_class=pyelement.PyTextLabel, row=7)
        token_text.text = "Input cost: 0 tokens"
        token_text.set_alignment("center")

    def _set_input_enabled(self, enabled):
        self["output"]["status"].text = "Ready" if enabled else "Generating response..."
        self["output"]["reset"].accept_input = enabled
        for element in self["input"].children:
            try: element.accept_input = enabled
            except: pass

    user_to_role = {
        "user": "You",
        "assistant": "ChatGPT"
    }
    def _set_content_from_message(self):
        self["output"]["content"].text = "\n".join([f"{self.user_to_role.get(message['role'], message['role'])}: {message['content']}\n" for message in self._messages])

    def _load_output(self, value):
        self._dialog = None
        if value is not None:
            if not value.endswith(".gpt"): value += ".gpt"
            try:
                with open(value, "r") as file:
                    self._messages = json.load(file)
                self._set_content_from_message()
                self["output"]["status"].text = f"Loaded from {value}"
            except Exception as e:
                print("ERROR", "Failed to load file", e)
                self["output"]["status"].text = f"Failed to load {value}, see log for details"

    def _save_layout(self, value):
        self._dialog = None
        if value is not None:
            if not value.endswith(".gpt"): value += ".gpt"
            try:
                with open(value, "w") as file:
                    json.dump(self._messages, file, indent=5)
                self["output"]["status"].text = f"Saved to {value}"
            except Exception as e:
                print("ERROR", "Failed to save file", e)
                self["output"]["status"].text = f"Failed to save {value}, see log for details"

    @property
    def model(self): return self.configuration.get_or_create("model", parameters.model.default_value)
    @model.setter
    def model(self, model): self.configuration["model"] = model

    @property
    def temperature(self): return self.configuration.get_or_create("temperature", parameters.temperature.default_value)
    @temperature.setter
    def temperature(self, temperature): self.configuration["temperature"] = float(temperature)

    @property
    def top_p(self): return self.configuration.get_or_create("top_p", parameters.top_p.default_value)
    @top_p.setter
    def top_p(self, top_p): self.configuration["top_p"] = float(top_p)

    @property
    def max_tokens(self): return self.configuration.get_or_create("max_tokens", parameters.max_tokens.default_value)
    @max_tokens.setter
    def max_tokens(self, tokens): self.configuration["max_tokens"] = int(tokens)

    @property
    def presence_penalty(self): return self.configuration.get_or_create("presence_penalty", parameters.presence_penalty.default_value)
    @presence_penalty.setter
    def presence_penalty(self, penalty): self.configuration["presence_penalty"] = float(penalty)

    @property
    def frequency_penalty(self): return self.configuration.get_or_create("frequency_penalty", parameters.frequency_penalty.default_value)
    @frequency_penalty.setter
    def frequency_penalty(self, penalty): self.configuration["frequency_penalty"] = float(penalty)

    def _check_token_count(self, element):
        if element is not None:
            self["input"]["token"].text = f"Input cost: {len(self._token_encoder.encode(element.text))} tokens"
            self._input_changed = True
        else: self["input"]["token"].text = "Input cost: 0 tokens"

    def _add_response_delta(self, delta):
        self["output"]["content"].append(f'{delta}'.replace("``", "=============================\n"), move_cursor=True)

    def _complete_response(self, response, reason):
        self._messages.append(response)
        self._request_worker = None
        self["output"]["content"].append(f"\n[Completed with reason '{reason}']\n", move_cursor=True)
        self._set_input_enabled(True)

    def _error_response(self):
        self["output"]["content"].append("\n[ERROR] Request failed, try again later. See log for more information...", move_cursor=True)
        self._set_input_enabled(True)

    def _send_message(self):
        if self._request_worker is not None: return

        message = self["input"]["content"].text
        if message:
            if self._input_changed:
                self._messages.append({"role": "user", "content": message})
                self["output"]["content"].append(f"\nYou: {message}\n\nChatGPT: ", move_cursor=True)
                self._check_token_count(None)
                self._input_changed = False

            if self._generate_response(): self["input"]["content"].text = ""

    def _generate_response(self):
        try:
            self._set_input_enabled(False)
            args = {
                "model": self.model,
                "messages": self._messages,
                "max_tokens": self.max_tokens if self.max_tokens > 0 else None,
                "temperature": self.temperature, "top_p": self.top_p,
                "stream": True,
                "presence_penalty": self.presence_penalty, "frequency_penalty": self.frequency_penalty
            }

            self._request_worker = requestworker.ChatGPTRequestWorker(self, args)
            self._request_worker.start()
            return True
        except Exception as e:
            print("ERROR", "Failed to process message", e)
            return False