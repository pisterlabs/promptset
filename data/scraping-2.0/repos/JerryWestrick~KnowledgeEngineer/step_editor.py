from textual import on
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal
from textual.message import Message
from textual.widgets import Static, Label, Select, Button, Input

from OpenAI_API_Costs import OpenAI_API_Costs
from input_copypaste import InputCP
from logger import Logger
from processes import ProcessList, ProcessList_save
from step import Step


class StepEditor(Static):
    wlog: Logger = Logger(namespace="StepEditor", debug=True)
    pname: str = ''
    step: Step | None = None
    changes_are_internal: bool = False

    class StepAction(Message):
        """Step selected message."""

        def __init__(self, cname: str, pname: str, sname: str) -> None:
            self.cname: str = str(cname)
            self.pname: str = str(pname)
            self.sname: str = str(sname)
            super().__init__()

    def compose(self) -> ComposeResult:
        self.border_title = 'Step Editor'

        self.fields = [
            # Step Name
            Label('Name:', id="name_lbl", classes="field_lbl"),
            InputCP("1", name="name", id="name_field", classes="field_input"),

            # Prompt Name
            # Label('Prompt Name:', id="prompt_name_lbl", classes="field_lbl"),
            Label('Prompt Name>', id="step_prompt_name_btn", classes="field_lbl"),
            InputCP("2", name="prompt_name", id="prompt_name_field", classes="field_input"),

            # Storage Path
            # Label('Storage Path:', id="storage_path_lbl", classes="field_lbl"),
            Label('Storage Path>', id="step_storage_path_btn", classes="field_lbl"),
            InputCP("3", name="storage_path", id="storage_path_field", classes="field_input"),

            # Text File
            # Label('Text File:', id="text_file_lbl", classes="field_lbl"),
            Label('Text File>', id="step_text_file_btn", classes="field_lbl"),
            InputCP("4", name="text_file", id="text_file_field", classes="field_input"),

            # Model
            Label('Model:', id="model_lbl", classes="field_lbl"),
            # Input("5", id="model_field", classes="field_input"),
            Select([(k, k) for k in OpenAI_API_Costs.keys()],
                   id="model_select", name="model",
                   classes="field_input",
                   allow_blank=True,
                   value='gpt-3.5-turbo'
                   ),

            # Temperature
            Label('Temperature:', id="temperature_lbl", classes="field_lbl"),
            InputCP("6", name="temperature", id="temperature_field", classes="field_input"),

            # Max Tokens
            Label('Max Tokens:', id="max_tokens_lbl", classes="field_lbl"),
            InputCP("7", name="max_tokens", id="max_tokens_field", classes="field_input"),
        ]

        yield Grid(*self.fields, id="step_fields_grid")

        self.step_exec_btn = Button("Execute", id="step_exec_btn", classes="small_btn hidden")
        self.step_save_btn = Button("Save", id="step_save_btn", classes="small_btn hidden")

        yield Horizontal(
            self.step_exec_btn, self.step_save_btn,
            id="step_btn_bar", classes="btn_bar"
        )

    async def step_action(self, sa: StepAction) -> None:

        self.pname: str = sa.pname
        self.border_title = f"Step Editor: {self.pname}/{sa.sname}"
        for s in ProcessList[sa.pname]:
            if s.name == sa.sname:
                self.step = s
                break

        for aWidget in self.fields:
            match aWidget.__class__.__name__:
                case 'Label':
                    continue

                case 'Button':
                    continue

                case 'Input' | 'InputCP':
                    name = aWidget.id[:-6]
                    value = getattr(self.step, name, None)
                    if value is None:
                        value = getattr(self.step.ai, name, None)
                    with self.prevent(Input.Changed):
                        aWidget.value = str(value)
                    continue

                case "Select":
                    name = aWidget.id[:-7]
                    value = getattr(self.step, name, None)
                    if value is None:
                        value = getattr(self.step.ai, name, None)
                    with self.prevent(Select.Changed):
                        aWidget.value = str(value)
                    continue

                case _:
                    self.wlog.error(f"Widget:{aWidget.id} is of unknown class {aWidget.__class__}")

        self.step_exec_btn.remove_class("hidden").label = f"Execute: {self.step.name}"

        if sa.cname == 'Select':
            return

        self.wlog.info(f"Running Worker to execute step: {self.pname}/{self.step.name}")
        self.run_worker(self.step.run(self.pname), exclusive=True)

    @on(Input.Changed)
    @on(Select.Changed)
    async def step_modified(self, c):
        if self.step is None:
            return

        self.step_save_btn.remove_class("hidden")
        self.step_exec_btn.add_class("hidden")

        if c.control.name in ["name", "prompt_name", "storage_path", "text_file"]:
            setattr(self.step, c.control.name, c.value)
            self.wlog.info(f"change step.{c.control.name} = {c.value}")

        if c.control.name in ["model", "temperature", "max_tokens"]:
            setattr(self.step.ai, c.control.name, c.value)
            self.wlog.info(f"change step.ai.{c.control.name} = {c.value}")
        return

    @on(Button.Pressed, "#step_save_btn")
    def save_process_list(self):
        self.step_save_btn.add_class("hidden")
        self.step_exec_btn.remove_class("hidden")
        ProcessList_save(ProcessList)

    @on(Button.Pressed, "#step_exec_btn")
    def exec_step(self):
        self.post_message(self.StepAction("Execute", self.pname, self.step.name))


