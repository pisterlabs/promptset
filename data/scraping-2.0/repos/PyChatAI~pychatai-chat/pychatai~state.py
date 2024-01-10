"""Code containing the state of the app."""

import asyncio
import csv
import gettext
import os
import pickle
import uuid
from typing import Any, Coroutine, Optional

import reflex as rx
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains import LLMChain
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage

from reflex_gptp.async_callback import CustomAsyncIteratorCallbackHandler
from reflex_gptp.utils import MessagePartType, OutputType, plugin_tool, providers_models

load_dotenv()

USE_ENV_API_KEYS = os.getenv("USE_ENV_API_KEYS", "false").lower() == "true"

UUID = str


def make_uuid() -> UUID:
    """Generate a UUID.

    Returns:
        UUID: A random UUID.
    """
    return str(uuid.uuid4())


class MessagePart(rx.Base):
    """A message part."""

    id: UUID  # noqa: A003
    type: MessagePartType  # noqa: A003
    text: str
    extra_output: Optional[str] = None
    extra_output1: Optional[str] = None


class Message(rx.Base):
    """A message."""

    id: UUID  # noqa: A003
    parts: list[MessagePart]
    own: bool
    is_loading: bool = False


class Convo(rx.Base):
    """A conversation."""

    name: str
    messages: list[Message]


class Prompt(rx.Base):
    """A prompt."""

    title: str
    text: str


PROMPTS: list[Prompt] = []

with open("prompts.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        PROMPTS.append(Prompt(title=row[0], text=row[1]))


first_uuid = make_uuid()
default_provider = "anthropic"
default_model = "claude-2"


async def wrap_done(fn: Coroutine[Any, Any, dict[str, Any]], queue: asyncio.Queue, interrupt: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        fn_task: asyncio.Task = asyncio.create_task(fn)
        interrupt_task = asyncio.create_task(interrupt.wait())
        _, pending = await asyncio.wait([fn_task, interrupt_task], return_when="FIRST_COMPLETED")
        for p in pending:
            p.cancel()
        if interrupt.is_set():
            queue.put_nowait((OutputType.INTERRUPT, "Interrupted", None))
        else:
            res = fn_task.result()
            queue.put_nowait((OutputType.AGENT_FINISH, res["output"] if "output" in res else res["text"], None))
            return res
    except Exception as e:
        # TODO: handle exception
        print(f"Caught exception: {type(e)}: {e}")
        queue.put_nowait((OutputType.LLM_ERROR, "Error", None))


class State(rx.State):
    """The app state."""

    convos: dict[UUID, Convo] = {first_uuid: Convo(name="New conversation", messages=[])}

    openai_api_key: rx.LocalStorage = os.getenv("OPENAI_API_KEY", "") if USE_ENV_API_KEYS else ""  # type: ignore
    anthropic_api_key: rx.LocalStorage = os.getenv("ANTHROPIC_API_KEY", "") if USE_ENV_API_KEYS else ""  # type: ignore
    show_api_key_modal: bool = False

    current_convo: UUID = first_uuid

    edit_convo: bool = False
    drawer_open: bool = False
    processing: bool = False
    question: str = ""
    shift_down: bool = False
    _interrupt_event: Optional[asyncio.Event] = None

    input_should_focus: bool = True

    convo_model: dict[UUID, dict[str, str]] = {first_uuid: {"provider": default_provider, "name": default_model}}

    prompts: list[Prompt] = PROMPTS

    local_storage_convos: rx.LocalStorage = ""  # type: ignore
    local_storage_current_convo: rx.LocalStorage = ""  # type: ignore
    local_storage_model: rx.LocalStorage = ""  # type: ignore
    local_storage_enabled_plugins: rx.LocalStorage = ""  # type: ignore

    enabled_plugins: dict[UUID, dict[str, bool]] = {first_uuid: {k: False for k in plugin_tool}}

    form_provider: str = default_provider
    form_model: str = default_model
    show_model_modal: bool = False
    show_plugins_modal: bool = False
    show_prompts_modal: bool = False

    show_extra_output_modal: bool = False

    chat_modals_visible: dict[UUID, bool] = {}

    chat_popovers_visible: dict[UUID, bool] = {}

    @rx.var
    def have_api_key(self) -> bool:
        """A computed var that returns whether the user has set an API key."""
        if self.current_provider == "openai":
            return self.openai_api_key != ""
        if self.current_provider == "anthropic":
            return self.anthropic_api_key != ""
        return True

    def handle_api_key_submit(self, form_data: dict, provider: str):
        """Handle a form submission."""
        if provider == "openai":
            self.openai_api_key = form_data["api_key"]
        elif provider == "anthropic":
            self.anthropic_api_key = form_data["api_key"]

    def handle_model_submit(self, form_data: dict):
        """Handle a form submission."""
        self.convo_model[self.current_convo]["provider"] = form_data["provider"]
        self.convo_model[self.current_convo]["name"] = form_data["name"]
        self.toggle_model_modal()
        return

    @rx.var
    def empty_question(self) -> bool:
        """A computed var that returns whether the question is empty."""
        return self.question == ""

    @rx.var
    def convo_keys_names(self) -> list[tuple[UUID, str]]:
        """A computed var that returns a list of all conversation names in reverse order (new to old)."""
        return [(x[0], x[1].name) for x in self.convos.items()][::-1]

    @rx.var
    def current_convo_name(self) -> str:
        """A computed var that returns the name of the current conversation."""
        return self.convos[self.current_convo].name

    @rx.var
    def current_convo_messages(self) -> list[Message]:
        """A computed var that returns the messages of the current conversation."""
        return self.convos[self.current_convo].messages

    @rx.var
    def current_convo_plugins(self) -> dict[str, bool]:
        """A computed var that returns the plugins of the current conversation."""
        return self.enabled_plugins[self.current_convo]

    @rx.var
    def n_enabled_plugins(self) -> int:
        """A computed var that returns the number of enabled plugins."""
        return sum(self.current_convo_plugins.values())

    @rx.var
    def convo_has_messages(self) -> bool:
        """A computed var that returns whether the current conversation has messages."""
        return len(self.convos[self.current_convo].messages) > 0

    def set_convo(self, convo_key: UUID) -> None:
        """Set the current conversation."""
        self.current_convo = convo_key
        self.local_storage_current_convo = convo_key  # type: ignore

    def handle_convo_link_click(self, convo_key: UUID) -> None:
        """Handle a click on a conversation link."""
        yield State.set_convo(convo_key)  # type: ignore
        yield State.toggle_drawer()  # type: ignore

    def load_data(self) -> None:
        """Load conversations and other data from local storage."""
        # Load convos from local storage
        if self.local_storage_convos != "":
            self.convos = pickle.loads(self.local_storage_convos.encode("latin1"))
        if self.local_storage_current_convo != "":
            self.current_convo = self.local_storage_current_convo
        else:
            self.current_convo = next(iter(self.convos.keys()))
        # Load convo model from local storage
        if self.local_storage_model != "":
            self.convo_model = pickle.loads(self.local_storage_model.encode("latin1"))
        # Load enabled plugins from local storage
        if self.local_storage_enabled_plugins != "":
            self.enabled_plugins = pickle.loads(self.local_storage_enabled_plugins.encode("latin1"))

    def toggle_api_key_modal(self) -> None:
        """Toggle the API key modal."""
        self.show_api_key_modal = not self.show_api_key_modal

    def toggle_model_modal(self) -> None:
        """Toggle the model for selecting a provider and model."""
        self.show_model_modal = not self.show_model_modal
        if self.show_model_modal:
            self.form_provider = self.current_provider
            self.form_model = self.current_model

    def toggle_plugins_modal(self) -> None:
        """Toggle the plugins modal."""
        self.show_plugins_modal = not self.show_plugins_modal

    def toggle_plugin(self, plugin_name: str, value: bool) -> None:
        """Toggle a plugin."""
        self.enabled_plugins[self.current_convo][plugin_name] = value

    def toggle_prompts_modal(self) -> None:
        """Toggle the prompts modal."""
        self.show_prompts_modal = not self.show_prompts_modal

    async def set_prompt(self, prompt: dict[str, str]) -> None:
        """Set a prompt in the input."""
        self.question = prompt["text"]
        yield rx.set_value("input", prompt["text"])  # type: ignore
        yield State.toggle_prompts_modal()  # type: ignore

    def save_data(self):
        """Save conversations and other data to local storage."""
        # Save convos to local storage
        stringified_convos = pickle.dumps(self.convos.copy())
        self.local_storage_convos = stringified_convos.decode("latin1")  # type: ignore
        # Save current convo to local storage
        self.local_storage_current_convo = self.current_convo  # type: ignore
        # Save convo model to local storage
        stringified_model = pickle.dumps(self.convo_model.copy())
        self.local_storage_model = stringified_model.decode("latin1")  # type: ignore
        # Save enabled plugins to local storage
        stringified_enabled_plugins = pickle.dumps(self.enabled_plugins.copy())
        self.local_storage_enabled_plugins = stringified_enabled_plugins.decode("latin1")  # type: ignore

    @rx.background
    async def handle_submit(self, form_data: dict):
        """Handle a form submission."""
        question = form_data["input"]
        async with self:
            if not self.convo_has_messages:
                self.change_convo_name(question[:20])
            self.question = ""
            if question is None or question == "":
                return
            m_id = make_uuid()
            mp_id = make_uuid()
            self.convos[self.current_convo].messages.append(
                Message(id=m_id, parts=[MessagePart(id=mp_id, type=MessagePartType.TEXT, text=question)], own=True)
            )
            self.chat_popovers_visible[m_id] = False
            self.chat_modals_visible[mp_id] = False
            self.processing = True
            yield

            messages = []
            for message in self.convos[self.current_convo].messages:
                msg_cls = HumanMessage if message.own else AIMessage
                all_text = "\n".join([x.text for x in message.parts])
                messages.append(msg_cls(content=all_text))

            m_id = make_uuid()
            mp_id = make_uuid()
            message = Message(
                id=m_id, parts=[MessagePart(id=mp_id, type=MessagePartType.TEXT, text="")], own=False, is_loading=True
            )
            self.chat_popovers_visible[m_id] = False
            self.chat_modals_visible[mp_id] = False
            self.convos[self.current_convo].messages.append(message)
            yield

            callback = CustomAsyncIteratorCallbackHandler()
            callback_manager = AsyncCallbackManager([callback])

            if self.current_provider == "openai":
                llm = ChatOpenAI(
                    model=self.current_model,
                    max_tokens=None,
                    temperature=0.7,
                    api_key=self.openai_api_key,
                    n=1,
                    streaming=True,
                    callback_manager=callback_manager,
                )
            elif self.current_provider == "anthropic":
                llm = ChatAnthropic(
                    anthropic_api_key=self.anthropic_api_key,  # type: ignore
                    streaming=True,
                    model_name=self.current_model,
                    callback_manager=callback_manager,
                    temperature=0.7,
                    verbose=True,
                )
            else:
                raise ValueError(f"Unknown provider {self.current_provider}")

            history = ChatMessageHistory(messages=messages[:-1])
            memory = ConversationBufferMemory(chat_memory=history, memory_key="chat_history", return_messages=True)

            tools = [
                plugin_tool[k](callback_manager=callback_manager)
                for k, v in self.enabled_plugins[self.current_convo].items()
                if v
            ]

            self._interrupt_event = asyncio.Event()
            self._interrupt_event.clear()
        if tools:
            agent = (
                AgentType.OPENAI_MULTI_FUNCTIONS
                if self.current_provider == "openai"
                else AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION
            )
            agent_chain = initialize_agent(
                tools,
                llm,
                callback_manager=callback_manager,
                memory=memory,
                agent=agent,
                verbose=True,
            )

            run = asyncio.create_task(
                wrap_done(agent_chain.ainvoke({"input": question}), callback.queue, self._interrupt_event)
            )
        else:
            prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        "You are a nice chatbot having a conversation with a human. The output should be valid markdown."
                    ),
                    # The `variable_name` here is what must align with memory
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{question}"),
                ]
            )  # type: ignore
            conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
            run = asyncio.create_task(
                wrap_done(conversation.ainvoke({"question": question}), callback.queue, self._interrupt_event)
            )

        async for output_type, text, extra_output in callback.aiter():
            message.is_loading = False
            if output_type == OutputType.TOKEN:
                if messages and (last := message.parts[-1]).type == MessagePartType.TEXT:
                    message.parts[-1].text += text
                else:
                    mp_id = make_uuid()
                    message.parts.append(MessagePart(id=mp_id, type=MessagePartType.TEXT, text=text))
            elif output_type == OutputType.TOOL_START:
                if messages and (last := message.parts[-1]).type == MessagePartType.TEXT:
                    last.type = MessagePartType.TOOL_START
                    last.text = text
                    last.extra_output = extra_output
            elif output_type == OutputType.TOOL_END:
                if messages and (last := message.parts[-1]).type == MessagePartType.TOOL_START:
                    last.type = MessagePartType.TOOL_END
                    last.text = text
                    if extra_output is not None:
                        last.extra_output1 = extra_output
            elif output_type == OutputType.AGENT_FINISH:
                if messages and (last := message.parts[-1]).type == MessagePartType.TEXT:
                    last.type = MessagePartType.AGENT_FINISH
                    last.text = text
                    last.extra_output = extra_output
            elif output_type == OutputType.LLM_ERROR:
                mp_id = make_uuid()
                message.parts.append(MessagePart(id=mp_id, type=MessagePartType.ERROR, text=text))
            elif output_type == OutputType.INTERRUPT:
                mp_id = make_uuid()
                message.parts.append(MessagePart(id=mp_id, type=MessagePartType.INTERRUPT, text=text))
            else:
                print(output_type, text)
            yield

        await run
        async with self:
            self.processing = False
            message.is_loading = False
            self.save_data()

    def handle_question_submit(self, form_data: dict):
        """Handle question being submitted through the input."""
        yield State.handle_submit(form_data)  # type: ignore
        yield rx.set_value("input", "")
        yield State.add_focus()  # type: ignore

    async def interrupt_chat(self):
        """Interrupt the chat."""
        if self._interrupt_event is not None:
            self._interrupt_event.set()
        yield

    def toggle_drawer(self) -> None:
        """Toggle the drawer."""
        self.drawer_open = not self.drawer_open

    def toggle_edit_convo(self) -> None:
        """Toggle the edit conversation mode."""
        self.edit_convo = not self.edit_convo

    def change_convo_name(self, new_name: str) -> None:
        """Change the name of the current conversation."""
        if new_name != "":
            self.convos[self.current_convo].name = new_name
            self.save_data()

    def new_convo(self, copy_current: bool = True) -> None:
        """Create a new conversation."""
        new_uuid = make_uuid()
        self.convos[new_uuid] = Convo(name="New conversation", messages=[])
        self.convo_model[new_uuid] = (
            self.convo_model[self.current_convo].copy()
            if copy_current
            else {"provider": default_provider, "name": default_model}
        )
        self.enabled_plugins[new_uuid] = (
            self.enabled_plugins[self.current_convo].copy() if copy_current else {k: False for k in plugin_tool}
        )
        self.current_convo = new_uuid
        self.save_data()

    def handle_new_convo_click(self) -> None:
        """Handle a click on the new conversation button."""
        if self.convo_has_messages:
            yield State.new_convo()  # type: ignore
            yield State.add_focus()  # type: ignore

    def delete_convo(self, convo_key: UUID) -> None:
        """Deletes a conversation identified by its key.

        Args:
            convo_key (UUID): The unique identifier of the conversation to be deleted.
        """
        del self.convos[convo_key]
        del self.convo_model[convo_key]
        # TODO: delete all related entries in self.chat_modals_visible
        if convo_key == self.current_convo:
            self.current_convo = next(iter(self.convos.keys()))
        self.save_data()

    def delete_convos(self) -> None:
        """Delete all conversations."""
        self.convos.clear()
        self.convo_model.clear()
        self.enabled_plugins.clear()
        self.chat_modals_visible.clear()
        self.chat_popovers_visible.clear()
        self.new_convo(copy_current=False)
        self.save_data()

    def handle_provider_change(self, provider: str) -> None:
        """Handle a change in the provider in the model form."""
        self.form_provider = provider
        self.form_model = providers_models[provider][0]

    def handle_model_change(self, model: str) -> None:
        """Handle a change in the model in the model form."""
        self.form_model = model

    @rx.var
    def current_provider(self) -> str:
        """A computed var that returns the provider of the current conversation."""
        return self.convo_model[self.current_convo]["provider"]

    @rx.var
    def current_model(self) -> str:
        """A computed var that returns the model of the current conversation."""
        return self.convo_model[self.current_convo]["name"]

    @rx.var
    def form_provider_models(self) -> list[str]:
        """A computed var that returns the available models of the current provider."""
        return providers_models[self.form_provider]

    def remove_focus(self, _):
        """Remove focus from the input."""
        self.input_should_focus = False

    def add_focus(self):
        """Add focus to the input."""
        self.input_should_focus = True

    def toggle_chat_modal(self, message_part_id: UUID):
        """Toggle the modal with extra output for a message."""
        self.chat_modals_visible[message_part_id] = not self.chat_modals_visible.get(message_part_id, False)

    def set_chat_popover_visible(self, message_id: UUID, visible: Optional[bool] = None):
        """Toggle the popover with actions for a message."""
        print("original visible arg:", visible)
        for mi in self.chat_popovers_visible:
            if mi != message_id:
                self.chat_popovers_visible[mi] = False
        if visible is None:  # Toggle
            visible = not self.chat_popovers_visible.get(message_id, False)
        print(message_id, self.chat_popovers_visible.get(message_id, None), visible)
        self.chat_popovers_visible[message_id] = visible

    def copy_message(self, message: dict[str, Any]):
        """Copy the text of the given message to the clipboard."""
        yield rx.set_clipboard(message["parts"][-1]["text"])
        yield State.set_chat_popover_visible(message["id"], False)  # type: ignore

    async def regenerate_response(self, message: dict[str, Any]):
        """Regenerate the response for the given message."""
        parsed_message = Message.parse_obj(message)
        yield State.set_chat_popover_visible(parsed_message.id, False)  # type: ignore
        idx = next(i for i, m in enumerate(self.convos[self.current_convo].messages) if m.id == parsed_message.id)
        new_messages = self.convos[self.current_convo].messages.copy()[:idx]
        self.convos[self.current_convo] = Convo(name=self.convos[self.current_convo].name, messages=new_messages)
        yield State.handle_submit({"input": parsed_message.parts[-1].text})  # type: ignore
