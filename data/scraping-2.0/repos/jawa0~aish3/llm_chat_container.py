# Copyright 2023 Jabavu W. Adams

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sdl2
import json
import logging
import matplotlib
import os
import openai
from typing import Optional
from gui import GUI, GUIContainer
from label import Label
from textarea import TextArea
from gui_layout import ColumnLayout
from session import ChatCompletionHandler


PANEL_WIDTH = 600
PANEL_HEIGHT = 120


class LLMChatContainer(GUIContainer):

    class ChatMessageUI(GUIContainer):
        @classmethod
        def from_json(cls, json, **kwargs):
            assert(json["class"] == cls.__name__)
            kwargs = super(LLMChatContainer.ChatMessageUI, cls)._enrich_kwargs(json, **kwargs)

            gui = kwargs.get('gui')
            assert(gui is not None)

            kwargs["default_setup"] = False
            instance = gui.create_control(json["class"], **kwargs)
            instance.set_bounds(*json["bounding_rect"])

            assert(len(json["children"]) == 2)
            assert(json["children"][0]["class"] == "Label")
            assert(json["children"][1]["class"] == "TextArea")

            for child_json in json["children"]:
                child_class_name = child_json["class"]
                child_class = GUI.control_class(child_class_name)
                child = child_class.from_json(child_json, **kwargs)
                instance.add_child(child)
                if child_class_name == "Label":
                    instance.label = child
                    instance.label._draggable = False
                elif child_class_name == "TextArea":
                    instance.text_area = child
                    instance.text_area._draggable = False
            
            instance._can_focus = False
            return instance


        def __init__(self, children=None, role="User", text="", **kwargs):
            super().__init__(children=children, layout=ColumnLayout(), **kwargs)
            
            default_setup = kwargs.get('default_setup', True)
            if default_setup:
                self.label = Label(text=role, w=PANEL_WIDTH, h=20, **kwargs)
                self.add_child(self.label)

                self.text_area = TextArea(18, w=PANEL_WIDTH, h=PANEL_HEIGHT, **kwargs)
                self.text_area.text_buffer.set_text(text)
                self.add_child(self.text_area)

                self.label._draggable = False
                self.text_area._draggable = False
                self._draggable = False
                self._can_focus = False


        def get_role(self):
            return self.label.get_text()
        
        def set_role(self, role):
            self.label.set_text(role)

        def get_text(self):
            return self.text_area.text_buffer.get_text()
        
        def set_text(self, text):
            self.text_area.text_buffer.set_text(text)


    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)
    

    def __init__(self, default_setup=True, **kwargs):
        super().__init__(**kwargs)
        self.draw_bounds = True
        
        self.set_layout(ColumnLayout())
        self.system = None
        self.utterances = []

        if default_setup:
            self.title = Label(text="LLM Chat [gpt-4]", 
                               w=PANEL_WIDTH, h=20, 
                               draggable=False,
                               editable=False,
                               **kwargs)
            
            self.add_child(self.title)

            text: str = os.getenv("DEFAULT_SYSTEM_PROMPT", "")
            self.system = self.ChatMessageUI(role="System", text=text, **kwargs)
            # self.gui.set_focus(self.system)

            self.utterances = [self.system, self.ChatMessageUI(role="User", **kwargs)]
            for utterance in self.utterances:
                self.add_child(utterance)

        self.accumulated_response_text = None

        self.busy_colormap = matplotlib.colormaps['summer']
        self._t_busy = 0.0


    def remove_child(self, child):
        super().remove_child(child)


    @classmethod
    def from_json(cls, json, **kwargs):
        assert(json["class"] == cls.__name__)
        kwargs = super(LLMChatContainer, cls)._enrich_kwargs(json, **kwargs)

        gui = kwargs.get('gui')
        assert(gui is not None)

        kwargs["default_setup"] = False
        instance = gui.create_control(json["class"], **kwargs)
        instance.set_bounds(*json["bounding_rect"])

        for child_json in json["children"]:
            child_class_name = child_json["class"]
            # child = gui.create_control(child_class_name, **kwargs)
            child_class = GUI.control_class(child_class_name)
            child = child_class.from_json(child_json, **kwargs)
            instance.add_child(child)
            if child_class_name == "ChatMessageUI":
                instance.utterances.append(child)

        # @note Override settings on legacy saved components...
        for child in instance.children:
            if isinstance(child, Label) and child.get_text().startswith("LLM Chat"):
                instance.title = child
                break

        instance.title._draggable = False
        instance.title._editable = False
        for utterance in instance.utterances:
            utterance._draggable = False
            utterance._editable = False

            utterance.text_area._draggable = False
            utterance.label._draggable = False
            utterance.label._editable = False


        return instance


    def handle_event(self, event):
        if event.type == sdl2.SDL_KEYDOWN:
            shiftPressed: bool = 0 != event.key.keysym.mod & (sdl2.KMOD_LSHIFT | sdl2.KMOD_RSHIFT)
            cmdPressed = (event.key.keysym.mod & (sdl2.KMOD_LGUI | sdl2.KMOD_RGUI))
            ctrlPressed: bool = 0 != event.key.keysym.mod & (sdl2.KMOD_LCTRL | sdl2.KMOD_RCTRL)
            keySymbol = event.key.keysym.sym

            if cmdPressed:
                # Cmd+G sends messages to GPT
                if keySymbol == sdl2.SDLK_g:
                    self.send()
                    return True
                
                # Cmd+U creates a new user message
                elif keySymbol == sdl2.SDLK_u: 
                    user = self.ChatMessageUI(role="User", gui=self.gui)
                    self.add_child(user)
                    self.utterances.append(user)
                    self.gui.set_focus(user.text_area, True)
                    return True  # event was handled
                
                # Cmd+Backspace/Delete deletes the currently focused message
                # If the whole LLMChatContainer is focused, then delete this chat.
                elif keySymbol == sdl2.SDLK_BACKSPACE:
                    focused_control = self.gui.get_focus()
                    if focused_control is self:
                        self.parent.remove_child(self)
                        return True

                    chat_message = focused_control.parent
                    if chat_message is not None and isinstance(chat_message, self.ChatMessageUI):
                        ancestors = self.gui.get_ancestor_chain(chat_message)
                        assert(self in ancestors)

                        # Can't delete first, System message
                        has_system = hasattr(self, "system") and self.system is not None
                        if not has_system or chat_message != self.system:
                            # @todo wrap in a remove message method
                            try:
                                self.utterances.remove(chat_message)
                            except ValueError:  # We don't always add, so won't always be in list (see derived classes if any)
                                pass

                            self.focus_ring.focus_previous()
                            self.remove_child(chat_message)
                            return True
                            
            if keySymbol == sdl2.SDLK_RETURN:
                # Delegate to GUIContainer
                handled = super().handle_event(event)
                if handled:
                    return True
            
            elif keySymbol == sdl2.SDLK_TAB:
                # TAB focuses next control in focus ring
                # Shift+TAB focuses previous control

                if not ctrlPressed:
                    # Ctrl+TAB inserts a tab character - we handle this in the TextArea class

                    focused = self.gui.get_focus()
                    if focused is not None and isinstance(focused, TextArea):
                        ancestors = self.gui.get_ancestor_chain(focused)
                        if self in ancestors:
                            message = focused.parent
                            assert(message is not None and isinstance(message, self.ChatMessageUI))
                            i = self.utterances.index(message)

                            if shiftPressed:  # if shift was also held
                                i = (i - 1) % len(self.utterances)
                            else:
                                i = (i + 1) % len(self.utterances)

                            self.gui.set_focus(self.utterances[i].text_area, True)
                            return True

        return self.parent.handle_event(event)


    # @note what is the point? This is saying that we can always accept focus, but we don't do anything on focus.
    def _change_focus(self, am_getting_focus: bool) -> bool:
        return True


    def send(self):
        messages = [{"role": u.get_role().lower(), "content": u.get_text()} for u in self.utterances]

        handler = ChatCompletionHandler(start_handler=self.on_llm_response_start,
                                        chunk_handler=self.on_llm_response_chunk,
                                        done_handler=self.on_llm_response_done)

        self.pulse_busy = True
        self._t_busy = 0.0

        model = 'gpt-4-1106-preview'
        # model = 'gpt-4'
        self.gui.session.llm_send_streaming_chat_request(model, messages, handlers=[handler])

        # Add Answer TextArea
        answer = self.gui.create_control("ChatMessageUI", role="Assistant", text='')
        self.add_child(answer)

        # Shrink previous messages
        for u in self.utterances:
            u.text_area.set_size(PANEL_WIDTH, 60)
        self.utterances.append(answer)
        self.updateLayout()


    def on_llm_response_start(self) -> None:
        self.accumulated_response_text = ""


    def on_llm_response_chunk(self, chunk_text: Optional[str]) -> None:
        assert(len(self.utterances) > 0)
        answer = self.utterances[-1]
        assert(isinstance(answer, self.ChatMessageUI) and answer.get_role() == "Assistant")

        if chunk_text is not None:
            answer.text_area.text_buffer.move_point_to_end()
            answer.text_area.text_buffer.insert(chunk_text)
            answer.text_area.set_needs_redraw()

            self.accumulated_response_text += chunk_text


    def on_llm_response_done(self) -> None:
        self.pulse_busy = False
        self._t_busy = 0.0

        # self.gui.say(self.accumulated_response_text)


    def get_json(self):
        return {
            "type": "LLMChatContainer",
            "version": 1,
            "system_text": self.system.text_area.text_buffer.get_text(),
        }

    
    def on_update(self, dt):
        self._t_busy += dt
    

    def _draw_bounds(self, vr):
        # Draw the bounding rectangle after all text has been drawn
        # Save the current color
        r, g, b, a = sdl2.Uint8(), sdl2.Uint8(), sdl2.Uint8(), sdl2.Uint8()
        sdl2.SDL_GetRenderDrawColor(self.renderer.sdlrenderer, r, g, b, a)
        old_color = (r.value, g.value, b.value, a.value)

        # Set the new color
        if self.pulse_busy:
            PERIOD_SECONDS = 1.0
            # triangle wave
            t = self._t_busy % PERIOD_SECONDS / PERIOD_SECONDS
            if t < PERIOD_SECONDS / 2:            
                fr, fg, fb, fa = self.busy_colormap(t)
            else:
                fr, fg, fb, fa = self.busy_colormap(1 - t)

            r = int(fr * 255)
            g = int(fg * 255)
            b = int(fb * 255)

        elif self.has_focus():
            r, g, b = (0, 127, 255)
        else:
            r, g, b = (100, 100, 100)
        sdl2.SDL_SetRenderDrawColor(self.renderer.sdlrenderer, r, g, b, 255)

        # Draw the bounding rectangle
        sdl2.SDL_RenderDrawRect(self.renderer.sdlrenderer, vr)

        # Reset to the old color
        sdl2.SDL_SetRenderDrawColor(self.renderer.sdlrenderer, old_color[0], old_color[1], old_color[2], old_color[3])


    def draw(self):
        super().draw()

        vr = self.get_view_rect()

        if self.pulse_busy:
            self._draw_bounds(vr)


GUI.register_control_type("LLMChatContainer", LLMChatContainer)
GUI.register_control_type("ChatMessageUI", LLMChatContainer.ChatMessageUI)