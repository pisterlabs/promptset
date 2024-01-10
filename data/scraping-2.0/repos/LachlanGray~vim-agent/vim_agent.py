import openai
import re

# Completion functions ########################################

def chat_3(messages: list[dict], stop=None):
    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages,
        temperature = 0.9,
        stop=stop,
        stream=True
    )
    for chunk in completion:
        if "content" in chunk.choices[0].delta:
            yield chunk.choices[0].delta["content"]

# content = "Can you write me a vim script to delete from lines 3 to 8?"
# messages = [{"role":"user", "content": f"{content}"} ]
# for chunk in chat_3(messages):
#     print(chunk, end='')


def chat_4(messages: list[dict], stop=None):
    completion = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = messages,
        temperature = 0.9,
        stop=stop,
        stream=True
    )
    for chunk in completion:
        if "content" in chunk.choices[0].delta:
            yield chunk.choices[0].delta["content"]

engines = {
    "gpt-3.5": chat_3,
    "gpt-4": chat_4
}


def stream_code(request: str, context=None, model="gpt-3.5"):
    content = f"{request}"
    if context:
        content += f"{context}"
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": content }
    ]

    completion = ""
    code = ""

    open_code_block = re.compile(r'```.*?(\n.*)', re.DOTALL)
    matched = False

    engine = engines[model]

    for chunk in engine(messages, stop="``` "):
        completion += chunk
        if not matched:
            matching = open_code_block.search(completion)
            if matching:
                code_chunk = matching.group(1)
                matched = True
                completion = ""
                code += code_chunk
                matching = ""
                yield code_chunk
        else:
            if "`" in chunk:
                matched = False
                yield chunk.split("`")[0]
                break
            yield chunk


def stream_chat(request: str, context=None, stop=None, model="gpt-3.5"):
    content = f"{request}"
    if context:
        content += f"{context}"
    messages = [
        # {"role": "system", "content": ""},
        {"role": "user", "content": content }
    ]

    engine = engines[model]

    for chunk in engine(messages, stop=stop):
        yield chunk

# request = "Can you write me a vim script to delete from lines 3 to 8?"
# context = None
# for chunk in stream_code(request, context=context):
#     print(chunk, end="")

# request = "What kind of files should I look for to find the main part of a javascript web app?"
# context = ""
# print("\033c")
# for chunk in stream_chat(request, context=context):
#     print(chunk, end="")


# Editor ########################################
from pynvim import attach
import os


class VimInstance:
    def __init__(self):
        try:
            self.vim = attach('tcp', address='127.0.0.1', port=7777)
        except:
            raise Exception("Could not connect to Nvim instance. Is it running?")

        self.mode_names= {
            "n": "NORMAL",
            "i": "INSERT",
            "v": "VISUAL",
            "V": "V-LINE",
            "^V": "V-BLOCK",
            "c": "COMMAND",
            "s": "SELECT",
            "S": "S-LINE",
            "^S": "S-BLOCK",
            "r": "HIT-ENTER",
            "R": "REPLACE",
            "Rv": "V-REPLACE",
            "cv": "EX",
            "ce": "EX",
            "r?": "CONFIRM",
            "!": "SHELL",
            "t": "TERMINAL"
        }

    @property
    def current_working_dir(self):
        return self.vim.eval('getcwd()')

    @property
    def current_buffer_content(self):
        return self.vim.request('nvim_buf_get_lines', self.vim.current.buffer, 0, -1, True)

    @property
    def cursor_position(self):
        # NOTE row is not zero indexed and column is; starts at [1,0]
        return self.vim.request('nvim_win_get_cursor', self.vim.current.window)

    @property
    def current_mode(self):
        mode = self.vim.eval('mode()')
        return self.mode_names[mode]

    @property
    def current_buffer(self):
        current_buffer = self.vim.request('nvim_get_current_buf')
        current_buffer_name = self.vim.request('nvim_buf_get_name', current_buffer)
        cwd = self.current_working_dir
        current_buffer_name = os.path.relpath(current_buffer_name, cwd)
        return current_buffer_name

    @property 
    def current_buffer_context(self):
        current_buffer = self.vim.request('nvim_get_current_buf')
        current_buffer_path = self.vim.request('nvim_buf_get_name', current_buffer)
        buffer_dir = os.path.dirname(current_buffer_path)
        buffer_context = os.listdir(buffer_dir)

        return buffer_context

    @property
    def open_buffers(self):
        buffers = self.vim.request('nvim_list_bufs')
        buffer_names = [self.vim.request('nvim_buf_get_name', buf) for buf in buffers]
        cwd = self.current_working_dir
        buffer_names = [os.path.relpath(buf, cwd) for buf in buffer_names]
        return buffer_names

    @property
    def current_view(self):
        # Get the range of visible lines in the current window
        first_visible_line_number = self.vim.request('nvim_eval', "line('w0')")
        last_visible_line_number = self.vim.request('nvim_eval', "line('w$')")

        # Get the contents of the visible lines in the current buffer
        visible_lines = self.vim.request('nvim_buf_get_lines', self.vim.current.buffer, first_visible_line_number - 1, last_visible_line_number, False)

        # add line numbers
        visible_lines = [f"{i+first_visible_line_number: >4} {line}" for i, line in enumerate(visible_lines)]

        return visible_lines

    def input(self, expr):
        self.vim.input(expr)

    def get_buffer_content(self, buffer_name, start=0,end=-1):
        # get the content from a buffer by name
        buffer_id = None
        for buf in self.vim.request('nvim_list_bufs'):
            if buffer_name in self.vim.request('nvim_buf_get_name', buf):
                buffer_id = buf
                break

        # Step 2: Get the contents of the buffer using the buffer ID
        buffer_contents = self.vim.request('nvim_buf_get_lines', buffer_id, start, end, True)
        return buffer_contents

    def send_command(self, command):
        self.vim.request('nvim_command', command)

    def send_message(self, message):
        # if there is a ' in the message it ends the echo, need to escape it
        return self.vim.request('nvim_command', f"echo '{message}'")


# vim = VimInstance()
# print("\n".join(vim.current_view))


# Agents ########################################

print_action = True

class ActionAgent:
    """
    Mediates actions with the editor
    """

    def __init__(self, vim):
        self.vim = vim

    def execute(self, request):

        request = f"Given these lines, please write a vimscript that will {request}. I have provided the line numbers to simplify the script for you:"
        context="\n".join(self.vim.current_view) + "\n"
        context += "\nBUT whatever you do, don't define functions and don't close vim."

        action = ""
        for chunk in stream_code(request, context, model="gpt-4"):
            if print_action:
                print(chunk, end="")
            action += chunk

            while "\n" in action:
                if action[0] == "\n":
                    action = action[1:]
                    continue

                expr, trailing = action.split("\n",1)

                if expr[0].lstrip() == '"':
                    action = trailing
                    continue

                if expr[0].lstrip() == ":":
                    expr += "<CR>"

                self.vim.input(expr)
                action = trailing

# vim = VimInstance()
# action_agent = ActionAgent(vim)
# action_agent.execute("delete the max_level option")

def main():
    vim = VimInstance()
    action_agent = ActionAgent(vim)

    while True:
        request = input("\033[35m>\033[0m ")
        action_agent.execute(request)

if __name__ == "__main__":
    main()



# We will need fancier agents to do more complex actions... progress so far is below

# print_summaries = False

# class ContextAgent:
#     """
#     Maintains a coherent context of what is happening in the editor.
#     """
#     def __init__(self, vim):
#         self.vim = vim
#         self.buffer_descriptions = {}
#         # TODO: save and load these from cache

#     @property
#     def open_buffers(self):
#         return self.vim.open_buffers

#     @property
#     def editor_context(self):
#         for buffer in self.open_buffers:
#             if buffer not in self.buffer_descriptions:
#                 self.describe_buffer(buffer)

#         current_buffer = self.vim.current_buffer
#         context = f"Currently, I have `{current_buffer}` open."

#         line, col = self.vim.cursor_position
#         context += f" Here is what I see, starting with line {line}:\n"
#         buffer_view = self.vim.current_view
#         buffer_view = "\n".join(buffer_view)
#         context += f"```\n{buffer_view}\n```"

#         if len(self.open_buffers) > 1:
#             other_buffers = "\n".join([f"- {buffer}: {self.buffer_descriptions[buffer]}" for buffer in self.open_buffers])
#             context += "\n\nThese are all the files I have open:\n" + other_buffers

#         return context


#     def describe_buffer(self, buffer_name):
#         buffer_content = "\n".join(self.vim.get_buffer_content(buffer_name))
#         request = f"In one sentence, what do I need to know about `{buffer_name}` to use it?:"
#         context = f"\n```\n{buffer_content}\n```"

#         # store long description
#         description = []
#         for chunk in stream_chat(request, context=context, stop="\n"):
#             if print_summaries:
#                 print(chunk, end="")
#             description.append(chunk)

#         description = "".join(description)
#         self.buffer_descriptions[buffer_name] = description


# # vim = VimInstance()
# # context_agent = ContextAgent(vim)
# # print(context_agent.editor_context)

