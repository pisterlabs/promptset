import os
import openai
import pynvim

def line_diff(subseq, line):
    ans = ""
    at = 0
    for c in line:
        if len(subseq) > at and c == subseq[at]:
            at += 1
        else:
            ans += c
    return ans

def get_system_prompt(prompt="default"):
    prompt_dir = os.path.expanduser("~/.config/chatvim/prompts")
    prompt_file = os.path.join(prompt_dir, prompt)
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            return f.read()
    return None
@pynvim.plugin
class GPTPlugin:
    def __init__(self, nvim):
        self.nvim = nvim

    @pynvim.function("GPTResponse")
    def gpt_response(self, args):
        history, last_talked = self._get_chat_history()
        prompt = get_system_prompt()
        if prompt:
            history = [{"role": "system", "content": prompt}] + history

        model = self.nvim.vars.get("gpt_model", "gpt-3.5-turbo")
        # last_talked overrides global variable
        if last_talked == "3":
            model = "gpt-3.5-turbo"
        elif last_talked == "4":
            model = "gpt-4"

        if len(history) > 0:
            self.make_gpt_request(history, model)

    def make_gpt_request(self, history, model):
        response = openai.ChatCompletion.create(
            model=model,
            messages=history,
            stream=True
        )

        initial_paste_value = self.nvim.command_output('set paste?')
        self.nvim.command('set paste')
        self.nvim.feedkeys("oGPT: ")

        total_response = ""
        interrupted = False
        try:
            for chunk in response:
                current_line = self.nvim.call('getline', '.')
                prefix = ""
                if current_line.startswith("GPT: "):
                    prefix = "GPT: "
                    current_line = current_line[5:]
                if len(current_line) != 0 and current_line != total_response[-len(current_line):]:
                    last_line_response = total_response.split("\n")[-1]
                    diff = line_diff(last_line_response, current_line)
                    self.nvim.feedkeys("\x03cc{}{}\n> {}\x03".format(prefix, last_line_response, diff))

                    interrupted = True
                    break
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    chunk = delta['content']
                    self.nvim.feedkeys(chunk)
                    total_response += chunk
        except KeyboardInterrupt:
            self.nvim.command('echom "Keyboard Interrupt received"')
        finally:
            response.close()
            if not interrupted:
                self.nvim.feedkeys("\x03o> \x03")
            self.nvim.command('set {}'.format(initial_paste_value))
            if interrupted:
                self.nvim.feedkeys("a")

    def _get_chat_history(self):
        cursor_line, _ = self.nvim.current.window.cursor
        lines = self.nvim.current.buffer[:cursor_line]
        history = []
        last_talked = None

        for line in lines:
            if line.startswith("//") or line.startswith("#"):
                continue
            if line.startswith("GPT:"):
                history.append({"role": "assistant", "content": line[4:].strip()})
            elif line.startswith(">") or line.startswith("3>") or line.startswith("4>"):
                if line.startswith(">>") or line.startswith("3>>") or line.startswith("4>>"):
                    history = []
                    history.append({"role": "user", "content": line[2:].strip()})
                else:
                    history.append({"role": "user", "content": line[1:].strip()})
                if line.startswith("3"):
                    last_talked = "3"
                elif line.startswith("4"):
                    last_talked = "4"
            else:
                if len(history) > 0:
                    history[-1]["content"] += "\n" + line.strip()
        return history, last_talked
