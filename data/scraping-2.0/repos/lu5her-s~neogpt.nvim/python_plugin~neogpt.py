import pynvim
import openai
import os

# openai.api_key = "dEXMiu282AKk6PZO5JvJvpGXMjrqflV_YzTQ9yX1DdY"
# openai.api_base = "https://chimeragpt.adventblocks.cc/api/v1"


# @pynvim.function
# def completions(prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         # max_tokens=1024,
#         # n=1,
#         # stop=None,
#         temperature=0.5,
#         stream=True,
#     )
#     return response


@pynvim.plugin
class OpenAICompletionPlugin(object):
    def __init__(self, nvim):
        self.nvim = nvim
        openai.api_key = "dEXMiu282AKk6PZO5JvJvpGXMjrqflV_YzTQ9yX1DdY"
        openai.api_base = "https://chimeragpt.adventblocks.cc/api/v1"

    @pynvim.command('Ai', nargs='*', range='')
    def ai_command(self, args, range):
        prompt = "".join(args)
        completion = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=20,
            temperature=0.8,
            n=1
        )
        completion_text = completion.choices[0].text.strip()
        self.nvim.command(f"normal! i{completion_text}<Esc>")

if __name__ == '__main__':
    nvim = pynvim.attach("socket", path=os.environ.get("NVIM_LISTEN_ADDRESS"))
    nvim.register_plugin(OpenAICompletionPlugin(nvim))
    nvim.command("command! -nargs=* Ai :python3 OpenAICompletionPlugin.ai_command(<q-args>)")
    nvim.command("set completefunc=<python3complete.openai_complete")

    nvim.run_loop()
