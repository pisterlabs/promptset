# █▀ █░█ ▄▀█ █▀▄ █▀█ █░█░█
# ▄█ █▀█ █▀█ █▄▀ █▄█ ▀▄▀▄▀

# Copyright 2023 t.me/shadow_modules
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# meta developer: @shadow_modules
# meta banner: https://i.imgur.com/fdEskim.jpeg


import openai
from .. import loader, utils
from telethon.tl.types import Message  # type: ignore


@loader.tds
class GPT2Mod(loader.Module):
    """ChatGPT в модуле"""

    strings = {
        "name": "ChatGPT",
        "wait": "<emoji document_id=5471981853445463256>🤖</emoji><b> GPT-2 is generating response, please wait</b>",
        "quest": "\n\n\n<emoji document_id=5819167501912640906>❔</emoji><b> Your question to GPT-2 was:</b> {args}",
        "args_err": "<emoji document_id=5215534321183499254>⛔️</emoji><b> You didn't ask a question GPT-2</b>",
        "conf_err": "<emoji document_id=5215534321183499254>⛔️</emoji><b> You didn't provide an api key for GPT-2</b>",
    }
    strings_ru = {
        "wait": "<emoji document_id=5471981853445463256>🤖</emoji><b> GPT-2 генерирует ответ, подождите</b>",
        "quest": "\n\n\n<emoji document_id=5819167501912640906>❔</emoji><b> Ваш вопрос к GPT-2 был:</b> {args}",
        "args_err": "<emoji document_id=5215534321183499254>⛔️</emoji><b> Вы не задали вопрос GPT-2</b>",
        "conf_err": "<emoji document_id=5215534321183499254>⛔️</emoji><b> Вы не указали api key для GPT-2</b>",
    }

    def __init__(self):
        self.config = loader.ModuleConfig(
            loader.ConfigValue(
                "api_key",
                None,
                lambda: "Api key for GPT-2",
                validator=loader.validators.Hidden(),
            ),
        )

    async def gptcmd(self, message: Message):
        """.gpt <args>"""
        args = utils.get_args_raw(message)
        if not args:
            await utils.answer(message, self.strings("args_err"))
            return
        if self.config["api_key"] is None:
            await utils.answer(message, self.strings("conf_err"))
            return
        await utils.answer(message, self.strings("wait").format(args=args))
        openai.api_key = self.config["api_key"]
        model_engine = "text-davinci-003"
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=args,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        response = completion.choices[0].text
        await utils.answer(message, response + self.strings("quest").format(args=args))
