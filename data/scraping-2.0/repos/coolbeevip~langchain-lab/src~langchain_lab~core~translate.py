# Copyright 2023 Lei Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain import LLMChain, PromptTemplate
from langchain.chat_models.base import BaseChatModel
from langdetect import detect

from langchain_lab import logger
from langchain_lab.core.llm import TrackerCallbackHandler

# https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
LANGUAGES = {
    "English": {
        "lang": "en",
    },
    "Chinese": {
        "lang": "zh-cn",
    },
    "Korean": {"lang": "ko"},
    "Japanese": {
        "lang": "ja",
    },
}

translate_template = """Translate the text delimited by triple quotes into {language} language.
Only give me the output and nothing else. Do not wrap responses in quotes.
\"\"\"
{selection}
\"\"\"
"""


def detect_language(selection: str):
    lang = detect(selection[:50])
    for language, details in LANGUAGES.items():
        if details["lang"] == lang:
            logger.info(f"Detected language: {lang} translate to {language}")
            return language
    return "English"


def translate(
    selection: str,
    language: str,
    llm: BaseChatModel,
    callback: TrackerCallbackHandler = None,
):
    lang = detect(selection[:50])
    logger.info(f"Detected language: {lang} translate to {language}")
    if lang == LANGUAGES[language]["lang"]:
        return selection
    else:
        translate_prompt = PromptTemplate.from_template(translate_template)
        inputs = {"language": language, "selection": selection}

        chain = LLMChain(
            llm=llm,
            prompt=translate_prompt,
            callbacks=[callback],
        )
        response = chain.run(inputs)
        return response
