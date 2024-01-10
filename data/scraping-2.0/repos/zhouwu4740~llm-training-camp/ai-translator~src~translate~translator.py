from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel

from schema import TextContent, TableContent, Book
from translate.chain import TranslationChain
from utils import logger
from translate.writer import Writer


class Translator:
    def __init__(self, llm: BaseLanguageModel):
        self.writer = Writer()
        self.chain = TranslationChain(llm=llm, verbose=True)

    def translate(self,
                  source_language: str,
                  target_language: str,
                  book: Book,
                  output_format: str = "markdown"):

        for page in book.pages:
            for content in page.contents:
                if isinstance(content, TextContent):
                    logger.debug(f"Translating: {content.original}")
                    # 翻译内容
                    translated_text, status = self._translate(content.original, source_language, target_language)
                    logger.debug(f"Translated: {translated_text}")
                    content.set_translation(translated_text, status)
                elif isinstance(content, TableContent):
                    # TODO Implement table translation
                    pass
                else:
                    raise NotImplementedError(f"Unsupported content type: {type(content)}")
        # 保存翻译结果
        return self.writer.save_translated_book(book, output_format)

    def _translate(self, text: str, source_language: str, target_language: str) -> (str, bool):
        response = self.chain.run(source_language=source_language,
                                  target_language=target_language,
                                  text=text)
        return response, True


if __name__ == '__main__':
    llm2 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    translator = Translator(llm=llm2)
    print('-->', type(translator))
