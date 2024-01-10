import os
import pprint

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LLMOverview:

    def __init__(self, model_name):

        self.setup()
        self.model_name = model_name
        self.temperatur = 0

    def setup(self):
        # .envファイルから環境変数を読み込む
        load_dotenv()

        # 環境変数にOpenAIのAPIキーを設定する
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    def generate_overview(self, mail_body):

        template1 = """
            次の文章を日本語で400文字程度に要約してください。
            文章：{text}
        """

        template2 = """
            次の文章を日本語で150文字程度に要約してください。
            文章：{text}
        """

        prompt1 = PromptTemplate(
            input_variables=['text'],
            template=template1,
        )

        prompt2 = PromptTemplate(
            input_variables=['text'],
            template=template2,
        )

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2000,
            chunk_overlap=0,
        )

        texts = text_splitter.split_text(mail_body)

        docs = [Document(page_content=t) for t in texts]

        llm = ChatOpenAI(
            temperature=self.temperatur,
            model_name=self.model_name
        )

        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=prompt1,
            combine_prompt=prompt2
        )

        result = chain.run(docs)

        return result


if __name__ == '__main__':

    sample_text = """
            設業や製造業を中心に広く利用されている CAD は、設計業務において重要な役割を果たしています。
            しかし、従来のワークステーション環境では、リモートワークが制約されたり、柔軟なリソースの拡張が難しかったりといった問題が見受けられます。
            また、建設業界では 2024 年 4 月からは 36 協定が適用されることもあり、CAD業務等も含めた働き方の改革が求められています。
            """

    overview_obj = LLMOverview(model_name="gpt-3.5-turbo")
    result = overview_obj.generate_overview(sample_text)

    pprint.pprint(result)
