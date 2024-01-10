from llama_index import download_loader
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
from langchain.chains.summarize import load_summarize_chain


summarize_prompt_template = """以下の文章を簡潔に要約してください。:

{text}

要約:"""

simplify_prompt_prefix = "元の文章の難しい表現を、元の意味を損なわないように注意して子ども向けのかんたんな表現に変換してください。語尾は「です」「ます」で統一してください。"

simplify_examples = [
    {
        "元の文章": "綿密な計画のもと、彼は革命を起こし、王朝の支配に終止符を打った。",
                "かんたんな文章": "よく考えられた計画で、彼は国の政治や社会の仕組みを大きく変えました。そして、王様の家族がずっと支配していた時代が終わりました。"
    },
    {
        "元の文章": "彼は無類の読書家であり、その博識ぶりは同僚からも一目置かれる存在だった。",
                "かんたんな文章": "彼はたくさんの本を読むのが大好きで、たくさんのことを知っています。友達も彼の知識を尊敬しています。"
    },
    {
        "元の文章": "彼女は劇団に所属し、舞台で熱演を繰り広げ、観客を魅了していた。",
                "かんたんな文章": "彼女は劇のグループに入っていて、舞台でとても上手に演じて、見ている人たちを楽しませています。"
    },
    {

        "元の文章": "宇宙の膨張は、エドウィン・ハッブルによって観測された銀河の運動から発見されました。",
                "かんたんな文章": "宇宙がどんどん広がっていることは、エドウィン・ハッブルさんがたくさんの星が集まった大きなものが動いていることを見つけることでわかりました。"
    }
]

simplify_example_formatter_template = """
元の文章: {元の文章}
かんたんな文章: {かんたんな文章}\n
"""

simplify_prompt_suffix = "元の文章: {input}\nかんたんな文章:"

simpify_system_message = "あなたは文章を子ども向けのかんたんな日本語に変換するのに役立つアシスタントです。"


class PdfSimplySummarizer():
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _load_documents(self, file_path: str) -> list:
        CJKPDFReader = download_loader("CJKPDFReader")
        loader = CJKPDFReader(concat_pages=False)

        documents = loader.load_data(file=file_path)
        langchain_documents = [d.to_langchain_format() for d in documents]
        return langchain_documents

    def _summarize(self, langchain_documents: list) -> str:
        summarize_template = PromptTemplate(
            template=summarize_prompt_template, input_variables=["text"])

        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=summarize_template,
            combine_prompt=summarize_template
        )

        summary = chain.run(langchain_documents)
        return summary

    def _simplify(self, summary: str) -> str:
        simplify_example_prompt = PromptTemplate(
            input_variables=["元の文章", "かんたんな文章"],
            template=simplify_example_formatter_template,
        )

        simplify_few_shot_prompt_template = FewShotPromptTemplate(
            examples=simplify_examples,
            example_prompt=simplify_example_prompt,
            prefix=simplify_prompt_prefix,
            suffix=simplify_prompt_suffix,
            input_variables=["input"],
            example_separator="\n\n",
        )

        simplify_few_shot_prompt = simplify_few_shot_prompt_template.format(
            input=summary)

        messages = [
            SystemMessage(content=simpify_system_message),
            HumanMessage(content=simplify_few_shot_prompt),
        ]
        result = self.llm(messages)

        return result.content

    def run(self, file_path: str):
        langchain_documents = self._load_documents(file_path)
        summary = self._summarize(langchain_documents)
        result = self._simplify(summary)
        return result
