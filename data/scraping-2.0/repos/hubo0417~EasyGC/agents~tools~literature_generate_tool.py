from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from agents.base_tools.base_tools import functional_Tool
from configs.prompt_config import TOOLS_CONTENT_TOOL_PROMPT_TEMPLATE
from configs.base_config import BASE_FILE_PATH
from utils.pipeline import Pipeline_Item, Pipeline_Process_Task
from utils.generate.note_generate_utils_literature import Note_Generate_Utils_Literature


class Literature_Generate_Tool(functional_Tool):
    llm: BaseLanguageModel
    name = "分镜脚本制作工具"
    description = "根据输入内容制作视频分镜脚本的工具，输入关键词，输出一篇相关内容的分镜脚本"
    qa_template = TOOLS_CONTENT_TOOL_PROMPT_TEMPLATE
    prompt = PromptTemplate(input_variables=["query"], template=qa_template)
    llm_chain: LLMChain = None
    base_file_path: str = BASE_FILE_PATH

    def parser_output(self, output: str):
        if not output:
            raise ValueError("未能获取到正确的输入信息")
        # 按换行符分割字符串
        lines = output.split('\n')
        article = lines[0].split('：')[1].strip()
        if "、" in article:
            article = article.split('、')[0].strip()
        if "，" in article:
            article = article.split('，')[0].strip()
        return article

    def _call_func(self, query) -> str:
        # 1、提取关键字
        if not query:
            raise ValueError("未能获取到正确的用户信息")
        self.get_llm_chain()
        # 通过用户输入识别出用户输入的关键字，用于到互联网平台搜索内容
        article = self.parser_output(self.llm_chain.predict(query=query))
        # 初始文章生成工具
        util = Note_Generate_Utils_Literature(
            llm=self.llm, base_file_path=self.base_file_path)

        # 初始化管道模型
        pipe = Pipeline_Process_Task()

        # 注册带备选方法的参考游记获取方法
        item = Pipeline_Item(Obj=util,
                             Method="read_note_list",
                             Is_Use_Pre_Result=False,
                             Params={
                                 "top": 1,
                                 "article": article
                             })
        pipe.add_item(item=item)

        # 注册文章生成方法
        item = Pipeline_Item(Obj=util,
                             Method="summary_response",
                             Is_Use_Pre_Result=True)
        pipe.add_item(item=item)

        # 执行管道方法
        sumary_result = pipe.execute_pipeline()
        if pipe.is_contain_halt_task is True:
            return {"sumary_result": sumary_result, "pipe": pipe}
        else:
            return {"sumary_result": sumary_result}

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
