import prompt_templates
from langchain.prompts import PromptTemplate
from langchain.chat_models import azure_openai
from langchain.chains import LLMChain

class InstructionGenerator:
    # OPENAI_API_KEY = "44a43d50221443c690df1c2fd0f9cc14"
    # OPENAI_API_BASE = "https://xiaoma-openai.openai.azure.com/"
    OPENAI_API_KEY = '5fae23bf317545e4b2d5c1fd8a8eb070'
    OPENAI_API_BASE = 'https://xiaoma-ai-jp.openai.azure.com/'

    OPENAI_API_VERSION = "2023-05-15"
    OPENAI_API_TYPE = "azure"
    REQUEST_TIMEOUT = 300
    input_variables = ["initial_prompt"]
    
    def __init__(
            self, 
            prompt_template: str, 
            prefix = '',
            examples = [''],
            example_separator = '',
            suffix = '',
            TEMPERATURE = 0,
            DEPLOYMENT_NAME = 'turbo35'
    ):
        # Prompt相关参数
        self.prompt_template = prompt_template
        ## Few Shot Prompt 相关参数
        self.prefix = prefix
        self.examples = examples
        self.example_separator = example_separator
        self.suffix = suffix
        # LLM相关参数
        self.DEPLOYMENT_NAME = DEPLOYMENT_NAME
        self.TEMPERATURE = TEMPERATURE

    def generate(self, initial_prompt):
        if self.examples == ['']:
            templatized_prompt = PromptTemplate(
                input_variables = InstructionGenerator.input_variables,
                template=self.prompt_template
            )
        else: # examples不为空，说明是Few Shot Prompt 的情况
            templatized_prompt = PromptTemplate.from_examples(
                input_variables=InstructionGenerator.input_variables,
                prefix=self.prefix,
                examples=self.examples,
                example_separator=self.example_separator,
                suffix=self.suffix
            )
        llmchain = LLMChain(
            prompt = templatized_prompt,
            llm = azure_openai.AzureChatOpenAI(
                deployment_name = self.DEPLOYMENT_NAME,
                temperature = self.TEMPERATURE,
                request_timeout = InstructionGenerator.REQUEST_TIMEOUT,
                openai_api_key = InstructionGenerator.OPENAI_API_KEY,
                openai_api_base = InstructionGenerator.OPENAI_API_BASE,
                openai_api_version = InstructionGenerator.OPENAI_API_VERSION,
                openai_api_type = InstructionGenerator.OPENAI_API_TYPE
            )
        )
        try:
            return llmchain.predict(initial_prompt = initial_prompt)
        except Exception as e: # 审核不通时输出当前生成的情况，并跳过
            print('----报错啦----')
            print(f"Caught an error: {e}")
            print('----待升级指令----')
            print(initial_prompt)
            print('----升级类型----')
            print(self.prompt_template)
    
add_constraints = InstructionGenerator(
    prompt_template=prompt_templates.add_constraints
)
# print(add_constraints.generate("1+1=?")) #测试用

deepen = InstructionGenerator(
    prompt_template=prompt_templates.deepening
)
# print(deepen.generate("1+1=?")) #测试用

concretize = InstructionGenerator(
    prompt_template=prompt_templates.concretizing
)
# print(concretize.generate("1+1=?")) #测试用

increase_reasoning = InstructionGenerator(
    prompt_template=prompt_templates.increase_reasoning
)
# print(increase_reasoning.generate("1+1=?")) #测试用

enhance_diversity = InstructionGenerator(
    prompt_template=prompt_templates.enhance_diversity,
    TEMPERATURE = 0.5
)
# print(enhance_diversity.generate("1+1=?")) #测试用
