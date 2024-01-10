# 创建一些示例
samples = [
    {
        "flower_type": "玫瑰",
        "occasion": "爱情",
        "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
    },
    {
        "flower_type": "康乃馨",
        "occasion": "母亲节",
        "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
    },
    {
        "flower_type": "百合",
        "occasion": "庆祝",
        "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
    },
    {
        "flower_type": "向日葵",
        "occasion": "鼓励",
        "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
    }
]
from langchain.prompts.prompt import PromptTemplate

# 生成例子的模板
template_str = "鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}"
sample_template = PromptTemplate.from_template(template_str)

from langchain.prompts.example_selector.semantic_similarity import SemanticSimilarityExampleSelector
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# 样例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=samples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=1,
)

from langchain.prompts.few_shot import FewShotPromptTemplate

# 生成整个模板
few_shot_template = FewShotPromptTemplate(
    # examples=samples,  # 样例，包含多份数据，每份数据都可以填充样例模板
    example_selector=example_selector,  # 样例选择器，已经包含了样例
    example_prompt=sample_template,  # 样例模板
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",  # 问题模板
    input_variables=["flower_type", "occasion"],  # 问题模板中的变量名
)
# 格式化获得输入
# 加了样例选择器后，只选择了一个样例
few_shot_input = few_shot_template.format(flower_type="牵牛花", occasion="努力")

from langchain.llms.openai import OpenAI

model = OpenAI(model_name='text-davinci-003')
result = model(prompt=few_shot_input)
print(result)
