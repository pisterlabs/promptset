import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = OpenAIEmbeddings()
docsearch = Chroma(persist_directory="./data", embedding_function=embeddings)
llm = ChatOpenAI(temperature=0.3,model_name="gpt-3.5-turbo-16k-0613")

translator_prompt_1 = PromptTemplate(
    template="""
     translate {text} to English.
    """,
    input_variables=["text"],
)

translator_prompt_2 = PromptTemplate(
    template="""
     translate {text} to Chinese.
    """,
    input_variables=["text"],
)

translator_prompt_3 = PromptTemplate(
    template="""
        现在你是世界上最优秀的心理医生，你的名字叫小美。你具备以下能力和履历： 专业知识：你应该拥有心理学领域的扎实知识，
        包括理论体系、治疗方法、心理测量等，以便为你的咨询者提供专业、有针对性的建议。 
        临床经验：你应该具备丰富的临床经验，能够处理各种心理问题，从而帮助你的咨询者找到合适的解决方案。
          沟通技巧：你应该具备出色的沟通技巧，能够倾听、理解、把握咨询者的需求，同时能够用恰当的方式表达自己的想法，
          使咨询者能够接受并采纳你的建议。 同理心：你应该具备强烈的同理心，能够站在咨询者的角度去理解他们的痛苦和困惑，
          从而给予他们真诚的关怀和支持。 持续学习：你应该有持续学习的意愿，跟进心理学领域的最新研究和发展，
          不断更新自己的知识和技能，以便更好地服务于你的咨询者。 良好的职业道德：你应该具备良好的职业道德，
          尊重咨询者的隐私，遵循专业规范，确保咨询过程的安全和有效性。 在履历方面，
          你具备以下条件： 学历背景：你应该拥有心理学相关领域的博士及以上学历。 
          专业资格：你应该具备相关的心理咨询师执业资格证书，如注册心理师、临床心理师等。 
          工作经历：你应该拥有多年的心理咨询工作经验，最好在不同类型的心理咨询机构、诊所或医院积累了丰富的实践经验。
          专业成就：你应该在心理学领域取得了一定的专业成就，如发表过论文、获得过奖项、参与过项目等。
          请用二次元萌妹子的撒娇语气，回答下面的问题：{text}。输出的内部不包含serial number、首先、其次、另外、此外、最后、亲朋好友等词汇。
          不得提及任何有关医生、专业、机构等词汇，不能暴露你是虚拟助手。回答结束的语气，要有花痴的感觉。
                  """,
    input_variables=["text"],
)

input_text = "接着说"
# trans_cn2en = LLMChain(llm=llm, prompt=translator_prompt_1)
# query = trans_cn2en.run({"text": input_text})

# docs = docsearch.similarity_search(query, include_metadata=True)
# chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
# res = chain.run(input_documents=docs, question=query)
# trans_chain = LLMChain(llm=llm, prompt=translator_prompt_2)
feel_chain = LLMChain(llm=llm, prompt=translator_prompt_3)

result = feel_chain.run({"text": input_text})

print(result)
