from langchain.agents import ZeroShotAgent
from langchain.agents import  AgentExecutor
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
import markdown  
from IPython.display import Image, display  
from llm2ebm import feature_importances_to_text
from tool import get_tools
from prompt import suffix_no_df,suffix_with_df,get_prefix


#用md语法表示的图的字符串生成图
def md2img(text):
    # 使用Markdown库将Markdown文本转换为HTML  
    html_output = markdown.markdown(text)  
      
    # 解析HTML中的图片标签，并显示图片  
    def process_image_tags(html):  
        from bs4 import BeautifulSoup  
        soup = BeautifulSoup(html, 'html.parser')  
          
        # 找到所有的图片标签  
        img_tags = soup.find_all('img')  
          
        # 遍历图片标签，显示图片  
        for img in img_tags:  
            url = img['src']  
            alt_text = img.get('alt', '')  
              
            # 使用IPython.display模块的Image类显示图片  
            display(Image(url=url, alt=alt_text))  
      
    # 调用函数解析图片标签并显示图片  
    process_image_tags(html_output)  

def get_agent(llm,ebm,df = None,dataset_description = None,y_axis_description = None):
    
    #获取需要的ebm的属性
    feature_importances = feature_importances_to_text(ebm) 
    global_explanation = global_explanation = ebm.explain_global().data
    
    #获取prompt的prefix部分
    prefix = get_prefix(ebm,feature_importances,dataset_description,y_axis_description)
    
    #获取工具
    tools=get_tools(ebm)
    python_tool = tools[0]
    #获取prompt的suffix部分
    if df is not None:
        python_tool.python_repl.locals={"df": df,"ft_graph":global_explanation}
        input_variables=["input", "chat_history", "agent_scratchpad","df_head"]
        suffix = suffix_with_df
    else:
        python_tool.python_repl.locals={"ft_graph":global_explanation}
        input_variables=["input", "chat_history", "agent_scratchpad"]
        suffix = suffix_no_df
    
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    if 'df_head' in input_variables:
        prompt = prompt.partial(df_head=str(df.head().to_markdown()))

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain,tools=tools, verbose=True)
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory,handle_parsing_errors=True
    )
