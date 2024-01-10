# 导入必要的库和模块
from llama_index import ServiceContext, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain import OpenAI
from modules.chat_options import cmd_opts
from modules.context import Context

from llama_index.data_structs.node import NodeWithScore
from llama_index.response.schema import Response
from llama_index.utils import truncate_text

import gradio as gr
import os


# 定义CSS和Javascript路径
css = "style.css"
script_path = "scripts"

# 保存原始的gradio模板响应
_gradio_template_response_orig = gr.routes.templates.TemplateResponse

# 初始化index变量
index = None

# 定义加载索引的函数
def load_index():
    global index
     # 初始化LLM预测器（这里使用gpt-3.5-turbo模型）
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=cmd_opts.temperature, model_name=cmd_opts.model_name))
    
    # 构建服务上下文
    service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor,
                prompt_helper=PromptHelper(max_input_size=cmd_opts.max_input_size,
                max_chunk_overlap=cmd_opts.max_chunk_overlap,
                num_output=cmd_opts.num_output),
                chunk_size_limit=cmd_opts.chunk_size_limit
                )
    
    # 构建存储上下文
    storage_context = StorageContext.from_defaults(persist_dir=cmd_opts.persist_dir)
    
    # 加载索引
    index = load_index_from_storage(storage_context, service_context=service_context)

# 定义聊天函数
def chat(ctx, message, model_type, refFlag):
    global index
  
    # 检查索引是否已加载
    if not index:
        raise "index not loaded"
  
    # 限制对话轮次
    ctx.limit_round()
  
    # 构建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=cmd_opts.similarity_top_k,
        response_mode=model_type
    )

    # 发出查询并获取回应
    response = query_engine.query(message)
    
    # 打印回应
    print(response)
    
    # 初始化参考文档列表
    refDoc = []  
    
    # 遍历来源节点，获取参考文档
    for node in response.source_nodes:  
        if node.similarity is not None:  
            refDoc.append(pprint_source_node(node))
    
    # 根据是否需要显示引用，生成最终的回应
    if(refFlag):
        res = "".join([
            response.response,  
            "\n引用:\n",  
            "\n".join(refDoc)])
    else: 
        res = response.response
    
    # 更新对话历史
    ctx.append(message, res)
    ctx.refresh_last()
    
    # 返回对话历史
    return ctx.rh

# 定义打印来源节点的函数
def pprint_source_node(
    source_node, source_length: int = 350, wrap_width: int = 70
) -> str:
    source_text_fmt = truncate_text(source_node.node.get_text().strip(), source_length)
    return "".join([
        f'(相似度{source_node.score}) ',  
        "\nnode id:",
        source_node.doc_id,  
        "\n",
        source_text_fmt]) 

# 定义创建用户界面的函数
def create_ui():
    reload_javascript();
    with gr.Blocks(analytics_enabled=False) as chat_interface:
        _ctx = Context()
        state = gr.State(_ctx)
        with gr.Row():
            with gr.Column(scale=3):
                input=gr.inputs.Textbox(lines=7, label="请输入")
                model_type = gr.inputs.Radio(
                    choices=["tree_summarize", "compact", "simple_summarize", "refine", "generation"],
                    label="选择模型",
                    default="simple_summarize",
                )
                refFlag=gr.inputs.Checkbox(default=True, label="显示引用", optional=False)
                submit = gr.Button("发送", elem_id="c_generate")
        
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="c_chatbot", show_label=False).style(height=500)
                savebutton = gr.Button("保存", elem_id="c_save")
        
        # 设置对话窗的点击事件
        submit.click(chat, inputs=[
                state,
                input,
                model_type,
                refFlag
            ], outputs=[
                chatbot,
            ])
       
    return chat_interface

# 定义重新加载Javascript的函数
def reload_javascript():
    scripts_list = [os.path.join(script_path, i) for i in os.listdir(script_path) if i.endswith(".js")]
    javascript = ""

    for path in scripts_list:
        with open(path, "r", encoding="utf8") as js_file:
            javascript += f"\n<script>{js_file.read()}</script>"

    # 修改gradio的模板响应，添加Javascript
    def template_response(*args, **kwargs):
        res = _gradio_template_response_orig(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
