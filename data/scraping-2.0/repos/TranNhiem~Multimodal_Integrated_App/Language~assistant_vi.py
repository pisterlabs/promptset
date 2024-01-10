'''
@TranNhiem 2023/06/06

This code is the first version of the LLM (Language Model) Assistant using the LangChain Tool. 

1. Loading The Finetuned Instruction LLM Model
    + Using Checkpoint 
    + Using HuggingFace Model Hub

2. Using The LangChain Tool 
    + LangChain Memory System (with Buffer Memory we no need openAI API)
    + LangChain VectorStore System (You Need OpenAI Embedding for this)

3. Connect to Vector Database (FAISS) for Indexing and Searching

4. Further Design LLMs for Auto Agent (AI Agent) LLM using Langchain Tool 

    4.1 Indexing LLM (Augmented Retrieved Documents )
    4.2 Agent LLM (Design & Invent New thing for Human)

5. Future work: Integrate Huggingface Text-generation-inference 
    + https://github.com/huggingface/text-generation-inference 
'''

import os
import torch
from langchain.memory import VectorStoreRetrieverMemory
import faiss
# from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
import os
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory,  ConversationBufferWindowMemory
from langchain.chains import ConversationChain 
from langchain import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.docstore import InMemoryDocstore

from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai

## Under Testing Azure API 

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://sslgroupservice.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

## -----------------------------------------------------------------
## Loading LLM Models (Loading from Checkpoint or HuggingFace Model Hub)
## -----------------------------------------------------------------
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
## Loading The FineTuned LoRa Adapter Model 
from peft import PeftModel, PeftConfig
import bitsandbytes as bnb
from transformers import  AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

## --------------------------Loading model from Checkpoint ---------------------------------------

##Helper Read the HTML title 
def read_content(file_path) :
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

## Helper Function to Load 8Bit Model from Checkpoint
def find_all_linear_names(bits, model):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_model_from_checkpoint(model_name, checkpoint_path=None):

    ## --------------------------Setting Update to Load Baseline Pretrained Model First---------------------------------------

    if model_name=="Alpha-7B1":
        base_model="bigscience/bloomz-7b1"
        if checkpoint_path is None:
            checkpoint_path="/data/rick/pretrained_weights/BLOOMZ/Alpaca_CN_500K/7b1_Bloomz_based/checkpoint-44000/"
    elif model_name=="Alpha-1B7":
        base_model="bigscience/bloomz-1b7"
        if checkpoint_path is None:
            checkpoint_path="/data/rick/pretrained_weights/BLOOMZ/Alpaca_CN_500K/1b7_Bloomz_based/checkpoint-48400/"
    else:
        raise ValueError(f"This Model {model_name} is not Supported")

    
    cache_dir_="/data/rick/pretrained_weights/BLOOMZ/"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache_dir_,
        #load_in_8bit=True, ## Currently RTX 1080Ti not working 
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model,  torch_dtype=torch.float16,cache_dir=cache_dir_,)#

    model = prepare_model_for_int8_training(model)
    # bits=16
    # modules = find_all_linear_names(bits, model)
        
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules= ["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    checkpoint_name = os.path.join(checkpoint_path, "pytorch_model.bin")
    print(f"Restarting from {checkpoint_name}")
    adapters_weights = torch.load(checkpoint_name)
    for name, param in model.named_parameters():
        #if name == "base_model.model.transformer.h.1.self_attention.query_key_value.lora_A.default.weight":
        weight_tensor = adapters_weights[name]  # Get the corresponding tensor from weight_value
        param.data = weight_tensor  # Replace the parameter tensor with weight_tensor
    
   
    #del (adapters_weights)
    model.to('cuda')
    return model, tokenizer


# base_model_1="huggyllama/llama-13b"

# checkpoint_path_1="/data/rick/pretrained_weights/LLaMA/alpaca_gpt4_llama_13B/checkpoint-2800"


# base_model_2="bigscience/bloomz-7b1"

# checkpoint_path_2="/data/rick/pretrained_weights/BLOOMZ/Alpaca_gpt4all_7b1/checkpoint-8400"

base_model_3="bigscience/bloomz-1b7"

checkpoint_path_3="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/Finetuned_Weights/Bloomz_Vi_alpaca_GPT4All_90k/checkpoint-9800"


cache_dir_="/data/rick/pretrained_weights/BLOOMZ/"
cache_dir_llama="/data/rick/pretrained_weights/LLaMA/"

# model_1 = AutoModelForCausalLM.from_pretrained(
#     base_model_1,
#     cache_dir=cache_dir_llama,
#     #load_in_8bit=True, ## Currently RTX 1080Ti not working 
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# tokenizer_1 = AutoTokenizer.from_pretrained(base_model_1,  torch_dtype=torch.float16,cache_dir=cache_dir_,)#

# model_1 = prepare_model_for_int8_training(model_1)
# # bits=16
# # modules = find_all_linear_names(bits, model)
    
# config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules= ["g_proj", "v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model_1 = get_peft_model(model_1, config)

# checkpoint_name_1 = os.path.join(checkpoint_path_1, "pytorch_model.bin")
# print(f"Restarting from {checkpoint_name_1}")
# adapters_weights_1 = torch.load(checkpoint_name_1)
# for name, param in model_1.named_parameters():
#     #if name == "base_model.model.transformer.h.1.self_attention.query_key_value.lora_A.default.weight":
#     weight_tensor = adapters_weights_1[name]  # Get the corresponding tensor from weight_value
#     param.data = weight_tensor  # Replace the parameter tensor with weight_tensor


# #del (adapters_weights)
# model_1.to('cuda')

# model_2 = AutoModelForCausalLM.from_pretrained(
#     base_model_2,
#     cache_dir=cache_dir_,
#     #load_in_8bit=True, ## Currently RTX 1080Ti not working 
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# tokenizer_2 = AutoTokenizer.from_pretrained(base_model_2,  torch_dtype=torch.float16,cache_dir=cache_dir_,)#

# model_2 = prepare_model_for_int8_training(model_2)
# # bits=16
# # modules = find_all_linear_names(bits, model)
    
# config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules= ["query_key_value"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model_2 = get_peft_model(model_2, config)

# checkpoint_name_2 = os.path.join(checkpoint_path_2, "pytorch_model.bin")
# print(f"Restarting from {checkpoint_name_2}")
# adapters_weights_2 = torch.load(checkpoint_name_2)
# for name, param in model_2.named_parameters():
#     #if name == "base_model.model.transformer.h.1.self_attention.query_key_value.lora_A.default.weight":
#     weight_tensor = adapters_weights_2[name]  # Get the corresponding tensor from weight_value
#     param.data = weight_tensor  # Replace the parameter tensor with weight_tensor

# #del (adapters_weights)
# model_2.to('cuda')

model_3 = AutoModelForCausalLM.from_pretrained(
    base_model_3,
    cache_dir=cache_dir_,
    #load_in_8bit=True, ## Currently RTX 1080Ti not working 
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer_3 = AutoTokenizer.from_pretrained(base_model_3,  torch_dtype=torch.float16,cache_dir=cache_dir_,)#

model_3 = prepare_model_for_int8_training(model_3)
# bits=16
# modules = find_all_linear_names(bits, model)
    
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules= ["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model_3 = get_peft_model(model_3, config)

checkpoint_name_3 = os.path.join(checkpoint_path_3, "pytorch_model.bin")
print(f"Restarting from {checkpoint_name_3}")
adapters_weights_3 = torch.load(checkpoint_name_3)
for name, param in model_3.named_parameters():
    #if name == "base_model.model.transformer.h.1.self_attention.query_key_value.lora_A.default.weight":
    weight_tensor = adapters_weights_3[name]  # Get the corresponding tensor from weight_value
    param.data = weight_tensor  # Replace the parameter tensor with weight_tensor


#del (adapters_weights)
model_3.to('cuda')


## --------------------------Loading model from Checkpoint ---------------------------------------
def load_model_from_hub_or_local_path(model, model_name, model_path=None): 
   
    if model_path is not None:
        print("Loading Model from Local Path")
        model_name=model_path
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto', use_auth_token=True)#load_in_8bit=True
    else:
        print("Loading Model from HuggingFace Model Hub") 
        model=PeftModel.from_pretrained(model, model_name)
    
    return model 


# checkpoint_path_1b1="/data/rick/pretrained_weights/BLOOMZ/Alpaca_CN_500K/1b1_Bloomz_cn_based/checkpoint-23000/" # Path to save model weight to Disk
# # checkpoint_path_1b7="/content/drive/MyDrive/Generative_Model_Applications/checkpoint-48400/"
# # checkpoint_path_7b1="/content/drive/MyDrive/Generative_Model_Applications/checkpoint-48400/"
# model=load_model_from_checkpoint(model_name, checkpoint_path_1b1)

## -----------------------------------------------------------------
## New Gradio WebAPP interface For New Feature and Advance interface 
## -----------------------------------------------------------------

# _DEFAULT_TEMPLATE = """ Below is an instruction that describes a task. Please provide a response that appropriately completes the request, considering both the relevant information discussed in the ongoing conversation and disregarding any irrelevant details. If the AI does not know the answer to a question, it truthfully says it does not know.

# ## Current conversation:{history}

# ## prompt: {input}

# ## Response:
# """
#Hãy viết một phản hồi thích hợp cho chỉ dẫn dưới đây.
# _DEFAULT_TEMPLATE="""
# Dưới đây là một cuộc trò chuyện thân thiện giữa Con người và AI được gọi là Vietnamese_LLMs. AI này có tính cách hóm hỉnh và cung cấp nhiều chi tiết cụ thể từ ngữ cảnh của nó. AI không biết câu trả lời cho một câu hỏi, nó trung thực nói rằng nó không biết.
# Cuộc trò chuyện hiện tại:
# {history}

# ## prompt: {input}

# ## response: """

_DEFAULT_TEMPLATE="""
Below is an instruction that describes a task. Write a response that appropriately completes the request. History of the conversation {history} \n\n### prompt:\n{prompt}\n\n### response:\n"""



# _DEFAULT_TEMPLATE="""The following is a friendly conversation between a human and an AI called Alpaca. The AI is talkative and provides lots of specific details from its context in Vietnamese Language. If the AI does not know the answer to a question, it truthfully says it does not know all in Vietnamese. 

# Current conversation:
# {history}
# Human: {input}
# AI:"""

# Dưới đây là một hướng dẫn mô tả một nhiệm vụ. Vui lòng cung cấp một phản hồi phù hợp hoàn thành yêu cầu, xem xét cả thông tin liên quan được thảo luận trong cuộc trò chuyện và bỏ qua bất kỳ chi tiết không liên quan nào. Nếu AI không biết câu trả lời cho một câu hỏi, nó trung thực nói rằng nó không biết.

# ### Thông tin liên quann trong cuộc đối thoại:{history}
##### Thông tin liên quann trong cuộc đối thoại:{history}


# _DEFAULT_TEMPLATE = """ 
# Below is an instruction that describes a task.
# Write a response that appropriately completes the request.\n\n



# ### prompt: {input}

# ### response:
# """

prompt_template_ = PromptTemplate(
    input_variables=[  "input","history" ], template=_DEFAULT_TEMPLATE
)


# base_model="Alpha-1B1"
# model, tokenizer=load_model_from_checkpoint(base_model,checkpoint_path=None )
# model.to('cuda')

import gradio as gr
with gr.Blocks() as demo:
    #gr.Markdown("""<h1><center> SIF-LLM Assistant (Alpha Released)  </center></h1>""")
    gr.HTML(read_content("/data/rick/LLM/Multimodal_Integrated_App/Language/html_header.html"))
    with gr.Row(scale=4, min_width=300, min_height=100):
        with gr.Column():
          base_model_input = gr.Dropdown(choices= ["Alpha-7B1", "Alpha-1B7","LLaMA_13B" ], value="LLaMA_13B", label="Choosing LLM", show_label=True)
        with gr.Column():
          conversation_style = gr.Dropdown( choices=["More Creative", "More Balance"], value="More Creative", label="Conversation Style", show_label=True)
                    
    chatbot = gr.Chatbot(label="Assistant").style(height=500)
    
    with gr.Row():
        message = gr.Textbox(show_label=False, placeholder="Enter your prompt and press enter", visible=True)
    state = gr.State()

    
    # max_token_limit=40 - token limits needs transformers installed
    #memory= ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)
    #memory= ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)
    ## VectorStore Memory
    embedding_size = 1536 #1536 # Dimensions of the OpenAIEmbeddings
    index = faiss.IndexFlatL2(embedding_size)
    embedding_fn = OpenAIEmbeddings(deployment="text-embedding-ada-002",  chunk_size=10).embed_query
    vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=4))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    
    ## Summary Buffer Memory 
    
    #memory = ConversationBufferWindowMemory(k=4)

    ## VectorStore Summary Buffer Memory 


    def respond(message, chat_history,repetition_penalty=1.2, temperature=0.6, top_p=0.95, penalty_alpha=0.4,top_k=20, max_output_tokens=512, base_model=base_model_input,conversation_style="More Creative"):
        
        # Setup VectorStore Memory OR Buffer Memory 


        ## Most Simple one without require any Extra LLM for Embedding or Summary
        # memory = ConversationBufferWindowMemory(k=4)



        ## Setting Up Pretrained Model 
        # checkpoint_path_7b1="/content/drive/MyDrive/Generative_Model_Applications/checkpoint-28200/"
        if base_model=="LLaMA_13B":
            model=model_1
            tokenizer=tokenizer_1
        
        elif base_model=="Alpha-7B1":
            model=model_2
            tokenizer=tokenizer_2
        elif base_model=="Alpha-1B7":
            model=model_3
            tokenizer=tokenizer_3
        
        #model, tokenizer=load_model_from_checkpoint(base_model,checkpoint_path=None )
        if conversation_style == "More Creative":
            #with torch.cuda.amp.autocast():
            pipe = pipeline(
                "text-generation",
                model=model, 
                tokenizer=tokenizer, 
                max_length=max_output_tokens,
                ## Contrastive Search Setting
                penalty_alpha=penalty_alpha, 
                top_k=top_k,
                repetition_penalty=repetition_penalty)
            #del(model)

        else: 
            #with torch.cuda.amp.autocast():
            pipe = pipeline(
                "text-generation",
                model=model, 
                tokenizer=tokenizer, 
                max_length=max_output_tokens,
                ## Beam Search
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
                )
            #del(model)
        
        local_llm = HuggingFacePipeline(pipeline=pipe)

        
        conversation_with_summary = ConversationChain(
            llm=local_llm, 
             memory=memory, 
            verbose=True, 
            prompt= prompt_template_, 
        )
        bot_message = conversation_with_summary.predict(input=message)
        message= "👤: "+ message
        bot_message= "Assistant 😃: "+ bot_message
        chat_history.append((message, bot_message))
        #time.sleep(1)
        return "", chat_history
     
     
    ## For Setting Hyperparameter 
    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        #It is a value between 1.0 and infinity, where 1.0 means no penalty
        repetition_penalty = gr.Slider(
            minimum=1.0,
            maximum=1.9,
            value=1.2,
            step=0.1,
            interactive=True,
            label="repetition_penalty",
        )
        
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        ## two values penalty_alpha & top_k use to set Contrastive Decoding for LLM 
        penalty_alpha = gr.Slider(
            minimum=0.001,
            maximum=1.0,
            value=0.4,# Values 0.0 mean equal to gready_search
            step=0.05,
            interactive=True,
            label="penalty_alpha",
        )
        top_k = gr.Slider(
            minimum=5.0,
            maximum=40.0,
            value=20,## Top number of candidates 
            step=2,
            interactive=True,
            label="Top_k",
        )
        max_output_tokens = gr.Slider(
            minimum=100,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    message.submit(respond, inputs=[message, chatbot, repetition_penalty,temperature, top_p, penalty_alpha, top_k, max_output_tokens,base_model_input, conversation_style], outputs=[message, chatbot], queue=False, )
    #gr.Interface(fn=respond, inputs=[message, chatbot, repetition_penalty, temperature, top_p, penalty_alpha, top_k, max_output_tokens, base_model_input, conversation_style], outputs=[message, chatbot], title="Alpha Assistant via SIF LLM", server_port=1234).launch(share=True)
    gr.HTML(
                """
                <div class="footer">
                    <p style="align-items: center; margin-bottom: 7px;" >

                    </p>
                    <div style="text-align: Center; font-size: 1.5em; font-weight: bold; margin-bottom: 0.5em;">
                        <div style="
                            display: inline-flex; 
                            gap: 0.6rem; 
                            font-size: 1.0rem;
                            justify-content: center;
                            margin-bottom: 10px;
                            ">
                        <p style="align-items: center; margin-bottom: 7px;" >
                            Hallooo (it's me!! Rick) i'm in charge of this development: 
                            <a Các Góp ý và ý kiến của các bạn rất ý nghĩa cho việc phát triển trong tươngg lại.</a>
                            <a href="https://docs.google.com/spreadsheets/d/1KO8gbbtsgk26oHinq_L58oaZLsaMGWIDS4Qjj3BYKEY/edit?usp=sharing" style="text-decoration: underline;" target="_blank"> 🙌  Share your feedback here </a> ; 
                            <a href="https://docs.google.com/presentation/d/1UkXsPcigKQxs9B2UDt_691LloCqiuVT7nUtLod5jG74/edit?usp=sharing" style="text-decoration: underline;" target="_blank"> 🙌 or upload screenshot images here</a> 
                        </p>
                       
                        </div>
                    <div class="footer">
                    <p style="align-items: center; margin-bottom: 7px;" >

                    </p>
                    <div style="text-align: Center; font-size: 1.5em; font-weight: bold; margin-bottom: 0.5em;">
                        <div style="
                            display: inline-flex; 
                            gap: 0.6rem; 
                            font-size: 1.0rem;
                            justify-content: center;
                            margin-bottom: 10px;
                            ">
                        <p style="align-items: center; margin-bottom: 7px;" >
                            Demo: @TranNhiem 🙋‍♂️ kết nối với Nhiệm Qua : 
                        <a href="https://www.linkedin.com/feed/" style="text-decoration: underline;" target="_blank"> 🙌 Linkedin</a> ;  
                            <a href="https://twitter.com/TranRick2" style="text-decoration: underline;" target="_blank"> 🙌 Twitter</a> ; 
                            <a href="https://www.facebook.com/jean.tran.336" style="text-decoration: underline;" target="_blank"> 🙌 Facebook</a> 
                        </p>
                        </p>
                        <p style="align-items: center; margin-bottom: 7px;" >
                        <a Demo này dựa trên 2 model lớn Bloom và LLaMA LLMs.</a>
                        </p>
                        </div>
     
        
                """
            )
            
demo.queue()
demo.launch(debug=True, server_port=1234, share=True)
