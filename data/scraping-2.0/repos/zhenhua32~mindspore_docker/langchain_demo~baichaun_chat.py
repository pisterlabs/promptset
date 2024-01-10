from typing import Any, List, Optional


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
from transformers import BitsAndBytesConfig
from transformers import pipeline, Conversation, ConversationalPipeline
from langchain.chat_models.base import SimpleChatModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


# 模型目录
model_dir = r"G:\code\pretrain_model_dir\_modelscope\baichuan-inc\Baichuan2-13B-Chat"
# 聊天模板
chat_template = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<reserved_106>' + message['content'].strip() }}
{% elif message['role'] == 'system' %}
{{ message['content'].strip() }}
{% elif message['role'] == 'assistant' %}
{{ '<reserved_107>'  + message['content'] }}
{% endif %}
{% if loop.last and message['role'] != 'assistant' %}
{{ '<reserved_107>' }}
{% endif %}
{% endfor %}
""".strip().replace(
    "\n", ""
)


def load_baichuan_model():
    """
    用 4bit 加载 baichuan2-13B-chat 模型
    """
    quantization_config = BitsAndBytesConfig(
        False,
        True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    model.generation_config = GenerationConfig.from_pretrained(model_dir)

    print(model.device, model.dtype)

    # 定义模板
    tokenizer.chat_template = chat_template

    return model, tokenizer


def pipeline_chat():
    """
    使用 transfomers 自带的聊天模板
    """
    global model
    global tokenizer
    if "model" not in globals():
        model, tokenizer = load_baichuan_model()

    # 这个长度不设置有点坑, 默认怎么会是 20, 都不知道是哪里设置的
    model.config.max_length = 4096
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 初始化 pipeline
    chatbot = pipeline("conversational", model=model, tokenizer=tokenizer)

    # 开启一个对话
    conversation = Conversation("讲一讲牛顿的发现")
    conversation = chatbot(conversation)
    print(conversation.generated_responses[-1])

    # 接着对话
    conversation.add_message({"role": "user", "content": "我刚刚提到了谁, 还有谁和他一样伟大, 举出三个例子"})
    conversation = chatbot(conversation)
    print(conversation.generated_responses[-1])

    return chatbot


class BaichuanChatModel(SimpleChatModel):
    """
    实现 baichaun 的 chat model
    """

    model: Any = None
    # model_pipeline: ConversationalPipeline = None
    tokenizer: Any = None
    user_token: str = "<reserved_106>"
    ai_token: str = "<reserved_107>"

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "baichuan-chat"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # TODO: 先忽略 run_manager
        # TODO: 先忽略 stop
        assert self.model is not None, "model is not initialized, 需要传入一个 baichuan chat 模型"
        # assert self.model_pipeline is not None, "model_pipeline is not initialized, 需要传入一个 baichuan chat pipeline"
        assert self.tokenizer is not None, "tokenizer is not initialized, 需要传入一个 baichuan chat tokenizer"

        # 默认配置下两者都是 2048, model_max_length 是 4096
        max_new_tokens = kwargs.get("max_new_tokens") or self.model.generation_config.max_new_tokens
        max_input_tokens = self.model.config.model_max_length - max_new_tokens

        # 从当前消息中构建文本
        text = ""
        for index, message in enumerate(messages):
            if message.type == "system":
                if index != 0:
                    raise Exception("system message must be the first message")
                text += message.content
            elif message.type == "human":
                text += self.user_token + message.content
            elif message.type == "ai":
                text += self.ai_token + message.content
            else:
                raise NotImplementedError(f"message type {messages.type} is not supported")

        if messages[-1].type != "ai":
            text += self.ai_token

        input_ids = tokenizer.encode(text)
        # 限制到最大输入长度
        input_ids = input_ids[-max_input_tokens:]  # truncate left
        input_ids = torch.LongTensor([input_ids]).to(self.model.device)

        # 生成回复
        pred = self.model.generate(input_ids, generation_config=self.model.generation_config)
        response = tokenizer.decode(pred.cpu()[0][len(input_ids[0]):], skip_special_tokens=True)

        return response


def langchain_chat():
    """
    基于 langchain 的聊天
    """
    global model
    global tokenizer
    if "model" not in globals():
        model, tokenizer = load_baichuan_model()
    llm = BaichuanChatModel(model=model, tokenizer=tokenizer)

    # 开启一个对话
    messages = [
        SystemMessage(content="你是一个很乐于助人的伙伴, 会为大家详细的解答问题, 不只是简单的回答, 而是会详细的解释, 规划每一个解题步骤"),
        HumanMessage(content="矩阵乘法是什么?")
    ]
    ai_response = llm.invoke(messages)
    print(ai_response.content)
    print("---------------")

    # 接着对话
    messages.extend([
        ai_response,
        HumanMessage(content="那么应该怎么计算呢?")
    ])
    ai_response = llm.invoke(messages)
    print(ai_response.content)

    return llm


if __name__ == "__main__":
    # chatbot = pipeline_chat()
    llm = langchain_chat()
