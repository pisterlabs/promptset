import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# model_id = 'llm-jp/llm-jp-13b-v1.0'
# model_name = 'llmjp'
# model_id = "stabilityai/japanese-stablelm-base-ja_vocab-beta-7b"
# model_name = 'stablelm'
# model_id = "stabilityai/japanese-stablelm-3b-4e1t-base"
# model_name = 'stablelm-3b'
model_id = "stabilityai/japanese-stablelm-base-gamma-7b"
model_name = 'stablelm'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name + '-tokenizer',
    local_files_only=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name + '-model',
    local_files_only=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10
)
hf = HuggingFacePipeline(pipeline=pipe)

print("Encoding input...")
instruction = "あなたは日本語を母国語としています。あなたの仕事は、ユーザーが提出した日本語の文を自然で正確に修正して送信することです。"

template = """
指示に応じてモデルにメッセージを送信するための標準化されたメッセージを生成します。指示、オプションの入力、および 「応答:」 フィールドを含みます。

以下はタスクを説明する指示と、さらなる文脈を提供する入力がペアになっています。
要求を適切に完了する応答を書いてください。

### 指示:
あなたは日本語を母国語としています。あなたの仕事は、ユーザーが提出した日本語の文を自然で正確に修正して送信することです。


### 入力:
{question}


以下はタスクを説明する指示です。
要求を適切に完了する応答を書いてください。

### 応答:
"""
prompt = PromptTemplate.from_template(template)
chain = prompt | hf

print("Generating output...")
text = "自然言語処理とは何か"
print(chain.invoke({"question": text}))

