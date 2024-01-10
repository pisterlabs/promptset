import openai
openai.api_base = "http://localhost:4891/v1"
openai.api_key = "not needed for a local LLM"

# model = "gpt-3.5-turbo"
# model = "mpt-7b-chat"
# model = "gpt4all-j-v1.3-groovy"
model = 'gpt4all-l13b-snoozy' # 已经下载了

# NOTE: 首先要启动gpt4all客户端, 然后进入server模式

prompt = "Who is Michael Jordan?"
# Make the API request
response = openai.Completion.create(
    model=model,
    prompt=prompt,
    max_tokens=50,
    temperature=0.28,
    top_p=0.95,
    n=1,
    echo=True,
    stream=False
)


