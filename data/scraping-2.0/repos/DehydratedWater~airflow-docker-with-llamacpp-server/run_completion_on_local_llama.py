from langchain.chat_models import ChatOpenAI


model = "models/llama-2-7b.Q5_K_M.gguf"
model = "models/mixtral-8x7b-v0.1.Q5_K_M.gguf"

llm = ChatOpenAI(temperature=0.7,
                model=model, 
                openai_api_base="http://0.0.0.0:5556/v1", 
                openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                max_tokens=2000,
                model_kwargs={
                    "logit_bias": {},
                    # "stop": ["[/INST]"],
                    # "max_tokens": 3000
                },
                streaming=True,
                )

# print(llm.predict("The poem about cat: "))

for chunk in llm.stream("Write 10 point list about cats: "):
    print(chunk.content, end="", flush=True)