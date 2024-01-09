from langchain.document_loaders import YoutubeLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate

loader = YoutubeLoader(video_id="tkqD9W5U9F4")
transcript = loader.load()[0]
print(transcript)
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    request_timeout=30,
    max_retries=1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
template = """
Here is a transcript of a Youtube video:
```
{transcript}
```
请翻译成中文
"""
prompt_template = HumanMessagePromptTemplate.from_template(template)
messages = [prompt_template.format(transcript=transcript)]
llm(messages)
