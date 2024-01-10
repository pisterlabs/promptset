from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

class CustomLLM(LLM):

    import langchain
    llm: langchain.llms.openai.OpenAI

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        reply = self.llm(prompt)

        f = open('calls.log', 'a')

        f.write('<request>\n')
        f.write(prompt)
        f.write('</request>\n')

        f.write('<response>\n')
        f.write(reply)
        f.write('</response>\n')
        f.close()

        return reply

    #@property
    #def _identifying_params(self) -> Mapping[str, Any]:
    #    """Get the identifying parameters."""
    #    return {"n": self.n}
    
llm = OpenAI(temperature=0,streaming=True)
wllm = CustomLLM(llm = llm)
chat_llm = ChatOpenAI(temperature=0,streaming=True)

