from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel

model_path = "D:\\models\\"
llm_model_name = model_path + "chatglm-6b"

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
llm_model = AutoModel.from_pretrained(llm_model_name, trust_remote_code=True).float()


class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.8
    top_p = 0.9
    history = []

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response, history = llm_model.chat(llm_tokenizer, prompt, history=[],
                                           max_length=self.max_token,
                                           temperature=self.temperature,
                                           top_p=self.top_p)
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"temperature": self.temperature}


from langchain.prompts import PromptTemplate
prompt = PromptTemplate(input_variables=['product'],
                        template="What is the creative name for a store makes {product}?")
from langchain.chains import LLMChain, SimpleSequentialChain

llm = ChatGLM()
chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run('Electric Vehicle'))

second_prompt = PromptTemplate(input_variables=['company_name'],
                        template="Write a slogan for company {company_name}?")
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
slogan = overall_chain.run('Electric Vehicle')
print(slogan)

