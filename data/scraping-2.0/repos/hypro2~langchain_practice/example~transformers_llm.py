
from functools import partial
from threading import Thread
from typing import List, Mapping, Optional, Any

from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import BaseOutputParser
from pydantic import Field
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from transformers import AutoTokenizer


class TransformersLLM(LLM):
    """
    generator를 정의해서 사용 가능하게 끔 코드 생성
    """
    model_folder_path: str = Field(None, alias='model_folder_path')
    model_name: str = Field(None, alias='model_name')
    backend: Optional[str] = 'llama'
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 200
    repetition_penalty: Optional[float] = 1.15
    ## 추가 ##
    model: Any = None
    tokenizer: Any = None
    #########

    def __init__(self, model_folder_path,callbacks, **kwargs):
        super(TransformersLLM, self).__init__()
        self.model_folder_path: str = model_folder_path
        self.callbacks = callbacks
        ## 추가 ##
        self.model = AutoModelForCausalLM.from_pretrained(self.model_folder_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder_path, use_fast=False)
        #########


    @property
    def _get_model_default_parameters(self):
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetition_penalty" : self.repetition_penalty,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            'model_name': self.model_name,
            'model_path': self.model_folder_path,
            'model_parameters': self._get_model_default_parameters
        }

    @property
    def _llm_type(self) -> str:
        return 'llama'

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs) -> str:

        params = {
            **self._get_model_default_parameters,
            **kwargs
        }

        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)

        ## 추가 ##
        input_ids = self.tokenizer(prompt, max_length=4096, truncation=True, return_tensors='pt').input_ids.cuda()
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(input_ids=input_ids,
                               max_new_tokens=params['max_tokens'],
                               temperature=params['temperature'],
                               repetition_penalty=params['repetition_penalty'],
                               do_sample=True,
                               streamer=streamer)

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        # token별 스트리밍 생성해서, callback을 사용할 수 있게 수정
        response = ""
        for i, new_text in enumerate(streamer):
            if text_callback:
                text_callback(new_text)
            response += new_text
            if response.endswith(new_text * 5) and (new_text != ""):
                break

        if stop:
            response = enforce_stop_tokens(response, stop)
        #########
        return response


### TEST ###

class CustomSpaceSeparatedListOutputParser(BaseOutputParser):

    def parse(self, text: str):
        return text.strip().split(" ")

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of space separated values, "
            "eg: `foo bar baz`"
        )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _type(self) -> str:
        return "space-separated-list"


if __name__=="__main__":
    model_name = "facebook/opt-125m"
    model = TransformersLLM(model_name, callbacks=([StreamingStdOutCallbackHandler()]))
    text = "Explain what are Deep Neural Networks in 2-3 sentences"
    prompt = PromptTemplate(
        template="Answer the user query.\n\n{query}\n",
        input_variables=["query"],
    )
    parser = CustomSpaceSeparatedListOutputParser()
    prompt_and_model = prompt | model | parser
    output = prompt_and_model.invoke({"query": text})
    print(output)


